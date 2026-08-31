from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from collections.abc import Mapping
from typing import Any

from agent_r1.env import AgentEnv
from agent_r1.env.base import Action, Observation


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise TypeError(f"Expected a boolean value, got {value!r}")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


def _read_schema_candidate(tool: Any, name: str) -> Any:
    candidate = getattr(tool, name, None)
    if callable(candidate):
        try:
            return candidate()
        except TypeError:
            return None
    return candidate


def _task_message(task: Any) -> str:
    """Build the single-turn task message used in solo_mode."""
    if task is None:
        return ""
    scenario = getattr(task, "user_scenario", None)
    if scenario is None and isinstance(task, Mapping):
        scenario = task.get("user_scenario")
    if scenario is None:
        return ""
    if not isinstance(scenario, Mapping):
        return str(scenario).strip()

    parts: list[str] = []
    persona = scenario.get("persona")
    if persona:
        parts.append(str(persona).strip())
    instructions = scenario.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        parts.append(instructions.strip())
    elif isinstance(instructions, Mapping):
        for key in ("reason_for_call", "known_info", "task_instructions"):
            value = instructions.get(key)
            if value:
                parts.append(str(value).strip())
    elif instructions is not None:
        parts.append(str(instructions).strip())
    return "\n\n".join(part for part in parts if part)


def _solo_mode_action(text: str) -> str:
    """Map non-tool text to tau2's `done` tool so DummyUser is never invoked."""
    stripped = text.strip()
    if not stripped:
        return json.dumps({"name": "done", "arguments": {}})
    if stripped.startswith("{") or stripped.startswith("done("):
        return text
    return json.dumps({"name": "done", "arguments": {}})


def tau2_tool_to_function_schema(tool: Any) -> dict[str, Any]:
    """Convert a tau2 Tool into the function schema consumed by Pi and Qwen."""
    for attribute in ("openai_schema", "function_schema", "schema"):
        candidate = _read_schema_candidate(tool, attribute)
        if candidate:
            data = _jsonable(candidate)
            if isinstance(data, dict) and data.get("type") == "function":
                data = data.get("function")
            if isinstance(data, dict) and "name" in data:
                return {
                    "name": str(data["name"]),
                    "description": str(data.get("description", "")),
                    "parameters": data.get("parameters") or {"type": "object", "properties": {}},
                }

    data = _jsonable(tool)
    if isinstance(data, dict) and data.get("type") == "function":
        data = data.get("function")
    if isinstance(data, dict) and "name" in data:
        parameters = data.get("parameters") or data.get("input_schema") or data.get("inputSchema")
        return {
            "name": str(data["name"]),
            "description": str(data.get("description", "")),
            "parameters": parameters or {"type": "object", "properties": {}},
        }

    name = getattr(tool, "name", None)
    if not name:
        raise TypeError(f"Cannot extract a function schema from tau2 tool {tool!r}")
    parameters = (
        _read_schema_candidate(tool, "parameters")
        or _read_schema_candidate(tool, "input_schema")
        or _read_schema_candidate(tool, "inputSchema")
        or {"type": "object", "properties": {}}
    )
    return {
        "name": str(name),
        "description": str(getattr(tool, "description", "")),
        "parameters": _jsonable(parameters),
    }


@AgentEnv.register("pi_tau_telecom")
class PiTauEnv(AgentEnv):
    """Thin async wrapper around tau2's public AgentGymEnv."""

    def __init__(
        self,
        *,
        task_id: str,
        domain: str = "telecom",
        max_steps: int = 100,
        seed: int | None = None,
        solo_mode: bool = False,
        user_llm: str | None = None,
        user_llm_args: Mapping[str, Any] | str | None = None,
    ) -> None:
        self.task_id = str(task_id)
        self.domain = domain
        self.max_steps = int(max_steps)
        self.seed = seed
        self.solo_mode = bool(solo_mode)
        self.user_llm = user_llm
        if isinstance(user_llm_args, str):
            user_llm_args = json.loads(user_llm_args)
        elif isinstance(user_llm_args, Mapping):
            user_llm_args = dict(user_llm_args)
        self.user_llm_args = user_llm_args
        self.tool_schemas: list[dict[str, Any]] = []
        self.policy = ""
        self.last_info: dict[str, Any] = {}
        self._env: Any = None

    def _task_tool_allowlist(self) -> list[str] | None:
        """Read the allowlist used by the canonical tau2-bench Pi extension."""
        root = os.environ.get("TAU2_BENCH_ROOT", "").strip()
        if not root:
            return None
        path = os.path.join(root, ".pi", "task_tool_allowlists.json")
        try:
            with open(path, encoding="utf-8") as handle:
                catalog = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read canonical tau2 Pi tool allowlists: {path}") from exc
        names = catalog.get(self.task_id)
        if names is None:
            return None
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Invalid tool allowlist for tau2 task {self.task_id}")
        return list(names)

    def reset(self, **kwargs) -> Observation:
        try:
            from tau2.gym.gym_agent import AgentGymEnv
        except ImportError as exc:
            raise RuntimeError(
                "tau2-bench with the gym extra is required. Install the pinned dependency from "
                "requirements-tau2.txt."
            ) from exc

        self._env = AgentGymEnv(
            domain=self.domain,
            task_id=self.task_id,
            max_steps=self.max_steps,
            solo_mode=self.solo_mode,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            all_messages_as_observation=False,
        )
        observation, info = self._env.reset(seed=self.seed)
        self.last_info = _jsonable(info)
        self.policy = str(info.get("policy", ""))
        # In tau2 solo mode AgentGymEnv exposes both assistant tools and
        # device/user tools through ``info["tools"]``.  Pi's agent must only
        # see the former: device tools are for the simulated user and calling
        # them from the assistant side terminates the episode with
        # ``agent_error``.  Recover the assistant-side names from the freshly
        # constructed domain environment and retain tau2's explicit stop
        # tools, which are injected by GymAgent after environment creation.
        assistant_tool_names: set[str] | None = None
        if self.solo_mode and hasattr(self._env, "_get_environment"):
            assistant_environment = self._env._get_environment()
            assistant_tool_names = {
                str(tool.name) for tool in assistant_environment.get_tools()
            }
            assistant_tool_names.update({"done", "transfer_to_human_agents"})
        schemas = [tau2_tool_to_function_schema(tool) for tool in info.get("tools", [])]
        if assistant_tool_names is not None:
            schemas = [schema for schema in schemas if schema["name"] in assistant_tool_names]
        task_allowlist = self._task_tool_allowlist()
        if task_allowlist is not None:
            available = {schema["name"] for schema in schemas}
            missing = set(task_allowlist) - available
            if missing:
                raise RuntimeError(
                    f"Canonical tau2 Pi tool allowlist references unavailable tools: {sorted(missing)}"
                )
            by_name = {schema["name"]: schema for schema in schemas}
            schemas = [by_name[name] for name in task_allowlist]
        self.tool_schemas = schemas
        text = str(observation or "").strip()
        if not text and self.solo_mode:
            text = _task_message(info.get("task"))
        if not text:
            raise RuntimeError(
                f"tau2 returned an empty initial observation for task {self.task_id}. "
                "solo_mode requires a task message from user_scenario."
            )
        return Observation(text=text)

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("PiTauEnv.reset() must be called before step()")
        if action.text is None:
            raise ValueError("PiTauEnv requires Action.text")
        action_text = _solo_mode_action(action.text) if self.solo_mode else action.text
        observation, reward, terminated, truncated, info = await asyncio.to_thread(self._env.step, action_text)
        normalized_info = _jsonable(info)
        normalized_info["terminated"] = bool(terminated)
        normalized_info["truncated"] = bool(truncated)
        self.last_info = normalized_info
        return Observation(text=str(observation)), float(reward), bool(terminated or truncated), normalized_info

    async def close(self) -> None:
        if self._env is None:
            return
        close = getattr(self._env, "close", None)
        if callable(close):
            await asyncio.to_thread(close)
        self._env = None
