from __future__ import annotations

import asyncio
import dataclasses
import json
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
        self.tool_schemas = [tau2_tool_to_function_schema(tool) for tool in info.get("tools", [])]
        return Observation(text=str(observation))

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("PiTauEnv.reset() must be called before step()")
        if action.text is None:
            raise ValueError("PiTauEnv requires Action.text")
        observation, reward, terminated, truncated, info = await asyncio.to_thread(self._env.step, action.text)
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
