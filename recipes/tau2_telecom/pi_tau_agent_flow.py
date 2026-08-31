from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.env.base import Action
from recipes.tau2_telecom.pi_sidecar_client import PiSidecarError, get_shared_pi_sidecar
from recipes.tau2_telecom.pi_tau_env import PiTauEnv, _as_bool
from verl.experimental.agent_loop.tool_parser import ToolParser

logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected a JSON object, got {type(parsed).__name__}")
        return parsed
    if hasattr(value, "item"):
        return _as_dict(value.item())
    raise TypeError(f"Expected a dict, got {type(value).__name__}")


def _metric_add(metrics: dict[str, float], key: str, started_at: float) -> None:
    metrics[key] = metrics.get(key, 0.0) + perf_counter() - started_at


def _extract_runtime_info(info: dict[str, Any]) -> dict[str, Any]:
    reward_info = info.get("reward_info", {})
    if isinstance(reward_info, str):
        try:
            reward_info = json.loads(reward_info)
        except json.JSONDecodeError:
            reward_info = {"raw": reward_info}
    simulation_run = info.get("simulation_run", {})
    if isinstance(simulation_run, str):
        try:
            simulation_run = json.loads(simulation_run)
        except json.JSONDecodeError:
            simulation_run = {"raw": simulation_run}
    return {
        "reward_info": reward_info,
        "simulation_run": simulation_run,
        "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)),
    }


@register("pi_tau_telecom_agent")
class PiTauAgentFlow(AgentFlowBase):
    """Drive tau2 Telecom through the real Pi agent-core runtime."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        recipe_dir = Path(__file__).resolve().parent
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_steps = int(kwargs.get("max_steps", 30))
        self.event_timeout = float(kwargs.get("event_timeout", 300.0))
        self.node_binary = str(kwargs.get("node_binary", os.getenv("PI_NODE_BINARY", "node")))
        self.sidecar_entrypoint = str(
            kwargs.get(
                "sidecar_entrypoint",
                os.getenv("PI_TAU_SIDECAR_ENTRYPOINT", recipe_dir / "pi_sidecar" / "src" / "main.mjs"),
            )
        )
        self.skills_dir = str(kwargs.get("skills_dir", recipe_dir / "skills"))
        self.domain = str(kwargs.get("domain", "telecom"))
        self.tau2_max_steps = int(kwargs.get("tau2_max_steps", 100))
        self.solo_mode = _as_bool(kwargs.get("solo_mode", False))
        self.default_user_llm = kwargs.get("user_llm")
        self.default_user_llm_args = kwargs.get("user_llm_args")
        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra_info = _as_dict(kwargs.get("extra_info"))
        task_id = str(extra_info.get("task_id") or kwargs.get("task_id") or "")
        if not task_id:
            raise ValueError("tau2 Telecom sample is missing extra_info.task_id")
        split = str(extra_info.get("split", "train"))
        trajectory = _as_dict(kwargs.get("_trajectory"))
        rollout_n = int(trajectory.get("rollout_n", extra_info.get("rollout_n", 0)))
        base_seed = int(extra_info.get("seed", kwargs.get("seed", 0)))
        trajectory_seed = base_seed + rollout_n
        user_llm = extra_info.get("user_llm", self.default_user_llm)
        user_llm_args = extra_info.get("user_llm_args", self.default_user_llm_args)

        env = PiTauEnv(
            task_id=task_id,
            domain=self.domain,
            max_steps=self.tau2_max_steps,
            seed=trajectory_seed,
            solo_mode=self.solo_mode,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
        )
        initial_observation = await asyncio.to_thread(env.reset, **kwargs)
        if not (initial_observation.text or "").strip():
            raise RuntimeError("tau2 returned an empty initial observation")

        sidecar = await get_shared_pi_sidecar(
            node_binary=self.node_binary,
            entrypoint=self.sidecar_entrypoint,
        )
        session_id = uuid4().hex
        session = await sidecar.open_session(
            {
                "session_id": session_id,
                "initial_observation": initial_observation.text,
                "domain_policy": env.policy,
                "skills_dir": self.skills_dir,
                "tools": env.tool_schemas,
                "max_turns": self.max_steps,
            }
        )

        metrics: dict[str, float] = {}
        pending_steps: dict[str, dict[str, Any]] = {}
        steps: list[AgentFlowStep] = []
        session_finished = False

        try:
            while not session_finished:
                event = await session.next_event(timeout=self.event_timeout)
                event_type = event.get("type")
                if event_type == "session_started":
                    continue
                if event_type == "generation_request":
                    started_at = perf_counter()
                    generation_id = str(event["generation_id"])
                    messages = event.get("messages")
                    tools = event.get("tools")
                    if not isinstance(messages, list):
                        raise PiSidecarError("generation_request.messages must be a list")
                    prompt_ids = await self.apply_chat_template(messages, tools=tools)
                    if len(prompt_ids) > self.prompt_length:
                        await session.respond_error(
                            str(event["id"]),
                            f"Prompt length {len(prompt_ids)} exceeds configured limit {self.prompt_length}",
                        )
                        raise RuntimeError(
                            f"Pi/tau2 prompt length {len(prompt_ids)} exceeds configured prompt_length "
                            f"{self.prompt_length} for task {task_id}"
                        )

                    output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                    )
                    response_ids = output.token_ids[: self.response_length]
                    decoded_response = await self.loop.run_in_executor(
                        None,
                        lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=True),
                    )
                    parsed_text, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                    if isinstance(parsed_text, str):
                        response_text = parsed_text
                    else:
                        response_text = "" if tool_calls else decoded_response
                    if not tool_calls and not response_text:
                        response_text = decoded_response
                    tool_call_payload = []
                    for tool_call in tool_calls:
                        try:
                            arguments = json.loads(tool_call.arguments)
                        except json.JSONDecodeError:
                            arguments = {"__invalid_json__": tool_call.arguments}
                        tool_call_payload.append(
                            {
                                "id": uuid4().hex,
                                "name": tool_call.name,
                                "arguments": arguments,
                            }
                        )

                    pending_steps[generation_id] = {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "response_logprobs": output.log_probs[: self.response_length] if output.log_probs else None,
                        "routed_experts": (
                            output.routed_experts[: len(prompt_ids) + self.response_length]
                            if output.routed_experts is not None
                            else None
                        ),
                        "anchor_obs": str(event["anchor_obs"]),
                        "response_text": response_text,
                    }
                    await session.respond(
                        str(event["id"]),
                        {
                            "text": response_text,
                            "tool_calls": tool_call_payload,
                            "stop_reason": "toolUse" if tool_call_payload else "stop",
                        },
                    )
                    _metric_add(metrics, "generate_sequences", started_at)
                    continue

                if event_type == "environment_request":
                    started_at = perf_counter()
                    generation_id = str(event["generation_id"])
                    pending = pending_steps.get(generation_id)
                    if pending is None:
                        raise PiSidecarError(f"environment_request references unknown generation {generation_id}")
                    pending["action"] = str(event["action"])
                    try:
                        next_observation, reward, done, info = await env.step(Action(text=str(event["action"])))
                        pending["observation"] = next_observation.text or ""
                        await session.respond(
                            str(event["id"]),
                            {
                                "observation": next_observation.text or "",
                                "reward": reward,
                                "terminated": done,
                                "truncated": bool(info.get("truncated", False)),
                                "info": info,
                            },
                        )
                    except Exception as exc:
                        await session.respond_error(str(event["id"]), str(exc))
                        raise
                    finally:
                        _metric_add(metrics, "tool_calls", started_at)
                    continue

                if event_type == "step_complete":
                    generation_id = str(event["generation_id"])
                    pending = pending_steps.pop(generation_id, None)
                    if pending is None:
                        raise PiSidecarError(f"step_complete references unknown generation {generation_id}")
                    info = event.get("info") if isinstance(event.get("info"), dict) else {}
                    diagnostics = event.get("diagnostics") if isinstance(event.get("diagnostics"), dict) else {}
                    runtime_info = _extract_runtime_info(info)
                    step = AgentFlowStep(
                        prompt_ids=pending["prompt_ids"],
                        response_ids=pending["response_ids"],
                        response_logprobs=pending["response_logprobs"],
                        routed_experts=pending["routed_experts"],
                        reward_score=float(event.get("reward", 0.0)),
                        extra_fields={
                            "anchor_obs": pending["anchor_obs"],
                            "pi_session_id": session_id,
                            "pi_generation_id": generation_id,
                            "tau2_task_id": task_id,
                            "tau2_split": split,
                            "tau2_terminated": bool(event.get("terminated", False)),
                            "tau2_truncated": bool(event.get("truncated", False)),
                            "pi_invalid_action": bool(event.get("invalid_action", False)),
                            "pi_action": pending.get("action", ""),
                            "tau2_observation": pending.get("observation", ""),
                            "pi_diagnostics": diagnostics,
                            "reward_extra_info": {
                                "score": float(event.get("reward", 0.0)),
                                "task_id": task_id,
                                "split": split,
                                "invalid_action": bool(event.get("invalid_action", False)),
                                **diagnostics,
                                **runtime_info,
                            },
                        },
                    )
                    steps.append(await self._postprocess(step, **kwargs))
                    continue

                if event_type == "session_complete":
                    session_finished = True
                    continue
                if event_type in {"session_error", "sidecar_error"}:
                    raise PiSidecarError(str(event.get("error", event)))
                raise PiSidecarError(f"Unknown Pi sidecar event: {event}")
        finally:
            await session.close()
            await env.close()

        if pending_steps:
            raise PiSidecarError(f"Pi session ended with unfinished generations: {sorted(pending_steps)}")
        if not steps:
            raise RuntimeError(f"Pi/tau2 session for task {task_id} produced no trainable steps")
        return AgentFlowOutput(steps=steps, metrics=metrics)
