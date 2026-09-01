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

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, register
from recipes.tau2_telecom.pi_sidecar_client import PiSidecarClient, PiSidecarError
from recipes.tau2_telecom.pi_training_recorder import PiTrainingRecorder
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


@register("pi_tau_telecom_agent")
class PiTauAgentFlow(AgentFlowBase):
    """Use the official Pi loop while Agent-R1 supplies generations and trains."""

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
        self.tau2_root = str(
            kwargs.get("tau2_root", os.getenv("TAU2_BENCH_ROOT", ""))
        )
        self.pi_coding_agent_entrypoint = str(
            kwargs.get("pi_coding_agent_entrypoint", os.getenv("PI_CODING_AGENT_ENTRYPOINT", ""))
        )
        self.training_extension = str(
            kwargs.get(
                "training_extension",
                os.getenv(
                    "PI_TAU_TRAINING_EXTENSION",
                    Path(self.tau2_root) / ".pi" / "extensions" / "agent-r1-training.ts",
                ),
            )
        )
        self.agent_dir = str(kwargs.get("agent_dir", os.getenv("PI_AGENT_DIR", "")))
        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra_info = _as_dict(kwargs.get("extra_info"))
        task_id = str(extra_info.get("task_id") or kwargs.get("task_id") or "")
        if not task_id:
            raise ValueError("tau2 Telecom sample is missing extra_info.task_id")
        if not self.tau2_root:
            raise ValueError("TAU2_BENCH_ROOT/tau2_root is required for the official Pi extension")
        if not self.pi_coding_agent_entrypoint:
            raise ValueError("PI_CODING_AGENT_ENTRYPOINT/pi_coding_agent_entrypoint is required")

        split = str(extra_info.get("split", "train"))
        session_id = uuid4().hex
        sidecar = PiSidecarClient(
            node_binary=self.node_binary,
            entrypoint=self.sidecar_entrypoint,
            env={
                "TAU2_BENCH_ROOT": self.tau2_root,
                "TAU2_TELECOM_TASK_ID": task_id,
                "PI_CODING_AGENT_ENTRYPOINT": self.pi_coding_agent_entrypoint,
            },
        )
        session = None
        recorder = PiTrainingRecorder(session_id=session_id, task_id=task_id, split=split)
        metrics: dict[str, float] = {}
        evaluation: dict[str, Any] | None = None
        session_finished = False

        try:
            session = await sidecar.open_session(
                {
                    "session_id": session_id,
                    "task_id": task_id,
                    "tau2_root": self.tau2_root,
                    "training_extension": self.training_extension,
                    "agent_dir": self.agent_dir or str(Path(self.tau2_root) / ".pi" / "agent"),
                    "pi_coding_agent_entrypoint": self.pi_coding_agent_entrypoint,
                    "max_turns": self.max_steps,
                }
            )

            while not session_finished:
                event = await session.next_event(timeout=self.event_timeout)
                event_type = event.get("type")
                if event_type in {"session_started", "session_info"}:
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
                    response_text = parsed_text if isinstance(parsed_text, str) else ("" if tool_calls else decoded_response)
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

                    recorder.record_generation(
                        generation_id,
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                        routed_experts=(
                            output.routed_experts[: len(prompt_ids) + len(response_ids)]
                            if output.routed_experts is not None
                            else None
                        ),
                        anchor_obs=str(event.get("anchor_obs", "")),
                        response_text=response_text,
                    )
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

                if event_type == "step_complete":
                    generation_id = str(event["generation_id"])
                    diagnostics = event.get("diagnostics") if isinstance(event.get("diagnostics"), dict) else {}
                    recorder.record_turn(
                        generation_id,
                        assistant_message=event.get("assistant_message"),
                        tool_results=event.get("tool_results", []),
                        truncated=bool(event.get("truncated", False)),
                        diagnostics=diagnostics,
                    )
                    continue

                if event_type == "evaluation_result":
                    result = event.get("result")
                    if isinstance(result, dict):
                        evaluation = result
                    else:
                        raise PiSidecarError("evaluation_result.result must be a JSON object")
                    continue

                if event_type == "session_complete":
                    session_finished = True
                    continue
                if event_type in {"session_error", "sidecar_error"}:
                    raise PiSidecarError(str(event.get("error", event)))
                raise PiSidecarError(f"Unknown Pi sidecar event: {event}")
        finally:
            if session is not None:
                await session.close()
            await sidecar.shutdown()

        steps = recorder.finalize(evaluation)
        processed_steps = [await self._postprocess(step, **kwargs) for step in steps]
        return AgentFlowOutput(steps=processed_steps, metrics=metrics)
