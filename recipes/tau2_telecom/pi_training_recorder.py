from __future__ import annotations

from typing import Any

from agent_r1.agent_flow.agent_flow import AgentFlowStep


class PiTrainingRecorder:
    """Keep model-token data separate from the tau2 execution session.

    The recorder never calls tau2, owns no task/database state, and only turns
    Pi lifecycle events plus Agent-R1 generation outputs into trainable steps.
    Tau2's terminal evaluator payload is attached after the Pi session ends.
    """

    def __init__(self, *, session_id: str, task_id: str, split: str) -> None:
        self.session_id = session_id
        self.task_id = task_id
        self.split = split
        self._pending: dict[str, dict[str, Any]] = {}
        self._steps: list[AgentFlowStep] = []

    def record_generation(
        self,
        generation_id: str,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: Any,
        routed_experts: Any,
        anchor_obs: str,
        response_text: str,
    ) -> None:
        if generation_id in self._pending:
            raise RuntimeError(f"Duplicate Pi generation id: {generation_id}")
        self._pending[generation_id] = {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_logprobs": response_logprobs,
            "routed_experts": routed_experts,
            "anchor_obs": anchor_obs,
            "response_text": response_text,
        }

    def record_turn(
        self,
        generation_id: str,
        *,
        assistant_message: Any,
        tool_results: Any,
        truncated: bool,
        diagnostics: dict[str, Any],
    ) -> None:
        pending = self._pending.pop(generation_id, None)
        if pending is None:
            raise RuntimeError(f"Pi step references unknown generation: {generation_id}")
        self._steps.append(
            AgentFlowStep(
                prompt_ids=pending["prompt_ids"],
                response_ids=pending["response_ids"],
                response_logprobs=pending["response_logprobs"],
                routed_experts=pending["routed_experts"],
                reward_score=0.0,
                extra_fields={
                    "anchor_obs": pending["anchor_obs"],
                    "pi_session_id": self.session_id,
                    "pi_generation_id": generation_id,
                    "tau2_task_id": self.task_id,
                    "tau2_split": self.split,
                    "tau2_truncated": truncated,
                    "pi_response_text": pending["response_text"],
                    "pi_assistant_message": assistant_message,
                    "pi_tool_results": tool_results,
                    "pi_diagnostics": diagnostics,
                },
            )
        )

    def finalize(self, evaluation: dict[str, Any] | None) -> list[AgentFlowStep]:
        if self._pending:
            raise RuntimeError(f"Pi session ended with unfinished generations: {sorted(self._pending)}")
        if not self._steps:
            raise RuntimeError("Pi session produced no trainable turns")

        payload = dict(evaluation or {})
        reward = float(payload.get("reward", 0.0))
        runtime_info = {
            "reward_info": payload.get("reward_info", {}),
            "simulation_run": payload.get("simulation_run", {}),
            "terminated": bool(payload.get("terminated", False)),
            "truncated": bool(payload.get("truncated", False)),
        }
        terminal = self._steps[-1]
        terminal.reward_score = reward
        terminal.extra_fields.update(
            {
                "tau2_evaluation": payload,
                "tau2_terminated": True,
                "tau2_evaluation_reward": reward,
                "reward_extra_info": {
                    "score": reward,
                    "task_id": self.task_id,
                    "split": self.split,
                    **runtime_info,
                },
            }
        )
        return self._steps
