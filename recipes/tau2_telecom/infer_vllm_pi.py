#!/usr/bin/env python3
"""Run Qwen3 inference through vLLM + the Pi sidecar on tau2 Telecom tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from transformers import AutoTokenizer

from agent_r1.env.base import Action
from recipes.tau2_telecom.pi_sidecar_client import PiSidecarClient, PiSidecarError
from recipes.tau2_telecom.pi_tau_env import PiTauEnv, _as_bool

HERMES_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_args() -> argparse.Namespace:
    recipe_dir = Path(__file__).resolve().parent
    project_dir = recipe_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("STUDENT_MODEL", "Qwen/Qwen3-0.6B"))
    parser.add_argument(
        "--data",
        nargs="+",
        default=[
            str(project_dir / "data/tau2_telecom_smoke/train.parquet"),
            str(project_dir / "data/tau2_telecom_smoke/test.parquet"),
        ],
    )
    parser.add_argument("--output_dir", default="artifacts/tau2_telecom_infer")
    parser.add_argument("--max_tasks", type=int, default=2)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_prompt_len", type=int, default=12288)
    parser.add_argument("--max_model_len", type=int, default=12800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--solo_mode", default="true")
    parser.add_argument("--node_binary", default=os.getenv("PI_NODE_BINARY", "node"))
    parser.add_argument("--sidecar", default=str(recipe_dir / "pi_sidecar" / "src" / "main.mjs"))
    parser.add_argument("--skills_dir", default=str(recipe_dir / "skills"))
    parser.add_argument("--event_timeout", type=float, default=180.0)
    return parser.parse_args()


def _load_rows(paths: list[str], max_tasks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            extra = row.get("extra_info", {})
            if hasattr(extra, "item"):
                extra = extra.item()
            if not isinstance(extra, dict):
                extra = json.loads(extra) if extra else {}
            rows.append(
                {
                    "task_id": str(extra.get("task_id") or ""),
                    "split": str(extra.get("split") or Path(path).stem),
                    "seed": int(extra.get("seed", 0)),
                    "source": path,
                }
            )
            if 0 < max_tasks <= len(rows):
                return rows
    return rows


def _parse_hermes(text: str) -> tuple[str, list[dict[str, Any]]]:
    tool_calls: list[dict[str, Any]] = []
    for raw in HERMES_TOOL_CALL.findall(text):
        payload = raw.strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            tool_calls.append({"id": uuid4().hex, "name": "unknown", "arguments": {"__invalid_json__": payload}})
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__invalid_json__": arguments}
        tool_calls.append(
            {
                "id": uuid4().hex,
                "name": str(parsed.get("name") or "unknown"),
                "arguments": arguments if isinstance(arguments, dict) else {"value": arguments},
            }
        )
    leftover = HERMES_TOOL_CALL.sub("", text).strip()
    return leftover, tool_calls


def _print_step(task_id: str, step: dict[str, Any]) -> None:
    print(f"\n--- {task_id} turn {step['turn']} ---")
    print("decoded:")
    print(step.get("decoded") or "")
    if step.get("tool_calls"):
        print("parsed_tool_calls:", json.dumps(step["tool_calls"], ensure_ascii=False))
    if step.get("env_action"):
        print("env_action:", step["env_action"])
    if step.get("observation"):
        print("observation:", step["observation"][:800])
    print(f"reward={step.get('reward')} done={step.get('done')} invalid={step.get('invalid_action')}")


async def _run_task(
    *,
    row: dict[str, Any],
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    sidecar: PiSidecarClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = PiTauEnv(
        task_id=row["task_id"],
        domain="telecom",
        max_steps=max(args.max_turns * 2, 12),
        seed=row["seed"],
        solo_mode=_as_bool(args.solo_mode),
    )
    observation = await asyncio.to_thread(env.reset)
    session_id = uuid4().hex
    session = await sidecar.open_session(
        {
            "session_id": session_id,
            "initial_observation": observation.text,
            "domain_policy": env.policy,
            "skills_dir": args.skills_dir,
            "tools": env.tool_schemas,
            "max_turns": args.max_turns,
        }
    )
    pending: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    error: str | None = None
    try:
        while True:
            event = await session.next_event(timeout=args.event_timeout)
            event_type = event.get("type")
            if event_type == "session_started":
                continue
            if event_type == "generation_request":
                messages = event["messages"]
                tools = event.get("tools")
                try:
                    prompt_ids = tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt_ids = tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                if len(prompt_ids) > args.max_prompt_len:
                    raise RuntimeError(
                        f"prompt length {len(prompt_ids)} exceeds --max_prompt_len {args.max_prompt_len}"
                    )
                try:
                    from vllm.inputs import TokensPrompt

                    prompts = [TokensPrompt(prompt_token_ids=prompt_ids)]
                except Exception:
                    prompts = [{"prompt_token_ids": prompt_ids}]
                outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
                token_ids = list(outputs[0].outputs[0].token_ids)
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                leftover, tool_calls = _parse_hermes(decoded)
                response_text = leftover if leftover else ("" if tool_calls else decoded)
                if not tool_calls and not response_text:
                    response_text = decoded
                pending[str(event["generation_id"])] = {
                    "turn": len(steps) + 1,
                    "prompt_tokens": len(prompt_ids),
                    "decoded": decoded,
                    "response_text": response_text,
                    "tool_calls": tool_calls,
                }
                await session.respond(
                    str(event["id"]),
                    {
                        "text": response_text,
                        "tool_calls": tool_calls,
                        "stop_reason": "toolUse" if tool_calls else "stop",
                    },
                )
                continue
            if event_type == "environment_request":
                generation_id = str(event["generation_id"])
                current = pending.get(generation_id)
                if current is None:
                    raise PiSidecarError(f"unknown generation {generation_id}")
                current["env_action"] = str(event["action"])
                next_obs, reward, done, info = await env.step(Action(text=str(event["action"])))
                current["observation"] = next_obs.text or ""
                current["reward"] = reward
                current["done"] = done
                await session.respond(
                    str(event["id"]),
                    {
                        "observation": next_obs.text or "",
                        "reward": reward,
                        "terminated": done,
                        "truncated": bool(info.get("truncated", False)),
                        "info": info,
                    },
                )
                continue
            if event_type == "step_complete":
                generation_id = str(event["generation_id"])
                current = pending.pop(generation_id, {})
                current.update(
                    {
                        "reward": float(event.get("reward", current.get("reward", 0.0))),
                        "done": bool(event.get("terminated") or event.get("truncated")),
                        "invalid_action": bool(event.get("invalid_action", False)),
                        "diagnostics": event.get("diagnostics") or {},
                    }
                )
                steps.append(current)
                _print_step(row["task_id"], current)
                continue
            if event_type == "session_complete":
                break
            if event_type in {"session_error", "sidecar_error"}:
                error = str(event.get("error", event))
                break
            raise PiSidecarError(f"unknown sidecar event: {event}")
    finally:
        await session.close()
        await env.close()

    final_reward = steps[-1]["reward"] if steps else 0.0
    return {
        "task_id": row["task_id"],
        "split": row["split"],
        "seed": row["seed"],
        "error": error,
        "n_steps": len(steps),
        "final_reward": final_reward,
        "initial_observation": observation.text,
        "steps": steps,
    }


async def _amain(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    rows = _load_rows(args.data, args.max_tasks)
    if not rows:
        raise SystemExit("no tasks loaded")
    print(f"loaded {len(rows)} tasks from {args.data}")
    print(f"model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=4,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    sidecar = PiSidecarClient(node_binary=args.node_binary, entrypoint=args.sidecar)
    await sidecar.start()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    results: list[dict[str, Any]] = []
    try:
        with traces_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                print(f"\n==== task {index}/{len(rows)} {row['task_id']} ====")
                trace = await _run_task(
                    row=row,
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    sidecar=sidecar,
                    args=args,
                )
                results.append(
                    {
                        "task_id": trace["task_id"],
                        "split": trace["split"],
                        "n_steps": trace["n_steps"],
                        "final_reward": trace["final_reward"],
                        "error": trace["error"],
                    }
                )
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"task_done reward={trace['final_reward']} steps={trace['n_steps']} error={trace['error']}"
                )
    finally:
        await sidecar.shutdown()

    summary_path = output_dir / "summary.json"
    n_success = sum(1 for item in results if item["final_reward"] and not item["error"])
    summary = {
        "model": args.model,
        "n_tasks": len(results),
        "n_reward_positive": n_success,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {traces_path}")
    print(f"wrote {summary_path}")
    print(json.dumps({"n_tasks": len(results), "n_reward_positive": n_success}, ensure_ascii=False))


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
