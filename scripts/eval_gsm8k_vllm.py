#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def _extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None


def _extract_final_answer(text: str) -> str | None:
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed
    matches = re.findall(r"####\s*([^\n]+)", text)
    return matches[-1].strip() if matches else None


def _normalize_number(value: Any) -> Decimal | None:
    text = str(value).strip()
    text = re.sub(r"^\\text\{(.*)\}$", r"\1", text)
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.strip().rstrip(".")
    fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    try:
        if fraction:
            return Decimal(fraction.group(1)) / Decimal(fraction.group(2))
        return Decimal(text)
    except (InvalidOperation, ZeroDivisionError):
        return None


def _messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise TypeError(f"expected row['prompt'] to be a list, got {type(prompt).__name__}")
    return [dict(message) for message in prompt]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HF checkpoint on full GSM8K with greedy vLLM decoding.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"GSM8K parquet does not exist: {args.data}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists; pass --overwrite to replace it: {args.output}")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dataset = load_dataset("parquet", data_files={"test": str(args.data)}, split="test")
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    chat_template_kwargs: dict[str, Any] = {}
    if args.enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = args.enable_thinking

    prompts: list[str] = []
    rows: list[dict[str, Any]] = []
    for row in dataset:
        row_dict = dict(row)
        messages = _messages_from_row(row_dict)
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        )
        rows.append(row_dict)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=args.max_tokens,
        n=1,
    )
    generations = llm.generate(prompts, sampling_params, use_tqdm=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    missing_final_answer = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, (row, generation) in enumerate(zip(rows, generations, strict=True)):
            response = generation.outputs[0].text
            predicted_text = _extract_final_answer(response)
            if predicted_text is None:
                missing_final_answer += 1
            predicted = _normalize_number(predicted_text) if predicted_text is not None else None
            ground_truth_text = row.get("ground_truth")
            if ground_truth_text is None:
                reward_model = row.get("reward_model") or {}
                ground_truth_text = reward_model.get("ground_truth")
            ground_truth = _normalize_number(ground_truth_text)
            is_correct = predicted is not None and ground_truth is not None and predicted == ground_truth
            correct += int(is_correct)
            record = {
                "index": index,
                "prompt": row.get("prompt"),
                "ground_truth": ground_truth_text,
                "prediction_text": predicted_text,
                "prediction_normalized": str(predicted) if predicted is not None else None,
                "correct": is_correct,
                "response": response,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(rows)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_final_answer": missing_final_answer,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_tokens": args.max_tokens,
        },
        "enable_thinking": args.enable_thinking,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
