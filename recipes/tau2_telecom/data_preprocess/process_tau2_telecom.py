#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_tasks(split: str) -> list[Any]:
    try:
        from tau2.runner import get_tasks
    except ImportError as exc:
        raise SystemExit(
            "tau2-bench is required. Install requirements-tau2.txt in a Python 3.12 or 3.13 environment."
        ) from exc

    try:
        return list(get_tasks("telecom", task_split_name=split))
    except TypeError:
        return list(get_tasks("telecom", split))


def _task_id(task: Any) -> str:
    task_id = getattr(task, "id", None)
    if task_id is None and isinstance(task, dict):
        task_id = task.get("id")
    if task_id is None:
        raise ValueError(f"tau2 task has no id: {task!r}")
    return str(task_id)


def _make_row(task: Any, split: str, index: int, seed: int) -> dict[str, Any]:
    task_id = _task_id(task)
    return {
        "data_source": f"tau2_telecom_{split}",
        "prompt": [{"role": "user", "content": f"Run tau2 Telecom task {task_id}."}],
        "reward_model": {"ground_truth": {"task_id": task_id}, "style": "rule"},
        "extra_info": {
            "index": index,
            "task_id": task_id,
            "split": split,
            "seed": seed + index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tau2 Telecom tasks as Agent-R1 parquet files.")
    parser.add_argument("--output_dir", default="data/tau2_telecom")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"domain": "telecom", "splits": {}}
    for split in args.splits:
        tasks = _load_tasks(split)
        if args.max_tasks > 0:
            tasks = tasks[: args.max_tasks]
        rows = [_make_row(task, split, index, args.seed) for index, task in enumerate(tasks)]
        output_path = output_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        task_ids = [_task_id(task) for task in tasks]
        manifest["splits"][split] = {"count": len(rows), "task_ids": task_ids}
        print(f"[{split}] wrote {len(rows)} rows -> {output_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

