#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "commit": _command("git", "rev-parse", "HEAD", cwd=path),
        "branch": _command("git", "branch", "--show-current", cwd=path),
        "dirty": bool(_command("git", "status", "--porcelain", cwd=path)),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a reproducibility manifest for Pi/tau2 training.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pi_repo", default="../pi")
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-8B")
    parser.add_argument("--task_split", default="train")
    parser.add_argument("--user_simulator", default="openai/gpt-4.1-mini")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    pi_repo = (project_dir / args.pi_repo).resolve()
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "student": args.student,
            "teacher": args.teacher,
            "user_simulator": args.user_simulator,
        },
        "tau2": {"domain": "telecom", "task_split": args.task_split, "version": _package_version("tau2")},
        "repositories": {
            "agent_r1": _git_state(project_dir),
            "pi": _git_state(pi_repo) if (pi_repo / ".git").exists() else {"path": str(pi_repo), "missing": True},
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "node": _command("node", "--version"),
            "npm": _command("npm", "--version"),
            "nvidia_smi": _command(
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "vllm", "verl", "ray", "pandas", "pyarrow")
        },
        "pins": {
            "pi_agent_core": "0.84.4",
            "tau2_git_ref": "v1.0.1",
            "opd_verl_commit": "5779c7c6782733f77ef640f557bea572dfeacc12",
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest -> {output}")


if __name__ == "__main__":
    main()
