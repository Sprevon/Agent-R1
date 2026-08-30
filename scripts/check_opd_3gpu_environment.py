#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _gib(value: int) -> float:
    return value / (1024**3)


def _existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _system_memory_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return _gib(pages * page_size)


def _package_report() -> tuple[dict[str, Any], list[str]]:
    packages: dict[str, Any] = {}
    failures: list[str] = []
    for module_name in ("torch", "vllm", "ray", "transformers", "hydra", "omegaconf"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report the exact runtime import failure
            packages[module_name] = {"ok": False, "error": repr(exc)}
            failures.append(f"cannot import {module_name}: {exc}")
            continue
        packages[module_name] = {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "file": getattr(module, "__file__", None),
        }

    try:
        import verl
        from verl.experimental.teacher_loop import MultiTeacherModelManager  # noqa: F401
        from verl.trainer.distillation import distillation_ppo_loss  # noqa: F401
        from verl.trainer.ppo.utils import Role

        version = getattr(verl, "__version__", "unknown")
        packages["verl"] = {
            "ok": True,
            "version": version,
            "file": getattr(verl, "__file__", None),
            "has_teacher_role": hasattr(Role, "TeacherModel"),
        }
        if "agentr1.opd" not in version:
            failures.append(f"verl version does not identify the Agent-R1 OPD fork: {version}")
        if not hasattr(Role, "TeacherModel"):
            failures.append("verl Role.TeacherModel is missing")
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic entrypoint
        packages["verl"] = {"ok": False, "error": repr(exc)}
        failures.append(f"Agent-R1 OPD verl APIs cannot be imported: {exc}")
    return packages, failures


def _query_gpus() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return [], repr(exc)

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",", maxsplit=2)]
        if len(fields) != 3:
            return [], f"unexpected nvidia-smi row: {raw_line!r}"
        index, name, memory_mib = fields
        gpus.append({"index": index, "name": name, "memory_mib": int(memory_mib)})
    return gpus, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Agent-R1 OPD runtime before renting a full run.")
    parser.add_argument("--expected-gpus", type=int, default=3)
    parser.add_argument("--minimum-memory-mib", type=int, default=90_000)
    parser.add_argument("--minimum-system-memory-gib", type=float, default=180.0)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=350.0)
    parser.add_argument("--storage-path", type=Path, default=Path("checkpoints"))
    parser.add_argument("--skip-gpu-check", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    packages, package_failures = _package_report()
    failures.extend(package_failures)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_entries = [] if not visible_devices else [item.strip() for item in visible_devices.split(",") if item.strip()]
    gpus, gpu_error = _query_gpus()
    if not args.skip_gpu_check:
        if gpu_error:
            failures.append(f"cannot query GPUs: {gpu_error}")
        if visible_entries and len(visible_entries) != args.expected_gpus:
            failures.append(
                f"CUDA_VISIBLE_DEVICES exposes {len(visible_entries)} devices, expected {args.expected_gpus}: "
                f"{visible_devices}"
            )
        if len(gpus) < args.expected_gpus:
            failures.append(f"nvidia-smi reports {len(gpus)} GPUs, expected at least {args.expected_gpus}")
        for gpu in gpus[: args.expected_gpus]:
            if gpu["memory_mib"] < args.minimum_memory_mib:
                failures.append(
                    f"GPU {gpu['index']} has {gpu['memory_mib']} MiB, below {args.minimum_memory_mib} MiB"
                )

    system_memory_gib = _system_memory_gib()
    if system_memory_gib is None:
        warnings.append("could not determine system memory")
    elif system_memory_gib < args.minimum_system_memory_gib:
        warnings.append(
            f"system memory is {system_memory_gib:.1f} GiB; CPU offload is safer with at least "
            f"{args.minimum_system_memory_gib:.1f} GiB"
        )

    storage_root = _existing_path(args.storage_path)
    disk = shutil.disk_usage(storage_root)
    free_disk_gib = _gib(disk.free)
    if free_disk_gib < args.minimum_free_disk_gib:
        warnings.append(
            f"free disk under {storage_root} is {free_disk_gib:.1f} GiB; plan calls for at least "
            f"{args.minimum_free_disk_gib:.1f} GiB"
        )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "packages": packages,
        "cuda_visible_devices": visible_devices,
        "gpus": gpus,
        "system_memory_gib": system_memory_gib,
        "storage": {"checked_path": str(storage_root), "free_gib": free_disk_gib},
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Agent-R1 OPD environment check: FAIL")
    print("Agent-R1 OPD environment check: PASS")


if __name__ == "__main__":
    main()
