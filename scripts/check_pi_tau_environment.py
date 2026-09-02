#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _user_simulator_failures() -> list[str]:
    """Fail before GPU launch if tau2's user simulator cannot be configured.

    Credential values are never printed. This only checks whether the required
    environment keys or JSON fields exist.
    """
    failures: list[str] = []
    user_llm = os.environ.get("TAU2_USER_LLM", "").strip()
    if not user_llm:
        failures.append(
            "TAU2_USER_LLM is unset. The user simulator is a separate OpenAI-compatible dependency "
            "and is not the OPD teacher."
        )
        return failures

    raw_args = os.environ.get("TAU2_USER_LLM_ARGS", "{}")
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        failures.append("TAU2_USER_LLM_ARGS must be valid JSON")
        return failures
    if not isinstance(args, dict):
        failures.append("TAU2_USER_LLM_ARGS must be a JSON object")
        return failures

    has_inline_key = "api_key" in args
    has_env_key = "OPENAI_API_KEY" in os.environ
    if not has_inline_key and not has_env_key:
        failures.append(
            "tau2 user simulator credentials are missing. Set OPENAI_API_KEY, or include an api_key "
            "field in TAU2_USER_LLM_ARGS, before starting a GPU run."
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Pi/tau2/OPD runtime before a GPU smoke test.")
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-8B")
    parser.add_argument("--skip_tokenizers", action="store_true")
    parser.add_argument("--skip_user_simulator", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    solo_mode = os.environ.get("TAU2_SOLO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if not args.skip_user_simulator and not solo_mode:
        failures.extend(_user_simulator_failures())
    python_supported = (3, 12) <= sys.version_info[:2] < (3, 14)
    experimental_py310 = sys.version_info[:2] == (3, 10)
    node = shutil.which("node")
    if node is None:
        failures.append("node is not on PATH")
    else:
        version = subprocess.check_output([node, "--version"], text=True).strip()
        major, minor, *_ = (int(part) for part in version.removeprefix("v").split("."))
        if (major, minor) < (22, 19):
            failures.append(f"Node >=22.19.0 is required; found {version}")

    imports = [
        ("tau2.gym.gym_agent", "AgentGymEnv"),
        ("verl.experimental.teacher_loop", "MultiTeacherModelManager"),
        ("verl.trainer.distillation", "distillation_ppo_loss"),
    ]
    tau2_import_ok = False
    for module_name, attribute in imports:
        try:
            module = __import__(module_name, fromlist=[attribute])
            getattr(module, attribute)
            if module_name.startswith("tau2."):
                tau2_import_ok = True
        except Exception as exc:
            failures.append(f"{module_name}.{attribute}: {exc}")
    if not python_supported:
        if experimental_py310 and tau2_import_ok:
            print(
                f"Python {sys.version.split()[0]} is outside tau2's declared range; "
                "continuing because AgentGymEnv imported successfully.",
                file=sys.stderr,
            )
        else:
            failures.append(
                f"Python 3.12 or 3.13 is required by tau2 1.0.1; found {sys.version.split()[0]}"
            )

    sidecar_dir = Path(__file__).resolve().parents[1] / "recipes" / "tau2_telecom" / "pi_sidecar"
    if not (sidecar_dir / "node_modules" / "@earendil-works" / "pi-agent-core").exists():
        failures.append(f"Pi sidecar dependencies are missing under {sidecar_dir}")
    elif node is not None:
        result = subprocess.run(
            [node, "--check", str(sidecar_dir / "src" / "main.mjs")],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            failures.append(f"Pi sidecar syntax check failed: {result.stderr.strip()}")

    coding_agent_entrypoint = Path(
        os.environ.get(
            "PI_CODING_AGENT_ENTRYPOINT",
            "/root/autodl-tmp/code/pi/packages/coding-agent/dist/index.js",
        )
    )
    if not coding_agent_entrypoint.is_file():
        failures.append(
            f"PI_CODING_AGENT_ENTRYPOINT does not exist: {coding_agent_entrypoint}"
        )
    elif node is not None:
        sdk_check = subprocess.run(
            [
                node,
                "--input-type=module",
                "--eval",
                (
                    "const sdk = await import(process.argv[1]);"
                    "const required = ['createAgentSession', 'DefaultResourceLoader', "
                    "'SessionManager', 'ModelRuntime'];"
                    "const missing = required.filter((name) => typeof sdk[name] === 'undefined');"
                    "if (missing.length) { console.error(missing.join(',')); process.exit(1); }"
                ),
                coding_agent_entrypoint.resolve().as_uri(),
            ],
            text=True,
            capture_output=True,
        )
        if sdk_check.returncode != 0:
            failures.append(
                "Pi coding-agent SDK exports are incomplete: "
                f"{sdk_check.stderr.strip() or sdk_check.stdout.strip()}"
            )

    try:
        import torch

        gpu_report = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
        print(json.dumps(gpu_report, ensure_ascii=False, indent=2))
    except Exception as exc:
        failures.append(f"torch/CUDA check failed: {exc}")

    if not args.skip_tokenizers and not failures:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("check_opd_tokenizers.py")),
                "--student",
                args.student,
                "--teacher",
                args.teacher,
            ]
        )
        if result.returncode != 0:
            failures.append("student/teacher tokenizer compatibility failed")

    if failures:
        print("Pi/tau2 environment check: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Pi/tau2 environment check: PASS")


if __name__ == "__main__":
    main()
