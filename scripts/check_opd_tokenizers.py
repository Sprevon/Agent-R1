#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from transformers import AutoTokenizer


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact tokenizer compatibility required by Agent-R1 OPD.")
    parser.add_argument("--student", default="Qwen/Qwen3-4B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    student = AutoTokenizer.from_pretrained(args.student, **tokenizer_kwargs)
    teacher = AutoTokenizer.from_pretrained(args.teacher, **tokenizer_kwargs)
    student_vocab = student.get_vocab()
    teacher_vocab = teacher.get_vocab()
    checks = {
        "vocab": student_vocab == teacher_vocab,
        "added_vocab": student.get_added_vocab() == teacher.get_added_vocab(),
        "special_tokens_map": student.special_tokens_map == teacher.special_tokens_map,
        "chat_template": student.chat_template == teacher.chat_template,
    }
    report = {
        "student": args.student,
        "teacher": args.teacher,
        "student_vocab_size": len(student_vocab),
        "teacher_vocab_size": len(teacher_vocab),
        "student_vocab_sha256": _digest(student_vocab),
        "teacher_vocab_sha256": _digest(teacher_vocab),
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    required = {"vocab", "added_vocab", "special_tokens_map"}
    failed = sorted(name for name in required if not checks[name])
    if failed:
        raise SystemExit(f"Agent-R1 OPD tokenizer compatibility: FAIL ({', '.join(failed)})")
    if not checks["chat_template"]:
        print("Warning: chat templates differ; Agent-R1 sends student token IDs directly to the teacher.")
    print("Agent-R1 OPD tokenizer compatibility: PASS")


if __name__ == "__main__":
    main()
