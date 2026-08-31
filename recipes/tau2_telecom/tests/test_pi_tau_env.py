from __future__ import annotations

import json
import os
import tempfile
import unittest

from recipes.tau2_telecom.pi_tau_env import (
    PiTauEnv,
    _as_bool,
    _jsonable,
    _solo_mode_action,
    _task_message,
    tau2_tool_to_function_schema,
)


class FakeModel:
    def model_dump(self, mode: str = "python"):
        del mode
        return {"value": 3}


class FakeTool:
    name = "lookup_account"
    description = "Look up an account."
    parameters = {
        "type": "object",
        "properties": {"account_id": {"type": "string"}},
        "required": ["account_id"],
    }


class PiTauEnvHelpersTest(unittest.TestCase):
    def test_tool_schema_is_preserved_for_pi(self) -> None:
        schema = tau2_tool_to_function_schema(FakeTool())
        self.assertEqual(schema["name"], "lookup_account")
        self.assertEqual(schema["parameters"]["required"], ["account_id"])

    def test_jsonable_handles_pydantic_style_models(self) -> None:
        self.assertEqual(_jsonable({"nested": FakeModel()}), {"nested": {"value": 3}})

    def test_string_boolean_is_not_treated_as_truthy(self) -> None:
        self.assertFalse(_as_bool("false"))
        self.assertTrue(_as_bool("true"))

    def test_user_llm_args_accepts_mapping_or_json(self) -> None:
        from collections.abc import Mapping

        class FrozenArgs(Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        mapped = PiTauEnv(task_id="t1", user_llm_args=FrozenArgs({"temperature": 0.0}))
        self.assertEqual(mapped.user_llm_args, {"temperature": 0.0})
        parsed = PiTauEnv(task_id="t2", user_llm_args='{"temperature": 0.0}')
        self.assertEqual(parsed.user_llm_args, {"temperature": 0.0})

    def test_task_message_uses_user_scenario(self) -> None:
        task = {
            "user_scenario": {
                "persona": "A customer abroad.",
                "instructions": {
                    "reason_for_call": "Mobile data is not working.",
                    "known_info": "Airplane mode is on.",
                    "task_instructions": "Turn off airplane mode and enable roaming.",
                },
            }
        }
        message = _task_message(task)
        self.assertIn("Mobile data is not working.", message)
        self.assertIn("Turn off airplane mode and enable roaming.", message)

    def test_solo_mode_text_is_mapped_to_done_tool(self) -> None:
        self.assertEqual(
            json.loads(_solo_mode_action("I finished the request.")),
            {"name": "done", "arguments": {}},
        )
        tool = '{"name": "get_customer_by_phone", "arguments": {"phone_number": "1"}}'
        self.assertEqual(_solo_mode_action(tool), tool)

    def test_canonical_allowlist_preserves_extension_order(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, ".pi"))
            with open(os.path.join(root, ".pi", "task_tool_allowlists.json"), "w", encoding="utf-8") as handle:
                json.dump({"task": ["second", "first"]}, handle)
            previous = os.environ.get("TAU2_BENCH_ROOT")
            os.environ["TAU2_BENCH_ROOT"] = root
            try:
                env = PiTauEnv(task_id="task", solo_mode=True)
                self.assertEqual(env._task_tool_allowlist(), ["second", "first"])
            finally:
                if previous is None:
                    os.environ.pop("TAU2_BENCH_ROOT", None)
                else:
                    os.environ["TAU2_BENCH_ROOT"] = previous


if __name__ == "__main__":
    unittest.main()
