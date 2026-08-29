from __future__ import annotations

import unittest

from recipes.tau2_telecom.pi_tau_env import PiTauEnv, _as_bool, _jsonable, tau2_tool_to_function_schema


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


if __name__ == "__main__":
    unittest.main()
