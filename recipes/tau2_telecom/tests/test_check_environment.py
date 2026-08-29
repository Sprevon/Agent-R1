from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_check_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "check_pi_tau_environment.py"
    spec = importlib.util.spec_from_file_location("check_pi_tau_environment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UserSimulatorPrecheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_check_module()

    def test_missing_model_fails(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            failures = self.module._user_simulator_failures()
        self.assertTrue(any("TAU2_USER_LLM is unset" in item for item in failures))

    def test_missing_credentials_fail_without_reading_values(self) -> None:
        env = {"TAU2_USER_LLM": "openai/gpt-4.1-mini"}
        with patch.dict(os.environ, env, clear=True):
            failures = self.module._user_simulator_failures()
        self.assertTrue(any("credentials are missing" in item for item in failures))
        self.assertFalse(any("sk-" in item for item in failures))

    def test_api_key_field_presence_is_enough(self) -> None:
        env = {
            "TAU2_USER_LLM": "openai/local-user",
            "TAU2_USER_LLM_ARGS": '{"api_base":"http://127.0.0.1:8000/v1","api_key":"unused"}',
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(self.module._user_simulator_failures(), [])


if __name__ == "__main__":
    unittest.main()
