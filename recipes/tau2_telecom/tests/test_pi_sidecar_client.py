from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from recipes.tau2_telecom.pi_sidecar_client import PiSidecarClient


class PiSidecarClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_routes_events_and_responses_by_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "fake_sidecar.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    def emit(payload):
                        print(json.dumps(payload), flush=True)

                    emit({"type": "ready", "protocol_version": 1})
                    for line in sys.stdin:
                        message = json.loads(line)
                        if message["type"] == "start_session":
                            session_id = message["session_id"]
                            emit({"type": "session_started", "session_id": session_id})
                            emit({
                                "type": "generation_request",
                                "id": "request-1",
                                "session_id": session_id,
                                "generation_id": "generation-1",
                                "messages": [],
                                "tools": [],
                                "anchor_obs": "{}",
                            })
                        elif message["type"] == "response":
                            emit({
                                "type": "session_complete",
                                "session_id": "session-a",
                                "terminated": True,
                            })
                        elif message["type"] == "close_session":
                            break
                    """
                ),
                encoding="utf-8",
            )
            client = PiSidecarClient(node_binary=sys.executable, entrypoint=str(script))
            session = await client.open_session(
                {
                    "session_id": "session-a",
                    "initial_observation": "hello",
                    "domain_policy": "policy",
                    "skills_dir": temp_dir,
                    "tools": [],
                    "max_turns": 1,
                }
            )
            self.assertEqual((await session.next_event(timeout=2))["type"], "session_started")
            request = await session.next_event(timeout=2)
            self.assertEqual(request["generation_id"], "generation-1")
            await session.respond(request["id"], {"text": "done", "tool_calls": []})
            completed = await session.next_event(timeout=2)
            self.assertTrue(completed["terminated"])
            await session.close()
            if client._reader_task is not None:
                await asyncio.wait_for(client._reader_task, timeout=2)


if __name__ == "__main__":
    unittest.main()

