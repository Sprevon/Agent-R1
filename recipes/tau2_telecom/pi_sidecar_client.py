from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PiSidecarError(RuntimeError):
    """Raised when the Pi sidecar exits or violates the JSONL protocol."""


class PiSidecarSession:
    """One multiplexed Pi agent session hosted by a shared sidecar process."""

    def __init__(self, client: PiSidecarClient, session_id: str, queue: asyncio.Queue[dict[str, Any]]):
        self._client = client
        self.session_id = session_id
        self._queue = queue

    async def next_event(self, timeout: float | None = None) -> dict[str, Any]:
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    async def respond(self, request_id: str, result: dict[str, Any]) -> None:
        await self._client.send(
            {
                "type": "response",
                "response_to": request_id,
                "ok": True,
                "result": result,
            }
        )

    async def respond_error(self, request_id: str, message: str) -> None:
        await self._client.send(
            {
                "type": "response",
                "response_to": request_id,
                "ok": False,
                "error": message,
            }
        )

    async def close(self) -> None:
        await self._client.close_session(self.session_id)


class PiSidecarClient:
    """Async JSONL client for the long-lived Pi SDK sidecar."""

    def __init__(
        self,
        *,
        node_binary: str,
        entrypoint: str,
        startup_timeout: float = 30.0,
    ) -> None:
        self.node_binary = node_binary
        self.entrypoint = str(Path(entrypoint).expanduser().resolve())
        self.startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._ready: asyncio.Future[None] | None = None
        self._sessions: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        async with self._start_lock:
            if self._process is not None and self._process.returncode is None:
                return
            self._ready = asyncio.get_running_loop().create_future()
            self._process = await asyncio.create_subprocess_exec(
                self.node_binary,
                self.entrypoint,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
            )
            self._reader_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())
            await asyncio.wait_for(self._ready, timeout=self.startup_timeout)

    async def open_session(self, payload: dict[str, Any]) -> PiSidecarSession:
        await self.start()
        session_id = str(payload["session_id"])
        if session_id in self._sessions:
            raise PiSidecarError(f"Pi session already exists: {session_id}")
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sessions[session_id] = queue
        await self.send({"type": "start_session", **payload})
        return PiSidecarSession(self, session_id, queue)

    async def close_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return
        try:
            await self.send({"type": "close_session", "session_id": session_id})
        finally:
            self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        try:
            if self._process is not None and self._process.returncode is None:
                await self.send({"type": "shutdown"})
                await asyncio.wait_for(self._process.wait(), timeout=5)
        except Exception:
            if self._process is not None and self._process.returncode is None:
                self._process.kill()

    async def send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.returncode is not None or self._process.stdin is None:
            raise PiSidecarError("Pi sidecar is not running")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            self._process.stdin.write(encoded.encode("utf-8"))
            await self._process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while line := await self._process.stdout.readline():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PiSidecarError(f"Invalid Pi sidecar JSON: {line.decode(errors='replace').strip()}") from exc
                if event.get("type") == "ready":
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(None)
                    continue
                session_id = event.get("session_id")
                if session_id in self._sessions:
                    await self._sessions[session_id].put(event)
                else:
                    logger.warning("Dropping Pi sidecar event for unknown session: %s", event)
        except Exception as exc:
            self._fail_all(exc)
        finally:
            if self._process.returncode is None:
                await self._process.wait()
            self._fail_all(PiSidecarError(f"Pi sidecar exited with status {self._process.returncode}"))

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while line := await self._process.stderr.readline():
            logger.warning("pi-sidecar: %s", line.decode(errors="replace").rstrip())

    def _fail_all(self, error: Exception) -> None:
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(error)
        event = {"type": "sidecar_error", "error": str(error)}
        for queue in self._sessions.values():
            queue.put_nowait(event)


_SHARED_CLIENTS: dict[tuple[int, str, str], PiSidecarClient] = {}


async def get_shared_pi_sidecar(
    *,
    node_binary: str,
    entrypoint: str,
    startup_timeout: float = 30.0,
) -> PiSidecarClient:
    """Return one sidecar process per asyncio loop and executable pair."""
    loop = asyncio.get_running_loop()
    key = (id(loop), node_binary, str(Path(entrypoint).expanduser().resolve()))
    client = _SHARED_CLIENTS.get(key)
    if client is None:
        client = PiSidecarClient(
            node_binary=node_binary,
            entrypoint=entrypoint,
            startup_timeout=startup_timeout,
        )
        _SHARED_CLIENTS[key] = client
    await client.start()
    return client
