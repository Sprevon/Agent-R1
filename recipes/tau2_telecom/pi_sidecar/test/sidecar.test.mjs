import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface } from "node:readline";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const testDir = dirname(fileURLToPath(import.meta.url));
const sidecarDir = join(testDir, "..");
const entrypoint = join(sidecarDir, "src", "main.mjs");
const fakeCodingAgentEntrypoint = join(testDir, "fixtures", "fake-coding-agent.mjs");

function startSidecar(extraEnv = {}) {
  const child = spawn(process.execPath, [entrypoint], {
    cwd: sidecarDir,
    env: { ...process.env, ...extraEnv },
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const events = lines[Symbol.asyncIterator]();
  const nextEvent = async () => {
    const next = await events.next();
    if (next.done) throw new Error("Sidecar stdout closed before the expected event");
    return JSON.parse(next.value);
  };
  return { child, nextEvent };
}

async function stopSidecar(child) {
  if (child.exitCode === null && child.signalCode === null) {
    child.stdin.write(`${JSON.stringify({ type: "shutdown" })}\n`);
    await Promise.race([once(child, "exit"), new Promise((resolve) => setTimeout(resolve, 2000))]);
  }
  if (child.exitCode === null && child.signalCode === null) child.kill("SIGTERM");
}

test("Pi sidecar validates the official runtime contract before starting a session", async () => {
  const { child, nextEvent } = startSidecar();

  try {
    const ready = await nextEvent();
    assert.equal(ready.type, "ready");
    assert.equal(ready.protocol_version, 2);
    assert.equal(ready.pi_runtime, "pi-coding-agent");

    child.stdin.write(
      `${JSON.stringify({ type: "start_session", session_id: "missing-config" })}\n`,
    );
    const failure = await nextEvent();
    assert.equal(failure.type, "session_error");
    assert.match(failure.error, /tau2_root/);

    child.stdin.write(`${JSON.stringify({ type: "shutdown" })}\n`);
    await once(child, "exit");
    assert.equal(child.exitCode, 0);
  } finally {
    await stopSidecar(child);
  }
});

test("Pi sidecar registers its provider and awaits the canonical task prompt", async () => {
  const { child, nextEvent } = startSidecar();
  try {
    assert.equal((await nextEvent()).type, "ready");
    child.stdin.write(
      `${JSON.stringify({
        type: "start_session",
        session_id: "happy-path",
        task_id: "fixture-task",
        tau2_root: testDir,
        pi_coding_agent_entrypoint: fakeCodingAgentEntrypoint,
      })}\n`,
    );

    assert.equal((await nextEvent()).type, "session_started");
    const request = await nextEvent();
    assert.equal(request.type, "generation_request");
    assert.match(request.messages.at(-1).content, /Task ID: fixture-task/);
    child.stdin.write(
      `${JSON.stringify({
        type: "response",
        response_to: request.id,
        ok: true,
        result: { text: "Fixture resolution", stop_reason: "stop" },
      })}\n`,
    );

    const step = await nextEvent();
    assert.equal(step.type, "step_complete");
    assert.equal(step.diagnostics.completed_turns, 1);
    assert.equal((await nextEvent()).type, "evaluation_result");
    const complete = await nextEvent();
    assert.equal(complete.type, "session_complete");
    assert.equal(complete.turns, 1);
  } finally {
    await stopSidecar(child);
  }
});

test("Pi sidecar surfaces training-extension startup failures", async () => {
  const { child, nextEvent } = startSidecar({ FAKE_PI_EXTENSION_ERROR: "1" });
  try {
    assert.equal((await nextEvent()).type, "ready");
    child.stdin.write(
      `${JSON.stringify({
        type: "start_session",
        session_id: "extension-failure",
        task_id: "fixture-task",
        tau2_root: testDir,
        pi_coding_agent_entrypoint: fakeCodingAgentEntrypoint,
      })}\n`,
    );
    assert.equal((await nextEvent()).type, "session_started");
    const failure = await nextEvent();
    assert.equal(failure.type, "session_error");
    assert.match(failure.error, /extension startup failed/i);
    assert.match(failure.error, /fixture extension startup failure/);
  } finally {
    await stopSidecar(child);
  }
});
