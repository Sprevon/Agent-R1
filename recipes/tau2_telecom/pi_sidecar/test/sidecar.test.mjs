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

test("Pi sidecar validates the official runtime contract before starting a session", async () => {
  const child = spawn(process.execPath, [entrypoint], {
    cwd: sidecarDir,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const nextEvent = async () => {
    const [line] = await once(lines, "line");
    return JSON.parse(line);
  };

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
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
      await Promise.race([once(child, "exit"), new Promise((resolve) => setTimeout(resolve, 2000))]);
    }
  }
});
