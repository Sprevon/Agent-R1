import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface } from "node:readline";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const testDir = dirname(fileURLToPath(import.meta.url));
const sidecarDir = resolve(testDir, "..");
const entrypoint = join(sidecarDir, "src", "main.mjs");
const skillsDir = resolve(sidecarDir, "..", "skills");

test("Pi SDK sidecar runs a tool turn and a terminal text turn", async () => {
  const child = spawn(process.execPath, [entrypoint], {
    cwd: sidecarDir,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const events = [];
  const waiters = [];
  lines.on("line", (line) => {
    const event = JSON.parse(line);
    const waiter = waiters.shift();
    if (waiter) waiter(event);
    else events.push(event);
  });
  const nextEvent = async () => {
    if (events.length > 0) return events.shift();
    return await new Promise((resolveWaiter, rejectWaiter) => {
      const timer = setTimeout(() => rejectWaiter(new Error("Timed out waiting for sidecar event")), 5000);
      waiters.push((event) => {
        clearTimeout(timer);
        resolveWaiter(event);
      });
    });
  };
  const send = (payload) => child.stdin.write(`${JSON.stringify(payload)}\n`);

  try {
  assert.equal((await nextEvent()).type, "ready");
  send({
    type: "start_session",
    session_id: "test-session",
    initial_observation: "Please inspect my account.",
    domain_policy: "Use tools before claiming account state.",
    skills_dir: skillsDir,
    max_turns: 3,
    tools: [
      {
        name: "lookup_account",
        description: "Look up an account.",
        parameters: {
          type: "object",
          properties: { account_id: { type: "string" } },
          required: ["account_id"],
        },
      },
    ],
  });
  assert.equal((await nextEvent()).type, "session_started");

  const firstGeneration = await nextEvent();
  assert.equal(firstGeneration.type, "generation_request");
  send({
    type: "response",
    response_to: firstGeneration.id,
    ok: true,
    result: {
      text: "",
      stop_reason: "toolUse",
      tool_calls: [{ id: "call-1", name: "lookup_account", arguments: { account_id: "A-1" } }],
    },
  });

  const toolRequest = await nextEvent();
  assert.equal(toolRequest.type, "environment_request");
  assert.deepEqual(JSON.parse(toolRequest.action), {
    name: "lookup_account",
    arguments: { account_id: "A-1" },
  });
  send({
    type: "response",
    response_to: toolRequest.id,
    ok: true,
    result: { observation: "Account is active.", reward: 0, terminated: false, truncated: false, info: {} },
  });
  assert.equal((await nextEvent()).type, "step_complete");

  const secondGeneration = await nextEvent();
  assert.equal(secondGeneration.type, "generation_request");
  send({
    type: "response",
    response_to: secondGeneration.id,
    ok: true,
    result: { text: "Your account is active.", stop_reason: "stop", tool_calls: [] },
  });
  const textRequest = await nextEvent();
  assert.equal(textRequest.type, "environment_request");
  assert.equal(textRequest.action, "Your account is active.");
  send({
    type: "response",
    response_to: textRequest.id,
    ok: true,
    result: { observation: "###STOP###", reward: 1, terminated: true, truncated: false, info: {} },
  });
  const finalStep = await nextEvent();
  assert.equal(finalStep.type, "step_complete");
  assert.equal(finalStep.reward, 1);
  assert.equal((await nextEvent()).type, "session_complete");

  send({ type: "close_session", session_id: "test-session" });

  send({
    type: "start_session",
    session_id: "invalid-session",
    initial_observation: "Please inspect my account.",
    domain_policy: "Use tools before claiming account state.",
    skills_dir: skillsDir,
    max_turns: 1,
    tools: [
      {
        name: "lookup_account",
        description: "Look up an account.",
        parameters: {
          type: "object",
          properties: { account_id: { type: "string" } },
          required: ["account_id"],
        },
      },
    ],
  });
  assert.equal((await nextEvent()).type, "session_started");
  const invalidGeneration = await nextEvent();
  assert.equal(invalidGeneration.type, "generation_request");
  send({
    type: "response",
    response_to: invalidGeneration.id,
    ok: true,
    result: {
      text: "I will check that now.",
      stop_reason: "toolUse",
      tool_calls: [{ id: "call-2", name: "lookup_account", arguments: { account_id: "A-1" } }],
    },
  });
  const invalidStep = await nextEvent();
  assert.equal(invalidStep.type, "step_complete");
  assert.equal(invalidStep.invalid_action, true);
  assert.equal(invalidStep.truncated, true);
  assert.equal(invalidStep.diagnostics.invalid_tool_calls, 1);
  assert.equal((await nextEvent()).type, "session_complete");
  send({ type: "close_session", session_id: "invalid-session" });

  send({
    type: "start_session",
    session_id: "missing-skills",
    initial_observation: "Please inspect my account.",
    domain_policy: "Use tools before claiming account state.",
    skills_dir: join(testDir, "does-not-exist"),
    max_turns: 1,
    tools: [],
  });
  const missingSkills = await nextEvent();
  assert.equal(missingSkills.type, "session_error");
  assert.match(String(missingSkills.error), /no skills/);

  send({ type: "shutdown" });
  await once(child, "exit");
  assert.equal(child.exitCode, 0);
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
      await Promise.race([once(child, "exit"), new Promise((resolve) => setTimeout(resolve, 2000))]);
    }
  }
});
