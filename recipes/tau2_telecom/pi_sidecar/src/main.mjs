import { createInterface } from "node:readline";
import {
  Agent,
  formatSkillInvocation,
  formatSkillsForSystemPrompt,
  loadSkills,
} from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";

const EMPTY_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const BRIDGE_MODEL = {
  id: "agent-r1-rollout",
  name: "Agent-R1 rollout bridge",
  api: "openai-completions",
  provider: "agent-r1",
  baseUrl: "",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 131072,
  maxTokens: 32768,
};

const sessions = new Map();
const pendingHostRequests = new Map();
let requestSequence = 0;

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function requestHost(sessionId, type, payload, abortSignal) {
  const id = `${sessionId}:${++requestSequence}`;
  emit({ type, id, session_id: sessionId, ...payload });
  return new Promise((resolve, reject) => {
    const pending = { resolve, reject };
    pendingHostRequests.set(id, pending);
    if (!abortSignal) return;
    const onAbort = () => {
      if (pendingHostRequests.get(id) !== pending) return;
      pendingHostRequests.delete(id);
      pending.reject(new Error("Generation aborted"));
    };
    if (abortSignal.aborted) {
      onAbort();
      return;
    }
    abortSignal.addEventListener("abort", onAbort, { once: true });
  });
}

function contentText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((item) => item && item.type === "text")
    .map((item) => item.text)
    .join("\n");
}

function contextToOpenAi(context) {
  const messages = [];
  if (context.systemPrompt) messages.push({ role: "system", content: context.systemPrompt });
  for (const message of context.messages) {
    if (message.role === "user") {
      messages.push({ role: "user", content: contentText(message.content) });
      continue;
    }
    if (message.role === "assistant") {
      const text = contentText(message.content);
      const toolCalls = message.content
        .filter((item) => item.type === "toolCall")
        .map((item) => ({
          id: item.id,
          type: "function",
          function: { name: item.name, arguments: JSON.stringify(item.arguments) },
        }));
      const converted = { role: "assistant", content: text };
      if (toolCalls.length > 0) converted.tool_calls = toolCalls;
      messages.push(converted);
      continue;
    }
    if (message.role === "toolResult") {
      messages.push({
        role: "tool",
        tool_call_id: message.toolCallId,
        name: message.toolName,
        content: contentText(message.content),
      });
    }
  }
  const tools = (context.tools ?? []).map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
  return { messages, tools };
}

function canonicalAnchor(messages, tools) {
  return JSON.stringify({
    messages,
    tools: tools.map((tool) => tool.function.name).sort(),
  });
}

function makeAssistantMessage(result) {
  const toolCalls = Array.isArray(result.tool_calls) ? result.tool_calls : [];
  const text = String(result.text ?? "");
  const content = [];
  if (text) content.push({ type: "text", text });
  content.push(
    ...toolCalls.map((toolCall) => ({
        type: "toolCall",
        id: String(toolCall.id),
        name: String(toolCall.name),
        arguments: toolCall.arguments ?? {},
      })),
  );
  if (content.length === 0) content.push({ type: "text", text: "" });
  const stopReason = result.stop_reason ?? (toolCalls.length > 0 ? "toolUse" : "stop");
  return {
    role: "assistant",
    content,
    api: BRIDGE_MODEL.api,
    provider: BRIDGE_MODEL.provider,
    model: BRIDGE_MODEL.id,
    usage: EMPTY_USAGE,
    stopReason,
    timestamp: Date.now(),
  };
}

function makeErrorMessage(message, stopReason = "error") {
  return {
    role: "assistant",
    content: [{ type: "text", text: "" }],
    api: BRIDGE_MODEL.api,
    provider: BRIDGE_MODEL.provider,
    model: BRIDGE_MODEL.id,
    usage: EMPTY_USAGE,
    stopReason,
    errorMessage: message,
    timestamp: Date.now(),
  };
}

function stepPayload(session, generationId, environmentResult, invalidAction = false) {
  return {
    type: "step_complete",
    session_id: session.id,
    generation_id: generationId,
    reward: Number(environmentResult.reward ?? 0),
    terminated: Boolean(environmentResult.terminated),
    truncated: Boolean(environmentResult.truncated),
    invalid_action: invalidAction,
    diagnostics: { ...session.diagnostics },
    info: environmentResult.info ?? {},
  };
}

function applyTurnLimit(session, environmentResult) {
  if (
    environmentResult.terminated ||
    environmentResult.truncated ||
    session.turnCount < session.maxTurns
  ) {
    return environmentResult;
  }
  return {
    ...environmentResult,
    truncated: true,
    info: { ...(environmentResult.info ?? {}), pi_max_turns: true },
  };
}

async function executeEnvironmentAction(session, generationId, action) {
  const isToolAction = typeof action === "string" && action.trimStart().startsWith("{");
  if (isToolAction) session.diagnostics.valid_tool_calls += 1;
  else session.diagnostics.text_actions += 1;
  if (session.seenActions.has(action)) session.diagnostics.repeated_actions += 1;
  session.seenActions.add(action);
  if (session.lastActionInvalid) session.diagnostics.failure_recoveries += 1;
  session.lastActionInvalid = false;
  const hostResult = await requestHost(session.id, "environment_request", {
    generation_id: generationId,
    action,
  });
  const result = applyTurnLimit(session, hostResult);
  session.environmentCompleted.add(generationId);
  session.lastEnvironmentResult = result;
  emit(stepPayload(session, generationId, result));
  return result;
}

function makeRemoteTool(session, schema) {
  return {
    name: schema.name,
    label: schema.name,
    description: schema.description ?? "",
    parameters: schema.parameters ?? { type: "object", properties: {} },
    executionMode: "sequential",
    execute: async (_toolCallId, params) => {
      const generationId = session.currentGenerationId;
      if (!generationId) throw new Error("Tool execution has no active generation");
      const action = JSON.stringify({ name: schema.name, arguments: params });
      const result = await executeEnvironmentAction(session, generationId, action);
      return {
        content: [{ type: "text", text: String(result.observation ?? "") }],
        details: result.info ?? {},
        terminate: Boolean(result.terminated || result.truncated),
      };
    },
  };
}

async function createSession(command) {
  const id = String(command.session_id);
  if (sessions.has(id)) throw new Error(`Session already exists: ${id}`);
  const skillsDir = String(command.skills_dir);
  const env = new NodeExecutionEnv({ cwd: process.cwd() });
  const loaded = await loadSkills(env, skillsDir);
  if (loaded.diagnostics.length > 0) {
    throw new Error(`Pi skill loading failed: ${JSON.stringify(loaded.diagnostics)}`);
  }
  if (loaded.skills.length === 0) {
    throw new Error(`Pi skill loading produced no skills from ${skillsDir}`);
  }
  const visibleSkills = formatSkillsForSystemPrompt(loaded.skills);
  const invokedSkills = loaded.skills.map((skill) => formatSkillInvocation(skill)).join("\n\n");
  const systemPrompt = [
    "You are a customer-service agent operating the tau2 Telecom environment.",
    String(command.domain_policy ?? ""),
    visibleSkills,
    invokedSkills,
    "Use at most one tool call per assistant response. Do not mix user-facing text and a tool call.",
  ]
    .filter(Boolean)
    .join("\n\n");

  const session = {
    id,
    maxTurns: Number(command.max_turns ?? 30),
    turnCount: 0,
    currentGenerationId: null,
    environmentCompleted: new Set(),
    lastEnvironmentResult: null,
    lastActionInvalid: false,
    seenActions: new Set(),
    diagnostics: {
      valid_tool_calls: 0,
      invalid_tool_calls: 0,
      text_actions: 0,
      repeated_actions: 0,
      failure_recoveries: 0,
    },
    closed: false,
    agent: null,
  };
  const tools = command.tools.map((schema) => makeRemoteTool(session, schema));
  const agent = new Agent({
    initialState: {
      systemPrompt,
      model: BRIDGE_MODEL,
      thinkingLevel: "off",
      tools,
    },
    toolExecution: "sequential",
    beforeToolCall: async ({ assistantMessage }) => {
      const toolCalls = assistantMessage.content.filter((item) => item.type === "toolCall");
      const text = contentText(assistantMessage.content).trim();
      if (toolCalls.length > 1) {
        return { block: true, reason: "tau2 Telecom permits one tool call per assistant response" };
      }
      if (text && toolCalls.length > 0) {
        return { block: true, reason: "Do not mix user-facing text and a tool call" };
      }
      return undefined;
    },
    shouldStopAfterTurn: () =>
      session.closed ||
      session.turnCount >= session.maxTurns ||
      Boolean(session.lastEnvironmentResult?.terminated || session.lastEnvironmentResult?.truncated),
    streamFn: (_model, context, options) => {
      const generationId = `${id}:generation:${++session.turnCount}`;
      session.currentGenerationId = generationId;
      const converted = contextToOpenAi(context);
      const anchorObs = canonicalAnchor(converted.messages, converted.tools);
      const stream = createAssistantMessageEventStream();
      const abortSignal = options?.signal;
      const pushFailure = (error, aborted) => {
        const reason = aborted ? "aborted" : "error";
        stream.push({
          type: "error",
          reason,
          error: makeErrorMessage(errorMessage(error), reason),
        });
      };
      if (abortSignal?.aborted) {
        pushFailure("Generation aborted", true);
        return stream;
      }
      requestHost(
        id,
        "generation_request",
        {
          generation_id: generationId,
          messages: converted.messages,
          tools: converted.tools,
          anchor_obs: anchorObs,
        },
        abortSignal,
      )
        .then((result) => {
          const message = makeAssistantMessage(result);
          if (message.stopReason === "error" || message.stopReason === "aborted") {
            stream.push({ type: "error", reason: message.stopReason, error: message });
          } else {
            stream.push({ type: "done", reason: message.stopReason, message });
          }
        })
        .catch((error) => {
          pushFailure(error, abortSignal?.aborted || errorMessage(error).includes("aborted"));
        });
      return stream;
    },
  });
  session.agent = agent;
  agent.subscribe(async (event) => {
    if (event.type !== "turn_end" || event.message.role !== "assistant") return;
    const generationId = session.currentGenerationId;
    if (!generationId) return;
    const toolCalls = event.message.content.filter((item) => item.type === "toolCall");
    if (toolCalls.length === 0) {
      const result = await executeEnvironmentAction(session, generationId, contentText(event.message.content));
      if (!result.terminated && !result.truncated && session.turnCount < session.maxTurns) {
        agent.followUp({
          role: "user",
          content: [{ type: "text", text: String(result.observation ?? "") }],
          timestamp: Date.now(),
        });
      }
      return;
    }
    if (!session.environmentCompleted.has(generationId)) {
      session.diagnostics.invalid_tool_calls += 1;
      session.lastActionInvalid = true;
      const result = applyTurnLimit(session, {
        reward: 0,
        terminated: false,
        truncated: false,
        info: { pi_tool_errors: event.toolResults.map((item) => contentText(item.content)) },
      });
      session.environmentCompleted.add(generationId);
      session.lastEnvironmentResult = result;
      emit(
        stepPayload(
          session,
          generationId,
          result,
          true,
        ),
      );
    }
  });
  sessions.set(id, session);
  emit({ type: "session_started", session_id: id, skill_count: loaded.skills.length });

  try {
    await agent.prompt(String(command.initial_observation ?? ""));
    emit({
      type: "session_complete",
      session_id: id,
      terminated: Boolean(session.lastEnvironmentResult?.terminated),
      truncated:
        Boolean(session.lastEnvironmentResult?.truncated) ||
        (!session.lastEnvironmentResult?.terminated && session.turnCount >= session.maxTurns),
      turns: session.turnCount,
    });
  } catch (error) {
    emit({ type: "session_error", session_id: id, error: errorMessage(error) });
  }
}

function closeSession(sessionId) {
  const session = sessions.get(sessionId);
  if (!session) return;
  session.closed = true;
  session.agent?.abort();
  sessions.delete(sessionId);
  for (const [requestId, pending] of pendingHostRequests.entries()) {
    if (requestId.startsWith(`${sessionId}:`)) {
      pendingHostRequests.delete(requestId);
      pending.reject(new Error(`Pi session closed: ${sessionId}`));
    }
  }
}

function handleResponse(message) {
  const pending = pendingHostRequests.get(message.response_to);
  if (!pending) return;
  pendingHostRequests.delete(message.response_to);
  if (message.ok) pending.resolve(message.result ?? {});
  else pending.reject(new Error(String(message.error ?? "Host request failed")));
}

async function handleCommand(message) {
  if (message.type === "response") {
    handleResponse(message);
    return;
  }
  if (message.type === "start_session") {
    void createSession(message).catch((error) => {
      emit({
        type: "session_error",
        session_id: String(message.session_id),
        error: errorMessage(error),
      });
    });
    return;
  }
  if (message.type === "close_session") {
    closeSession(String(message.session_id));
    return;
  }
  if (message.type === "shutdown") {
    for (const id of [...sessions.keys()]) closeSession(id);
    process.exit(0);
  }
  throw new Error(`Unknown command: ${message.type}`);
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", (line) => {
  let message;
  try {
    message = JSON.parse(line);
  } catch (error) {
    process.stderr.write(`Invalid JSON command: ${errorMessage(error)}\n`);
    return;
  }
  handleCommand(message).catch((error) => {
    const sessionId = message.session_id ? String(message.session_id) : undefined;
    if (sessionId) emit({ type: "session_error", session_id: sessionId, error: errorMessage(error) });
    else process.stderr.write(`${errorMessage(error)}\n`);
  });
});

emit({ type: "ready", protocol_version: 1, pi_agent_core: "0.84.4" });

export {
  applyTurnLimit,
  canonicalAnchor,
  contextToOpenAi,
  makeAssistantMessage,
};
