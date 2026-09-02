import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import {
  createAssistantMessageEventStream,
  createProvider,
} from "@earendil-works/pi-ai";

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
  return new Promise((resolveRequest, rejectRequest) => {
    const pending = { resolve: resolveRequest, reject: rejectRequest };
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
  for (const message of context.messages ?? []) {
    if (message.role === "user") {
      messages.push({ role: "user", content: contentText(message.content) });
      continue;
    }
    if (message.role === "assistant") {
      const content = Array.isArray(message.content) ? message.content : [];
      const toolCalls = content
        .filter((item) => item && item.type === "toolCall")
        .map((item) => ({
          id: item.id,
          type: "function",
          function: { name: item.name, arguments: JSON.stringify(item.arguments ?? {}) },
        }));
      const converted = { role: "assistant", content: contentText(message.content) };
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

function createBridgeProvider() {
  const unavailableStream = () => {
    const stream = createAssistantMessageEventStream();
    stream.push({
      type: "error",
      reason: "error",
      error: makeErrorMessage(
        "Agent-R1 bridge provider must be invoked through session.agent.streamFunction",
      ),
    });
    return stream;
  };
  return createProvider({
    id: BRIDGE_MODEL.provider,
    name: "Agent-R1 host bridge",
    auth: {
      apiKey: {
        name: "Agent-R1 host bridge",
        resolve: async () => ({ auth: {} }),
      },
    },
    models: [BRIDGE_MODEL],
    api: {
      stream: unavailableStream,
      streamSimple: unavailableStream,
    },
  });
}

async function loadCodingAgent(entrypoint) {
  if (!entrypoint) {
    throw new Error(
      "PI_CODING_AGENT_ENTRYPOINT is required; point it to pi/packages/coding-agent/dist/index.js",
    );
  }
  return import(pathToFileURL(resolve(entrypoint)).href);
}

function serializable(value) {
  try {
    JSON.stringify(value);
    return value;
  } catch {
    return String(value);
  }
}

async function createSession(command) {
  const id = String(command.session_id);
  if (sessions.size > 0) {
    throw new Error(
      "The canonical tau2 Telecom extension is process-scoped; run one Pi sidecar per trajectory",
    );
  }
  if (sessions.has(id)) throw new Error(`Session already exists: ${id}`);

  const tau2Root = String(command.tau2_root ?? "");
  if (!tau2Root) throw new Error("start_session.tau2_root is required");
  const trainingExtension = String(
    command.training_extension ?? resolve(tau2Root, ".pi", "extensions", "agent-r1-training.ts"),
  );
  const agentDir = String(command.agent_dir ?? resolve(tau2Root, ".pi", "agent"));
  const entrypoint = String(
    command.pi_coding_agent_entrypoint ?? process.env.PI_CODING_AGENT_ENTRYPOINT ?? "",
  );
  const { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager } =
    await loadCodingAgent(entrypoint);

  process.env.TAU2_TELECOM_TASK_ID = String(command.task_id ?? process.env.TAU2_TELECOM_TASK_ID ?? "");
  if (!process.env.TAU2_TELECOM_TASK_ID.trim()) {
    throw new Error("start_session.task_id is required");
  }

  const state = {
    id,
    maxTurns: Number(command.max_turns ?? 30),
    turnCount: 0,
    currentGenerationId: null,
    closed: false,
    evaluation: null,
    taskPrompt: null,
    extensionErrors: [],
    diagnostics: {
      generation_requests: 0,
      completed_turns: 0,
      tool_turns: 0,
      text_turns: 0,
    },
  };
  sessions.set(id, state);

  const resourceLoader = new DefaultResourceLoader({
    cwd: tau2Root,
    agentDir,
    noExtensions: true,
    additionalExtensionPaths: [trainingExtension],
  });
  await resourceLoader.reload();
  const extensionErrors = resourceLoader.getExtensions().errors ?? [];
  if (extensionErrors.length > 0) {
    throw new Error(`Pi extension loading failed: ${JSON.stringify(extensionErrors)}`);
  }

  const modelRuntime = await ModelRuntime.create({ modelsPath: null });
  modelRuntime.registerNativeProvider(createBridgeProvider());

  const { session } = await createAgentSession({
    cwd: tau2Root,
    agentDir,
    model: BRIDGE_MODEL,
    thinkingLevel: "off",
    noTools: "builtin",
    modelRuntime,
    resourceLoader,
    sessionManager: SessionManager.inMemory(tau2Root),
  });
  state.session = session;
  session.agent.streamFunction = (_model, context, options) => {
    const generationId = `${id}:generation:${++state.turnCount}`;
    state.currentGenerationId = generationId;
    state.diagnostics.generation_requests += 1;
    const converted = contextToOpenAi(context);
    const stream = createAssistantMessageEventStream();
    const abortSignal = options?.signal;
    const fail = (error, aborted = false) => {
      const reason = aborted ? "aborted" : "error";
      stream.push({
        type: "error",
        reason,
        error: makeErrorMessage(errorMessage(error), reason),
      });
    };
    if (abortSignal?.aborted) {
      fail("Generation aborted", true);
      return stream;
    }
    requestHost(
      id,
      "generation_request",
      {
        generation_id: generationId,
        messages: converted.messages,
        tools: converted.tools,
        anchor_obs: canonicalAnchor(converted.messages, converted.tools),
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
      .catch((error) => fail(error, abortSignal?.aborted || errorMessage(error).includes("aborted")));
    return stream;
  };
  session.agent.shouldStopAfterTurn = () => state.closed || state.turnCount >= state.maxTurns;

  session.subscribe((event) => {
    if (event.type === "entry_appended" && event.entry?.type === "custom") {
      if (event.entry.customType === "agent-r1-evaluation") {
        state.evaluation = event.entry.data;
        emit({
          type: "evaluation_result",
          session_id: id,
          result: serializable(event.entry.data),
        });
      } else if (event.entry.customType === "agent-r1-task-prompt") {
        const taskId = String(event.entry.data?.task_id ?? "");
        const prompt = String(event.entry.data?.prompt ?? "");
        if (taskId !== process.env.TAU2_TELECOM_TASK_ID) {
          state.extensionErrors.push(
            `Training extension returned task ${taskId || "<empty>"}, expected ${process.env.TAU2_TELECOM_TASK_ID}`,
          );
        } else if (!prompt.trim()) {
          state.extensionErrors.push("Training extension returned an empty task prompt");
        } else {
          state.taskPrompt = prompt;
        }
      }
    }
  });
  session.agent.subscribe((event) => {
    if (event.type !== "turn_end" || event.message?.role !== "assistant") return;
    const generationId = state.currentGenerationId;
    if (!generationId) return;
    const hasToolCalls = Array.isArray(event.message.content) &&
      event.message.content.some((item) => item?.type === "toolCall");
    if (hasToolCalls) state.diagnostics.tool_turns += 1;
    else state.diagnostics.text_turns += 1;
    state.diagnostics.completed_turns += 1;
    emit({
      type: "step_complete",
      session_id: id,
      generation_id: generationId,
      reward: 0,
      terminated: false,
      truncated: state.turnCount >= state.maxTurns,
      invalid_action: false,
      diagnostics: { ...state.diagnostics },
      assistant_message: serializable(event.message),
      tool_results: serializable(event.toolResults ?? []),
    });
  });

  emit({ type: "session_started", session_id: id, protocol_version: 2, runtime: "pi-coding-agent" });
  try {
    await session.bindExtensions({
      mode: "rpc",
      onError: (error) => state.extensionErrors.push(serializable(error)),
    });
    if (state.extensionErrors.length > 0) {
      throw new Error(`Pi extension startup failed: ${JSON.stringify(state.extensionErrors)}`);
    }
    if (!state.taskPrompt) {
      throw new Error("Training extension did not publish the canonical Telecom task prompt");
    }
    await session.prompt(state.taskPrompt, { expandPromptTemplates: true });
    if (state.extensionErrors.length > 0) {
      throw new Error(`Pi extension runtime failed: ${JSON.stringify(state.extensionErrors)}`);
    }
    await session.waitForIdle();
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    if (state.extensionErrors.length > 0) {
      throw new Error(`Pi extension shutdown failed: ${JSON.stringify(state.extensionErrors)}`);
    }
    if (state.evaluation === null) {
      throw new Error("Training extension did not publish a Telecom evaluation result");
    }
    session.dispose();
    emit({
      type: "session_complete",
      session_id: id,
      terminated: false,
      truncated: state.turnCount >= state.maxTurns,
      turns: state.turnCount,
      diagnostics: { ...state.diagnostics },
    });
  } catch (error) {
    emit({ type: "session_error", session_id: id, error: errorMessage(error) });
    try {
      session.dispose();
    } catch {
      // Best-effort cleanup; the host still receives the session_error above.
    }
  } finally {
    sessions.delete(id);
  }
}

function closeSession(sessionId) {
  const state = sessions.get(sessionId);
  if (!state) return;
  state.closed = true;
  state.session?.agent.abort();
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
      emit({ type: "session_error", session_id: String(message.session_id), error: errorMessage(error) });
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

emit({ type: "ready", protocol_version: 2, pi_runtime: "pi-coding-agent" });

export { canonicalAnchor, contextToOpenAi, makeAssistantMessage };
