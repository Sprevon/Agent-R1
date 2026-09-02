class ModelRuntime {
  constructor() {
    this.providers = new Map();
  }

  static async create() {
    return new ModelRuntime();
  }

  registerNativeProvider(provider) {
    this.providers.set(provider.id, provider);
  }
}

class DefaultResourceLoader {
  async reload() {}

  getExtensions() {
    return { errors: [] };
  }
}

class SessionManager {
  static inMemory() {
    return new SessionManager();
  }
}

async function createAgentSession(options) {
  if (!options.modelRuntime?.providers.has(options.model.provider)) {
    throw new Error(`Model provider was not registered: ${options.model.provider}`);
  }

  const sessionSubscribers = [];
  const agentSubscribers = [];
  const emitSession = (event) => {
    for (const subscriber of sessionSubscribers) subscriber(event);
  };
  const emitCustomEntry = (customType, data) => {
    emitSession({
      type: "entry_appended",
      entry: { type: "custom", customType, data },
    });
  };

  const agent = {
    streamFunction: null,
    shouldStopAfterTurn: () => false,
    subscribe(subscriber) {
      agentSubscribers.push(subscriber);
    },
    abort() {},
  };

  const session = {
    agent,
    subscribe(subscriber) {
      sessionSubscribers.push(subscriber);
    },
    async bindExtensions(bindings) {
      if (process.env.FAKE_PI_EXTENSION_ERROR === "1") {
        bindings.onError?.("fixture extension startup failure");
        return;
      }
      emitCustomEntry("agent-r1-task-prompt", {
        task_id: process.env.TAU2_TELECOM_TASK_ID,
        prompt:
          `/skill:telecom-solo-support Task ID: ${process.env.TAU2_TELECOM_TASK_ID}\n` +
          "Policy mode: workflow\n\nFixture ticket",
      });
    },
    async prompt(prompt) {
      const stream = agent.streamFunction(
        options.model,
        {
          systemPrompt: "fixture system prompt",
          messages: [{ role: "user", content: prompt }],
          tools: [],
        },
        {},
      );
      let assistantMessage;
      for await (const event of stream) {
        if (event.type === "done") assistantMessage = event.message;
        if (event.type === "error") throw new Error(event.error?.errorMessage ?? "stream failed");
      }
      if (!assistantMessage) throw new Error("Bridge stream produced no assistant message");
      for (const subscriber of agentSubscribers) {
        subscriber({ type: "turn_end", message: assistantMessage, toolResults: [] });
      }
    },
    async waitForIdle() {},
    extensionRunner: {
      async emit(event) {
        if (event.type === "session_shutdown") {
          emitCustomEntry("agent-r1-evaluation", {
            reward: 1,
            task_id: process.env.TAU2_TELECOM_TASK_ID,
          });
        }
      },
    },
    dispose() {},
  };

  return { session };
}

export { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager };
