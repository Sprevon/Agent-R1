# Pi/tau2 OPD smoke 故障复盘与 Agent-R1 衔接说明

## 1. 文档目的

本文记录 Qwen3-0.6B student、Qwen3-4B teacher、3 张 RTX 4080 SUPER 的
Pi/tau2 Telecom OPD smoke 中已经定位并修复的问题，并给出对应代码、验证证据和
仍然存在的边界。最后一节从真实运行时出发，说明一次 Pi LLM 请求如何进入
Agent-R1、如何返回 Pi 执行工具，以及如何变成可供 OPD 训练的
`AgentFlowStep`。

这次 smoke 的目标不只是“服务启动”或“checkpoint 写出”，而是验证以下链路：

```text
tau2 task/DB/tools/evaluator
        ↑
Pi skill + extension + transcript + agent loop
        ↕ JSONL generation protocol
Agent-R1 rollout server + tokenizer/parser + recorder
        ↓
response_mask + terminal reward + OPD optimizer update
```

本文涉及两个代码仓库：

- Agent-R1：`Agent-R1-opd`，实验分支
  `experiment/opd-qwen3-4b-32b-3xpro6000`；
- tau2：`tau2-bench`，提供 canonical Telecom extension、Python bridge、任务状态与
  evaluator。

对应修复提交：

- tau2 `8a890b4`：暴露可等待的 canonical Telecom training prompt；
- tau2 `fe8ac67`：从 task lifecycle 正确发布 prompt/evaluation callback；
- Agent-R1 `13a2a7a`：显式等待真实 Pi training lifecycle；
- Agent-R1 `b0fc3f1`：把 terminal evaluator result 设为 session 完成硬门槛。

## 2. 修复后的职责边界

三个系统各自只拥有一份权威状态：

| 层 | 权威职责 | 不负责什么 |
|---|---|---|
| tau2 | task、初始 DB、Telecom tools、tool side effect、assistant text history、最终 evaluator | 不生成模型 token，不维护 Agent-R1 rollout tensor |
| Pi | skill/template 展开、extension lifecycle、消息 transcript、tool schema、agent/tool loop | 不运行第二套 Telecom 环境，不计算 PPO/OPD loss |
| Agent-R1 | chat template、vLLM generation、Hermes tool parser、token/logprob 记录、reward tensor、OPD 更新 | 不重写 Pi agent loop，不复制 tau2 evaluator |

Telecom recipe 不经过通用 `AgentEnvLoop -> AgentEnv.step()`。原因是一个 Pi turn
包含“追加 assistant message、执行零个或多个工具、追加 tool result、决定是否继续”
这一整套语义。该循环继续由 Pi 官方 `createAgentSession()` 驱动，Agent-R1 只替换
LLM generation transport。

## 3. 已解决问题总览

| 编号 | 症状 | 根因 | 修复 | 验证结果 |
|---|---|---|---|---|
| P1 | 4B teacher vLLM 启动时 KV cache OOM | 24,129 token 上下文在 `gpu_memory_utilization=0.45` 下只剩约 1.03 GiB，实际约需 3.32 GiB | 仅在 3×4080 smoke profile 覆盖为 `0.60` | teacher 启动，1-step OPD 完成 |
| P2 | Pi 启动 tau2 bridge 报 `spawn python ENOENT` | Node 子进程继承环境中没有可用的裸 `python` | `common.sh` 将解析后的 `PYTHON_BIN` 导出为 `TAU2_PI_PYTHON` | bridge 使用 `/root/envs/toolcall/bin/python` 正常启动 |
| P3 | session 看似完成，但 `generation_request=0`、零 turn | bridge model 的 provider 未注册；Pi 在调用被覆写的 `streamFunction` 前先做 provider/auth 校验 | 创建 keyless native provider，注册进 `ModelRuntime`，再传给 `createAgentSession()` | 真实 Pi session 发出 generation request |
| P4 | extension 启动错误被吞掉，`waitForIdle()` 后仍产生空轨迹 | `pi.sendUserMessage()` 是 fire-and-forget；extension error 没有绑定到 sidecar fail-fast 路径 | training wrapper 禁用 auto prompt，sidecar 订阅 entry 后显式 `await session.prompt()`，全 lifecycle 收集并抛出 extension error | prompt/provider/extension 错误会成为 `session_error` |
| P5 | 完成 P4 后仍拿不到 task prompt | `loadTelecomTask(..., options)` 的局部参数遮蔽 factory 外层 `options` | 内层参数改名 `loadOptions`，callback 始终读取外层 `options.onTaskLoaded` | canonical prompt 正常发布并被 sidecar await |
| P6 | 无 `TAU2_TELECOM_EVAL_OUT` 时没有 evaluation callback | `writeEvaluation()` 把“是否写文件”错误地当成“是否 evaluate”的条件 | 文件输出与 callback 解耦；只要注册 `onEvaluation` 就 evaluate，训练路径失败时抛错 | `evaluation_result` 和 `session_complete` 均出现 |
| P7 | 环境缺项直到深层 runtime 才暴露 | preflight 只检查路径/普通 import，没有验证实际 SDK exports | 动态 import 指定 Pi entrypoint 并检查四个必需 export；增加 sidecar 单元测试 | preflight、3 个 Node 测试通过 |
| P8 | README 引用了不存在的 4080 smoke 脚本 | 文档落后于当前启动方式 | 改为完整的 3×4080 Qwen3 命令，并记录 teacher `0.60` 覆盖 | 命令与成功 smoke 参数一致 |

## 4. P1：4B teacher 的 KV cache OOM

### 4.1 现象

Qwen3-4B teacher 能加载权重，但 vLLM 在为最大 24,129-token context 分配 KV cache
时失败。原 smoke profile 的 teacher `gpu_memory_utilization=0.45` 只留下约
1.03 GiB KV cache 空间，而当前模型、dtype 和 context 组合约需 3.32 GiB。

因此这不是训练逻辑错误，也不是 student/teacher checkpoint 损坏；失败发生在
teacher inference engine 的容量规划阶段。

### 4.2 解决方法

保留全局默认值不变，只在这台 3×32 GiB RTX 4080 SUPER 的 smoke 命令中增加：

```bash
TAU2_TEACHER_GPU_MEMORY_UTILIZATION=0.60
```

这是硬件和模型 profile 的局部覆盖，不应无条件推广到不同显存、不同 context
长度或不同并发配置。更换硬件后应重新看 vLLM 的 model weight、non-torch memory、
activation peak 与 KV cache sizing 日志。

### 4.3 为什么没有通过缩短 prompt“解决”

Telecom prompt 包括 system prompt、canonical skill、task ticket 和当前激活工具的
schema。直接裁剪 prompt 会改变 Pi 实际看到的上下文，从而破坏训用一致；因此本次
先解决显存配置，没有为了让 smoke 启动而删 skill 或 tools。

## 5. P2：tau2 Python bridge 找不到解释器

### 5.1 现象和根因

Pi extension 在 Node 进程里通过 `child_process.spawn()` 启动：

```ts
const configuredPython = process.env.TAU2_PI_PYTHON?.trim();
const projectPython = join(this.cwd, ".venv", "bin", "python");
const python =
  configuredPython || (existsSync(projectPython) ? projectPython : "python");
```

远程训练实际使用 `/root/envs/toolcall/bin/python`，但 Node 环境中裸 `python` 不存在，
tau2 checkout 下也没有 `.venv/bin/python`，于是出现 `spawn python ENOENT`。

### 5.2 代码修复

`examples/tau2_telecom/common.sh` 先解析 Agent-R1 的 Python，再把同一个解释器传给
Pi extension：

```bash
PYTHON_BIN="${PYTHON_BIN:-python3}"
export TAU2_PI_PYTHON="${TAU2_PI_PYTHON:-$PYTHON_BIN}"
```

远程 smoke 显式使用：

```bash
PYTHON_BIN=/root/envs/toolcall/bin/python \
TAU2_PI_PYTHON=/root/envs/toolcall/bin/python \
...
```

这样 Agent-R1 driver 与 tau2 bridge 使用同一可导入当前 tau2 checkout 的解释器。
extension 仍保留 `.venv` 和裸 `python` fallback，方便交互式环境使用。

## 6. P3：bridge provider 未注册导致零 generation

### 6.1 为什么覆写 `streamFunction` 仍然不够

最初 sidecar 已经执行：

```js
session.agent.streamFunction = (...args) => { /* 发 generation_request */ };
```

但 bridge model 声明了自定义 provider：

```js
const BRIDGE_MODEL = {
  id: "agent-r1-rollout",
  provider: "agent-r1",
  api: "openai-completions",
  ...
};
```

Pi 的 `session.prompt()` 在进入 agent loop 之前会先通过 `ModelRuntime` 校验 model
provider 和 auth。未注册的 `agent-r1` 在 preflight 阶段就失败，因而根本不会调用
已覆写的 `session.agent.streamFunction`。从外面看就是 session 启动过，但没有任何
LLM request 和 turn。

### 6.2 代码修复

sidecar 使用 `@earendil-works/pi-ai` 创建不需要外部密钥的 native provider：

```js
function createBridgeProvider() {
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

const modelRuntime = await ModelRuntime.create({ modelsPath: null });
modelRuntime.registerNativeProvider(createBridgeProvider());

const { session } = await createAgentSession({
  model: BRIDGE_MODEL,
  modelRuntime,
  ...
});
```

`unavailableStream` 是一个保护门：如果代码绕开 sidecar 覆写，直接调用 provider 的
API stream，会立即报错。正常路径必须经过：

```text
Pi agent loop
  -> session.agent.streamFunction
  -> JSONL generation_request
  -> Agent-R1 server_manager.generate
```

它不会误发请求到 OpenAI-compatible URL，也不需要伪造 API key。

## 7. P4：fire-and-forget prompt 与 extension error 被吞

### 7.1 原行为

canonical extension 原来在 `session_start` 中调用：

```ts
pi.sendUserMessage(prompt, { expandPromptTemplates: true });
```

这是 extension 主动投递消息的交互式用法。其异步错误由 Pi extension runner 捕获，
不会自然成为 sidecar 正在等待的 Promise rejection。sidecar 当时也没有给
`bindExtensions()` 提供 `onError`，随后 `waitForIdle()` 可以在零 turn 状态返回，
导致“成功完成的空轨迹”。

### 7.2 training wrapper 的解决方法

`tau2-bench/.pi/extensions/agent-r1-training.ts` 不复制 canonical extension，而是只做
训练生命周期适配：

```ts
const telecom = createTau2TelecomExtension({
  taskId,
  autoPrompt: false,
  failFastOnStartError: true,
  onTaskLoaded: async (task) => {
    pi.appendEntry("agent-r1-task-prompt", {
      task_id: task.task_id,
      prompt: formatTelecomTaskPrompt(task),
    });
  },
  onEvaluation: async (result) => {
    pi.appendEntry("agent-r1-evaluation", result);
  },
});
```

wrapper 只发布两个 custom session entry：canonical task prompt 和 evaluator result。
tool 注册、allowlist、DB、tool execution 和 evaluation 仍全部复用
`createTau2TelecomExtension()`。

### 7.3 sidecar 的可等待生命周期

sidecar 在 bind 前先订阅 entry，然后按严格顺序执行：

```js
await session.bindExtensions({
  mode: "rpc",
  onError: (error) => state.extensionErrors.push(serializable(error)),
});
if (!state.taskPrompt) {
  throw new Error("Training extension did not publish the canonical Telecom task prompt");
}
await session.prompt(state.taskPrompt, { expandPromptTemplates: true });
await session.waitForIdle();
await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
```

并且分别在 startup、runtime、shutdown 后检查 `extensionErrors`。因此以下任何问题都
会成为 `session_error`，而不是悄悄产出空样本：

- tau2 bridge 启动失败；
- task ID 不一致或 prompt 为空；
- Pi provider/auth preflight 失败；
- agent loop 中 generation/tool 执行失败；
- evaluator 或 shutdown callback 失败。

## 8. P5：局部 `options` 遮蔽外层 extension options

### 8.1 现象

完成显式 prompt lifecycle 后，sidecar 仍报：

```text
Training extension did not publish the canonical Telecom task prompt
```

task 实际已经加载，但 `onTaskLoaded` 没有执行。

### 8.2 根因

factory 外层参数名是 `options`：

```ts
createTau2TelecomExtension(options: TelecomExtensionOptions = {})
```

旧的内部函数又使用同名参数：

```ts
loadTelecomTask(taskId, ctx, options: { sendPrompt: boolean })
```

因此函数体中的 `options.onTaskLoaded` 实际访问的是只有 `sendPrompt` 的局部对象，
而不是 factory 的 `TelecomExtensionOptions`。TypeScript 的可选链使它安静地变成
`undefined`，没有直接抛错。

### 8.3 修复

将内部参数改名为 `loadOptions`，明确区分两个作用域：

```ts
const loadTelecomTask = async (
  taskId: string,
  ctx: ...,
  loadOptions: { sendPrompt: boolean },
) => {
  ...
  if (loadOptions.sendPrompt) {
    pi.sendUserMessage(formatTelecomTaskPrompt(loaded), ...);
  }
  await options.onTaskLoaded?.(loaded, getSnapshot());
};
```

这次修复之后，`agent-r1-task-prompt` 才能稳定到达 sidecar。

## 9. P6：evaluation 被输出文件开关错误短路

### 9.1 原逻辑的问题

旧逻辑在没有配置 `TAU2_TELECOM_EVAL_OUT` 时直接 return：

```ts
if (!evalOut || client === undefined) return;
```

该条件适合“只向文件导出评测”的交互式流程，但 training wrapper 使用
`onEvaluation` callback 把结果放进 Pi session entry，并不要求落 JSON 文件。因此
任务即使运行结束，也不会触发 evaluator，Agent-R1 收不到 terminal reward。

### 9.2 修复

现在“是否执行 evaluate”和“是否写文件”是两件事：

```ts
const evalOut = process.env.TAU2_TELECOM_EVAL_OUT?.trim();
if ((!evalOut && options.onEvaluation === undefined) || client === undefined) return;
if (evalOut) mkdirSync(dirname(evalOut), { recursive: true });

try {
  const result = await evaluate();
  if (evalOut) writeFileSync(evalOut, `${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  ...
  if (options.onEvaluation !== undefined) throw error;
}
```

训练路径还增加了双重硬门槛：

1. tau2 evaluation 失败时，只要注册了 callback 就重新抛错；
2. sidecar 在 shutdown 后要求 `state.evaluation !== null`，否则 session 失败。

这避免把“没有 reward”自动解释为合法的 `0.0`。

## 10. P7：preflight 和回归测试补齐

`scripts/check_pi_tau_environment.py` 现在不仅检查文件是否存在，还会动态 import
`PI_CODING_AGENT_ENTRYPOINT` 并验证真实 runtime 需要的 export：

```text
createAgentSession
DefaultResourceLoader
SessionManager
ModelRuntime
```

sidecar 测试覆盖：

- bridge provider 被注册，且显式 await canonical prompt；
- 真实 generation request/response 可以结束 session；
- extension startup/runtime error 会传播为 `session_error`；
- 缺少 evaluator result 时不能报告成功。

测试不会用一个假的 Python environment 替换训练 acceptance。最终仍用真实 Pi SDK、
真实 tau2 extension 和真实 GPU rollout 验证。

## 11. 实际验证结果

### 11.1 静态和无 GPU runtime 验证

- Node sidecar tests：3/3 通过；
- Python tests：7/7 通过；远程环境没有 pytest，因此未安装新依赖，使用
  `unittest` 执行已有测试；
- environment preflight：通过；Python 3.10 虽不在 tau2 声明的首选范围内，但
  当前 `AgentGymEnv` 实际 import 成功；
- GPU 探测：3 张 RTX 4080 SUPER 可见；
- 真实 Pi/tau2 无 GPU session：canonical skill 展开，首个
  `generation_request` 携带 43 个 Telecom tools，完成 1 turn，并收到
  `evaluation_result` 与 `session_complete`；
- scripted tool path：Pi 实际执行
  `get_customer_by_phone(phone_number="555-123-2002")`，得到 John Smith，tool result
  被加入 transcript 后触发第二次 generation；最终 2 turns，其中 1 tool turn、
  1 text turn，`n_tool_calls=1`、`n_tool_errors=0`。

scripted tool path 只证明 tool schema、Hermes payload、Pi tool execution 和 tau2 DB
回灌是通的，不代表 0.6B policy 已学会稳定选择工具。

### 11.2 真实 GPU 1-step OPD

配置：Qwen3-0.6B student、Qwen3-4B teacher、3×RTX 4080 SUPER、1 optimizer step。

结果：

| 指标 | 观测值 |
|---|---:|
| process exit | 0 |
| `global_step` | 1 |
| `actor/loss` | 2.5145747661590576 |
| `actor/distillation/loss` | 1.2572874128818512 |
| student logprob | -0.3472908679 |
| teacher logprob | -2.7258440554 |
| grad norm | 378 |
| response length mean/min/max | 27.5 / 15 / 58 |
| number of Pi steps mean | 2 |
| response aborted ratio | 0 |

远程证据位置：

```text
/root/autodl-tmp/logs/tau2-pi-opd-0p6-4b-realpi-fix-smoke-20260902.log
/root/autodl-tmp/ckpt/tau2-pi-opd-0p6-4b-realpi-fix-smoke-20260902/global_step_1
```

checkpoint 包含 model、optimizer、extra state shards，以及：

```text
actor/huggingface/model.safetensors
```

进程结束后 3 张 GPU 均回落至约 1 MiB 占用。

### 11.3 response mask 的证据边界

这次日志没有直接打印完整 `response_mask` tensor，因此不能声称做过逐元素日志比对。
当前结论由代码契约和 runtime metric 联合得到：

- `PiTrainingRecorder` 为每个完成的 generation 保存非空 `response_ids`；
- `AgentFlowBase` 在 step 未显式指定 mask 时，默认生成
  `[1] * len(response_ids)`，再与 attention mask 相乘；
- 本次 response length 为 15～58，而不是 0；
- loss、distillation loss、logprob 和 optimizer step 均为有限值。

所以这次 smoke 的训练 response region 非空；但如果 acceptance 要求逐 token 证明，
下一轮应把 mask sum、首尾边界和 prompt/response token hash 作为结构化 telemetry
输出，而不是依靠这一推导。

## 12. 仍未解决或不应误判为已解决的问题

### 12.1 0.6B student 没有自主调用工具

真实 GPU trajectories 中，Qwen3-0.6B 输出了 text，`tool_calls=0`，最终 Telecom
reward 为 `0.0`。scripted real-Pi test 已证明 tool infrastructure 可用，因此当前
主要问题已从“链路坏了”收敛为“tiny base model 的 policy/行为质量不足”。可能相关的
因素包括约 13k token 的长 prompt、43 个 tools，以及 0.6B base model 的 tool-use
能力。

不能为了让 smoke 出现 tool call 而在训练路径强制注入工具选择；这会改变 on-policy
分布，反而破坏训练数据的真实性。后续应通过更强 student/更合适的 SFT warm start、
任务/tool schema 分析或训练过程改善行为。

### 12.2 成功退出后的 teardown warning

checkpoint 成功写出、进程 exit 0 之后出现过 DataLoader worker killed 和 vLLM core
died 的 teardown warning，GPU 最终已释放。这次不阻塞 1-step smoke，但长跑前仍应
观察它是否只出现在 Ray/vLLM shutdown，不能把 warning 永久忽略。

### 12.3 远程 Git 代理

远程 Git 配置的 `127.0.0.1:17890` 当前不可用。本次同步通过 command-scoped
`http.proxy=`、`https.proxy=` 加 ghfast mirror 完成，没有修改远程全局 Git 配置。
这是同步 workaround，不是代理根因修复。

### 12.4 验证工具缺失与 worktree 状态

- 远程没有 pytest/ruff，本次没有擅自安装依赖；使用 unittest、真实 loader、真实
  Pi session 和 GPU smoke 替代相应验证，不能表述为完整 `make check-all` 已通过；
- tau2 sidecar 目录存在预先已有的 untracked `node_modules/`，本次没有清理或提交。

## 13. 复现命令与 acceptance gate

远程 3×4080 profile：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHON_BIN=/root/envs/toolcall/bin/python \
TAU2_PI_PYTHON=/root/envs/toolcall/bin/python \
TAU2_BENCH_ROOT=/root/autodl-tmp/code/tau2-bench-official \
PI_CODING_AGENT_ENTRYPOINT=/root/autodl-tmp/code/pi/packages/coding-agent/dist/index.js \
STUDENT_MODEL=/root/autodl-tmp/models/Qwen3-0.6B \
TEACHER_MODEL=/root/autodl-tmp/models/Qwen3-4B \
STUDENT_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=1 \
TAU2_TRAIN_PATH=/root/autodl-tmp/data/tau2_telecom_smoke_2/train.parquet \
TAU2_VAL_PATH=/root/autodl-tmp/data/tau2_telecom_smoke_2/test.parquet \
TAU2_TRAIN_BATCH_SIZE=2 TAU2_PPO_MINI_BATCH_SIZE=2 \
TAU2_MAX_PROMPT_LEN=24000 TAU2_MAX_RESPONSE_LEN=128 \
AGENT_FLOW_WORKERS=1 \
TAU2_TEACHER_GPU_MEMORY_UTILIZATION=0.60 \
EXP_NAME=qwen3_0p6b_teacher4b_realpi_smoke \
SAVE_FREQ=1 TEST_FREQ=1 TOTAL_EPOCHS=1 \
bash examples/tau2_telecom/run_opd.sh \
  trainer.total_training_steps=1 \
  trainer.default_local_dir=/root/autodl-tmp/ckpt/tau2-pi-opd-0p6-4b-realpi-smoke
```

至少检查：

1. preflight 动态 import 真实 Pi SDK 成功；
2. `generation_request > 0` 且 prompt/tools 与 Pi context 一致；
3. response length 和 response mask region 非空；
4. Pi turn 完成，tool call 若存在则由 Pi 执行且回到 transcript；
5. evaluator result 存在，不用缺失 reward 冒充 `0.0`；
6. OPD token alignment、teacher/student logprob 和 loss 为有限值；
7. optimizer 至少更新一次；
8. checkpoint 完整写出；
9. 进程结束后 Ray/vLLM worker 与 GPU 资源被释放。

## 14. 实际 Pi LLM 调用与 Agent-R1 衔接细节

本节按一次真实 trajectory 的时序描述调用链。关键文件：

- Pi sidecar：`recipes/tau2_telecom/pi_sidecar/src/main.mjs`；
- Agent-R1 flow：`recipes/tau2_telecom/pi_tau_agent_flow.py`；
- training recorder：`recipes/tau2_telecom/pi_training_recorder.py`；
- AgentFlow tensor assembly：`agent_r1/agent_flow/agent_flow.py`；
- tau2 extension：`tau2-bench/.pi/lib/tau2-telecom.ts`；
- tau2 training wrapper：`tau2-bench/.pi/extensions/agent-r1-training.ts`；
- Pi 官方 loop：`pi/packages/coding-agent/src/core/agent-session.ts` 和
  `pi/packages/agent/src/agent-loop.ts`。

### 14.1 Agent-R1 为每条样本启动独立 Pi session

`PiTauAgentFlow.run()` 从 parquet sample 的 `extra_info.task_id` 取 task ID，为该轨迹
生成新的 `session_id`，启动一个 Node sidecar，并发送：

```json
{
  "type": "start_session",
  "session_id": "<uuid>",
  "task_id": "<tau2 task id>",
  "tau2_root": "<tau2 checkout>",
  "training_extension": "<agent-r1-training.ts>",
  "agent_dir": "<tau2 .pi/agent>",
  "pi_coding_agent_entrypoint": "<coding-agent/dist/index.js>",
  "max_turns": 30
}
```

canonical tau2 bridge 把 active task/DB 保存在 extension process scope，所以 sidecar
硬限制一个进程只能有一个活跃 trajectory。并发由 Agent-R1 启多个独立 sidecar
实现，不能在同一个进程的全局 tau2 state 中复用多个 session。

### 14.2 Pi 装载 skill、extension、task 和 tools

sidecar 用 `DefaultResourceLoader` 只加载 training wrapper extension：

```js
const resourceLoader = new DefaultResourceLoader({
  cwd: tau2Root,
  agentDir,
  noExtensions: true,
  additionalExtensionPaths: [trainingExtension],
});
await resourceLoader.reload();
```

随后创建带 bridge provider 的 `ModelRuntime` 和真实 Pi session。`bindExtensions()`
触发 `session_start`：

1. tau2 extension 启动 Python `tau2.domains.telecom.pi_bridge`；
2. `describe_tools` 返回 Telecom tool descriptors；
3. extension 用 `pi.registerTool()` 注册工具；
4. `load_task(task_id)` 初始化 task 和 DB；
5. task allowlist 选出 active tools，并调用 `pi.setActiveTools()`；
6. training wrapper 发布 canonical task prompt；
7. sidecar 验证 task ID 和非空 prompt，再显式 await `session.prompt()`。

canonical prompt 形态为：

```text
/skill:telecom-solo-support Task ID: <task_id>
Policy mode: workflow

<tau2 ticket>
```

`expandPromptTemplates: true` 让 Pi 自己展开 `/skill:telecom-solo-support`。因此训练
看到的是 Pi 实际服务时使用的 skill 内容，不是 Agent-R1 另写的一份 system prompt。

### 14.3 Pi 构造真实 LLM context

实际代码定位（行号对应本次文档提交时的代码）：

- Pi 官方 context 构造和 `streamFunction` 调用：
  `pi/packages/agent/src/agent-loop.ts:279-310`；其中 `295-300` 组装
  `systemPrompt + messages + tools`，`306-310` 调用实际 stream function；
- Pi 官方 turn/tool loop：`pi/packages/agent/src/agent-loop.ts:211-243`；其中
  `211-213` 等待 assistant response，`221-240` 提取 tool calls、执行工具并把
  tool results 放回当前 context；
- sidecar 将 Pi context 转成 OpenAI/Hermes message/tool schema：
  [`pi_sidecar/src/main.mjs:72-112`](./pi_sidecar/src/main.mjs#L72-L112)；
- sidecar 接管 Pi stream 并发出 `generation_request`：
  [`pi_sidecar/src/main.mjs:279-318`](./pi_sidecar/src/main.mjs#L279-L318)，其中
  `298-306` 是传给 Agent-R1 的 `messages`、`tools` 与 `anchor_obs`。

Pi 官方 agent loop 从 session transcript 构造：

```ts
{
  systemPrompt,
  messages,
  tools
}
```

其中 `messages` 已包含展开后的 user/task prompt，以及后续 turn 的 assistant tool call
和 tool result；`tools` 是当前 task allowlist 激活后的真实 schema。

Pi 调用 `session.agent.streamFunction(BRIDGE_MODEL, context, options)`。sidecar 覆写的
stream function 将 Pi message 类型转换为 OpenAI/Hermes 可消费结构：

- Pi user -> OpenAI `role=user`；
- Pi assistant text/toolCall -> OpenAI `role=assistant` + `tool_calls`；
- Pi toolResult -> OpenAI `role=tool`，保留 `tool_call_id` 和 tool name；
- Pi active tools -> OpenAI function tools，保留 name、description、parameters。

然后通过 stdout JSONL 发给 Python：

```json
{
  "type": "generation_request",
  "id": "<request id>",
  "session_id": "<session id>",
  "generation_id": "<turn generation id>",
  "messages": ["<normalized Pi context>"],
  "tools": ["<active Telecom schemas>"],
  "anchor_obs": "<canonical messages + sorted tool names>"
}
```

`anchor_obs` 用于保留这一 Pi 状态的稳定锚点；真正喂给模型的 token 仍由同一个
`messages + tools` 生成。

### 14.4 Agent-R1 执行实际模型生成

实际代码定位：

- Agent-R1 消费 sidecar event 的完整 generation 分支：
  [`pi_tau_agent_flow.py:121-193`](./pi_tau_agent_flow.py#L121-L193)；
- 读取 Pi 的 `messages/tools` 并生成 `prompt_ids`：
  [`pi_tau_agent_flow.py:126-143`](./pi_tau_agent_flow.py#L126-L143)；
- **真正调用 student rollout LLM** 的 `server_manager.generate()`：
  [`pi_tau_agent_flow.py:144-148`](./pi_tau_agent_flow.py#L144-L148)；
- 截取 response token、decode 并用 Hermes parser 提取 tool calls：
  [`pi_tau_agent_flow.py:149-170`](./pi_tau_agent_flow.py#L149-L170)；
- 保存 token/logprob，并将解析后的 response 回给 Pi：
  [`pi_tau_agent_flow.py:172-193`](./pi_tau_agent_flow.py#L172-L193)；
- `apply_chat_template(messages, tools=tools)` 的底层实现：
  [`agent_r1/agent_flow/agent_flow.py:188-251`](../../agent_r1/agent_flow/agent_flow.py#L188-L251)，
  其中纯 tokenizer 路径在 `237-246`，明确把同一份 `tools` 传给模型模板。

Python `PiTauAgentFlow` 收到 `generation_request` 后执行：

```python
prompt_ids = await self.apply_chat_template(messages, tools=tools)
output = await self.server_manager.generate(
    request_id=uuid4().hex,
    prompt_ids=prompt_ids,
    sampling_params=sampling_params,
)
response_ids = output.token_ids[: self.response_length]
parsed_text, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
```

这里有三个训用一致关键点：

1. prompt 直接来自 Pi 当前 `systemPrompt + transcript + active tools`；
2. tokenizer 的 `apply_chat_template(messages, tools=tools)` 与 rollout model 使用同一
   tools 参数，不在 sidecar 预先渲染第二份 prompt；
3. parser 使用当前 verl multi-turn 配置中的 Hermes tool parser，与 Qwen3 profile
   的工具调用格式一致。

`server_manager.generate()` 才是实际的 student LLM 调用。Pi 没有自己访问一个外部
模型服务；它等待 Agent-R1 返回这一 generation 的 token 结果。

Agent-R1 同时记录：

```text
prompt_ids
response_ids
response_logprobs
routed_experts（若模型返回）
anchor_obs
decoded response text
```

随后把 parser 结果回复给 sidecar：

```json
{
  "type": "response",
  "response_to": "<generation_request.id>",
  "ok": true,
  "result": {
    "text": "<assistant text>",
    "tool_calls": [
      {"id": "<id>", "name": "<tool>", "arguments": {}}
    ],
    "stop_reason": "toolUse"
  }
}
```

没有 tool call 时，`tool_calls=[]`，`stop_reason=stop`。

### 14.5 Pi 接回 assistant message 并执行 tau2 tool

sidecar 把响应重新组装成 Pi assistant message event stream。Pi 官方 loop 消费
`done` event，将 assistant message 写入 transcript。如果包含 `toolCall`：

```text
Pi agent loop
  -> 找到已注册的同名 Telecom tool
  -> extension execute(toolCallId, arguments)
  -> JSONL RPC: tau2 pi_bridge.call_tool
  -> TelecomEnvironment 修改/读取 DB
  -> tool result 返回 Pi
  -> Pi 将 toolResult 追加到 transcript
  -> 下一轮再次构造包含 tool result 的 LLM context
```

tool 不会返回 Agent-R1 Python 再由另一套环境执行。Agent-R1 只在下一次
`generation_request.messages` 中看到 Pi 已确认并落入 transcript 的结果，从而避免
双环境、双 DB 或 tool side effect 重放。

每个 assistant turn 结束时 sidecar 发送：

```json
{
  "type": "step_complete",
  "session_id": "<session>",
  "generation_id": "<generation>",
  "reward": 0,
  "terminated": false,
  "truncated": false,
  "assistant_message": {},
  "tool_results": [],
  "diagnostics": {
    "generation_requests": 1,
    "completed_turns": 1,
    "tool_turns": 1,
    "text_turns": 0
  }
}
```

`PiTrainingRecorder.record_turn()` 用 `generation_id` 将之前记录的 token/logprob 与
Pi 最终接受的 assistant message、tool results 和 diagnostics 对齐。没有对应
`step_complete` 的 pending generation 会在 `finalize()` 中触发错误，不会进入训练。

### 14.6 session shutdown、tau2 reward 和 AgentFlowStep

Pi loop 停止后，sidecar 显式发送 `session_shutdown`。tau2 extension 调用唯一的
canonical evaluator，training wrapper 将结果写成 `agent-r1-evaluation` entry；sidecar
依次转发：

```text
evaluation_result { result: <tau2 evaluator JSON> }
session_complete   { turns, diagnostics, truncated }
```

Python 只有同时收到完整 turns 和 evaluator result 后才调用：

```python
steps = recorder.finalize(evaluation)
processed_steps = [await self._postprocess(step, **kwargs) for step in steps]
```

recorder 把 terminal reward 只附到最后一个完成 step，并保存完整 evaluation 到
`reward_extra_info`。前面的 tool-use turns reward 为 0 是 credit assignment 结构，
不是 evaluator 缺失。

`AgentFlowBase` 随后完成训练 tensor 组装：

- `prompt_ids` 与 `response_ids` 形成 rollout 序列；
- response region 默认 mask 为 1，再乘有效 attention mask；
- rollout logprob 与 teacher logprob 在同一 response token 区域比较；
- terminal tau2 reward 进入 reward tensor；
- batch 进入 OPD loss、反向传播、optimizer step 和 checkpoint 保存。

### 14.7 4B teacher 在链路中的准确位置

Qwen3-4B teacher 不参与 Pi agent loop，也不会替 student 决定 tool call。真实在线
交互始终由 0.6B student 的 `server_manager.generate()` 产生；等整批 on-policy
trajectory 组装完成后，`AgentFlowManager._compute_teacher_logprobs()` 才执行 teacher
scoring：

```python
sequence_ids = input_ids[index][attention_mask[index]].tolist()
response_length = int(response_mask[index].sum().item())
teacher_ids, teacher_logprobs = (
    await self.teacher_server_manager.compute_teacher_logprobs_single(
        sequence_ids=sequence_ids,
        routing_key=routing_key,
    )
)
```

随后代码强制检查：

```python
if teacher_ids.ndim != 1 or not torch.equal(teacher_ids, expected_ids):
    raise RuntimeError("Teacher token IDs do not match the student rollout sequence ...")
```

也就是说 teacher 只对 student 已经实际走过 Pi/tau2 的同一条 token sequence 计算
log probability；它不能重新 tokenize 成另一条序列，也不能改写 transcript。对齐后
的 `teacher_logprobs` 被 pad 回 batch 布局，OPD loss 再在 `response_mask` 指定的
student response 区域比较 student/teacher logprob。P1 的 4B KV cache 调整解决的
正是这一离线 scoring engine 的启动容量，不是 Pi 的 agent-loop 显存。

### 14.8 训用一致检查表

| 一致性维度 | 训练路径中的唯一来源 | 当前保护机制 |
|---|---|---|
| prompt | Pi 的 system prompt、skill 展开和 live transcript | sidecar 只转换 Pi context；Agent-R1 对同一 `messages` 应用 chat template |
| tools | tau2 extension 注册、task allowlist 激活后的 Pi tools | 每次 generation 都从 `context.tools` 传递完整 schema |
| parser | Agent-R1 当前 multi-turn Hermes parser | parser 结果重新构造成 Pi `toolCall`，真实 Pi test 验证执行 |
| state | Pi transcript + tau2 process-scoped task/DB | 一 trajectory 一 sidecar；tools 只由 Pi/tau2 执行一次 |
| reward | tau2 canonical evaluator | callback 与文件输出解耦；缺 evaluator result 直接失败 |
| terminal | Pi loop/turn limit 后显式 extension shutdown | pending generation、extension error、evaluation 缺失均禁止 finalize |
| OPD token | student 实际生成的 `prompt_ids/response_ids` | teacher IDs 必须逐 token 等于 student rollout sequence |

因此最终衔接不是“Pi 生成一段文本，再交给 Agent-R1 猜 prompt”，而是：Pi 拥有并
推进真实 agent state，Agent-R1 在每个 Pi LLM 边界同步接收规范化 context、生成并
记录确切 token，Pi 再执行真实 tools；最后 tau2 的唯一 evaluator 作为 terminal
reward 回到同一条 token trajectory。
