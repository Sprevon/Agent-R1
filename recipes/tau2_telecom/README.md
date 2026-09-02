# Pi harness + tau2 Telecom + Agent-R1 OPD/GiGPO

This recipe keeps LLM generation and RL training in Agent-R1, while using the
official Pi coding-agent runtime for the complete agent loop. tau2 remains the
single owner of Telecom task data, task initialization, database state, tool
implementations, and evaluation.

The important rule is:

```text
Agent-R1        generates tokens and builds training steps
Pi              owns skills, extensions, transcript, tools, and the agent loop
tau2            owns tasks, DB state, tool execution, and reward evaluation
```

## Two Agent-R1 paths

Agent-R1's generic environment path is still available for other recipes:

```text
AgentFlowManager
  └─ AgentEnvLoop
       └─ AgentEnv
            └─ ToolEnv (one possible AgentEnv implementation)
```

`AgentEnvLoop` is an `AgentFlow` implementation; it is not a mandatory layer
below every flow. The Telecom recipe uses a dedicated `PiTauAgentFlow` instead:

```text
AgentFlowManager
  └─ PiTauAgentFlow
       └─ PiSidecarClient
            └─ Pi createAgentSession()
                 ├─ DefaultResourceLoader
                 │    ├─ telecom-solo-support skill
                 │    └─ Agent-R1 training wrapper extension
                 ├─ official Pi agent loop
                 └─ tau2-telecom extension
                      └─ tau2 pi_bridge
                           └─ TelecomEnvironment + DB + tools
```

This bypasses `AgentEnvLoop` deliberately. A Pi turn is not just one text
action passed to `AgentEnv.step()`: Pi must append the assistant message,
execute the registered tool, append the tool result, and decide whether to
continue. Keeping that loop inside Pi avoids reimplementing Pi semantics in
Python.

## Runtime and training boundaries

The sidecar uses the official `@earendil-works/pi-coding-agent` SDK. It loads
the training wrapper through `DefaultResourceLoader`, binds extensions in
`rpc` mode, and replaces only `session.agent.streamFunction` so Agent-R1 can
provide the actual model generation.

```text
Pi session
  ├─ session_start
  │    └─ load tau2 task, register tools, activate task allowlist, publish skill prompt
  ├─ Agent-R1 sidecar awaits session.prompt(canonical_prompt)
  ├─ generation_request
  │    └─ Agent-R1 server_manager.generate()
  ├─ Pi executes tau2 tool calls internally
  ├─ step_complete
  │    └─ Agent-R1 records this Pi turn
  └─ session_shutdown
       └─ tau2 evaluator → evaluation_result
```

There are two different kinds of state:

1. **Task execution state** belongs to tau2. `tau2.domains.telecom.pi_bridge`
   owns the live `TelecomEnvironment`, task initial state, DB state, tool
   calls, assistant-text history, and final evaluator.

2. **Training recording state** belongs to Agent-R1. `PiTrainingRecorder`
   stores prompt/response token IDs, response logprobs, routed experts, Pi
   turn metadata, and the terminal reward. It does not call tau2 and does not
   calculate advantage.

The recorder produces `AgentFlowStep` objects. Agent-R1's normal batching and
PPO/GRPO/GiGPO code later converts the terminal reward into reward tensors and
computes advantages.

The current sidecar is one-trajectory-per-process because the canonical tau2
bridge keeps task/environment state at extension-process scope. This prevents
different trajectories from sharing a Telecom DB or active task.

## Training wrapper extension

The canonical interactive extension is:

```ts
const telecom = createTau2TelecomExtension();
export default telecom.extension;
```

Training needs to inject a task ID and receive evaluation results through the
Pi session, so it uses:

```text
agent-r1-training.ts
  └─ createTau2TelecomExtension({
       taskId,
       autoPrompt: false,
       failFastOnStartError: true,
       onTaskLoaded: task => pi.appendEntry(canonicalPrompt),
       onEvaluation: result => pi.appendEntry(...)
     })
```

The wrapper does not duplicate Telecom tools, skills, DB logic, or reward
logic. It only adapts canonical tau2 extension configuration and publishes
the canonical task prompt and evaluator result as Pi session entries for
Agent-R1. The sidecar awaits `session.prompt()` directly so model/provider,
skill expansion, and extension failures cannot silently produce a zero-turn
trajectory.

## Event protocol

The Python/Node boundary is JSONL. The training path uses:

```text
generation_request
  → response { text, tool_calls, stop_reason }
  → step_complete
  → ...
  → evaluation_result
  → session_complete
```

`environment_request` is intentionally not part of this path. Tool calls are
executed by Pi's registered tau2 extension, not by `PiTauAgentFlow` or a
second Python-side environment.

## Fixed versions and model profiles

- Pi coding-agent SDK: `@earendil-works/pi-coding-agent==0.84.4`
- tau2: Telecom task/bridge code in the configured `TAU2_BENCH_ROOT`
- OPD verl: the version pinned by this Agent-R1 checkout
- formal student: `Qwen/Qwen3-0.6B`, non-thinking
- formal teacher: `Qwen/Qwen3-8B`

The formal OPD setup uses a dedicated Ray GPU pool for the teacher. A complete
optimizer-update smoke therefore needs the configured multi-GPU environment;
CPU-only checks do not establish training correctness.

## Prepare a Linux environment

Run these commands on the remote Linux training host, not on the Mac checkout.
Python/Node dependencies and Pi's `coding-agent/dist/index.js` must be
available before a real Pi session can start.

```bash
bash scripts/install_pi_tau_deps.sh
python3 scripts/check_pi_tau_environment.py --student Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-8B
npm --prefix recipes/tau2_telecom/pi_sidecar test
```

Set the paths used by the split runtime when they are not at the defaults:

```bash
export TAU2_BENCH_ROOT=/root/autodl-tmp/code/tau2-bench-official
export PI_CODING_AGENT_ENTRYPOINT=/root/autodl-tmp/code/pi/packages/coding-agent/dist/index.js
export PI_TAU_TRAINING_EXTENSION="$TAU2_BENCH_ROOT/.pi/extensions/agent-r1-training.ts"
export PI_AGENT_DIR="$TAU2_BENCH_ROOT/.pi/agent"
export TAU2_PI_PYTHON=/root/envs/toolcall/bin/python
```

`TAU2_PI_PYTHON` must point to an interpreter that can import the configured
tau2 checkout. `examples/tau2_telecom/common.sh` defaults it to the resolved
`PYTHON_BIN`; export it explicitly when Pi's Node process does not inherit a
usable bare `python` command.

The Telecom bridge runs in tau2's solo workflow mode. It does not use a
separate user simulator in this training path.

## Data

Create the train/test parquet files from tau2's official Telecom RL splits:

```bash
python3 -m recipes.tau2_telecom.data_preprocess.process_tau2_telecom \
  --output_dir data/tau2_telecom --splits train test
```

Create a one-task-per-split dataset for the 3080 smoke:

```bash
bash examples/tau2_telecom/prepare_3080_smoke_data.sh
```

The generated `manifest.json` records the exact task IDs. Keep train tasks
out of held-out final evaluation.

## Three-GPU 4080 smoke gate

After the Pi checkout has its dependencies and build output, the startup-fit
Qwen3 0.6B-student/4B-teacher launch profile is:

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

The teacher needs `0.60` on the observed 32 GiB RTX 4080 SUPER: `0.45` left
only about 1.03 GiB for KV cache while the 24,129-token profile required about
3.32 GiB. Treat model and data paths above as host-specific inputs, not files
created by this recipe.

The acceptance gate is stricter than process exit. Inspect logs for:

- finite loss;
- non-empty `response_mask`;
- valid Pi tool calls and completed Pi session;
- tau2 evaluator output and reward breakdown;
- exact OPD token alignment;
- at least one optimizer update;
- checkpoint creation.

The GiGPO log also exposes singleton and zero-variance group fractions under
`gigpo/*`.

## Dual-5090 formal runs

Run formal OPD with the default teacher:

```bash
bash examples/tau2_telecom/run_opd.sh
```

Run direct GiGPO from the base student:

```bash
bash examples/tau2_telecom/run_gigpo.sh
```

For OPD to GiGPO, point at the `hf_model` exported by the OPD actor
checkpoint. This starts a fresh GiGPO optimizer and avoids resuming an FSDP
optimizer state at a different world size:

```bash
OPD_HF_MODEL=/absolute/path/to/opd/checkpoint/actor/hf_model \
  bash examples/tau2_telecom/run_opd_then_gigpo.sh
```

## Evidence status

CPU-only checks can validate Python compilation, JSONL routing, schema
conversion, shell syntax, and sidecar JavaScript syntax. They cannot prove
that Pi's real `createAgentSession` loads the extension, that tau2 executes a
complete trajectory, or that vLLM/OPD updates and checkpoint handoff work.

Record those as verified only after the real Pi session and the GPU smoke gate
both succeed.
