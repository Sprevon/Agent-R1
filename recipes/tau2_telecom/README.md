# Pi harness + tau2 Telecom + OPD/GiGPO

This recipe keeps model generation and optimization in Agent-R1 while delegating the agent runtime to Pi's published
SDK. The Node sidecar instantiates `@earendil-works/pi-agent-core`'s `Agent`, loads Pi skills with its skill loader,
validates and executes tool calls through Pi, and sends only generation/environment requests over JSONL. Python does
not reproduce Pi's turn loop.

The local Pi 0.84.4 checkout also contains a higher-level `AgentHarness`, but its execution methods (including
`prompt`) currently raise `HarnessNotImplemented`. This integration therefore uses the runnable `Agent` loop plus the
harness skill/environment APIs. It can move up to `AgentHarness` after those methods become executable without
changing the Agent-R1/tau2 boundary.

## Fixed versions and model profiles

- Pi agent SDK: `@earendil-works/pi-agent-core==0.84.4`
- tau2: Git tag `v1.0.1`, with the public `AgentGymEnv`
- OPD verl: commit `5779c7c6782733f77ef640f557bea572dfeacc12`
- formal student: `Qwen/Qwen3-0.6B`, non-thinking
- formal teacher: `Qwen/Qwen3-8B`
- 3080 smoke teacher override: `Qwen/Qwen3-0.6B`

The pinned OPD implementation uses a dedicated Ray GPU pool for the teacher. A full OPD optimizer-update smoke
therefore needs two visible GPUs even when both models are 0.6B. Do not replace this with an ad-hoc Python teacher
loop merely to fit a one-GPU host.

## Runtime boundary

```text
Agent-R1 rollout server                 tau2 AgentGymEnv
       ^                                      ^
       | generation_request                   | environment_request
       v                                      v
                  Pi Agent SDK sidecar
          skills + transcript + tool runtime
```

Each GiGPO trajectory gets an independent `AgentGymEnv` and Pi `Agent` session. Sessions share only the long-lived
sidecar process. The `anchor_obs` used for GiGPO is a canonical serialization of the Pi-visible transcript and tool
names before the action.

## Prepare a Linux environment

Use Python 3.12 or 3.13 and Node 22.19 or newer. Install the repository's normal CUDA/PyTorch/vLLM dependencies first,
then install the pinned additions:

```bash
bash scripts/install_pi_tau_deps.sh
python3 scripts/check_pi_tau_environment.py --student Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-8B
npm --prefix recipes/tau2_telecom/pi_sidecar test
```

The installer adds the Pi npm packages, tau2, and the OPD-compatible verl pin. It assumes the repository's normal
training dependencies are already present; it does not select a CUDA/PyTorch/vLLM build for the machine.

tau2 drives a separate user simulator through LiteLLM. Configure its provider key and optionally override:

```bash
export TAU2_USER_LLM='openai/gpt-4.1-mini'
export TAU2_USER_LLM_ARGS='{"temperature":0.0}'
```

For an OpenAI-compatible local service, put `api_base`, `api_key`, and any provider-specific arguments in
`TAU2_USER_LLM_ARGS`. This user simulator is not the OPD teacher.

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

The generated `manifest.json` records the exact task IDs. Keep train tasks out of held-out final evaluation.

## 3080 smoke gate

On a host with two visible 3080-class GPUs, export `TAU2_USER_LLM` and simulator credentials first, then run the 0.6B/0.6B OPD update and a one-batch GiGPO update. The script calls `scripts/check_pi_tau_environment.py` before training and fails if Pi, tau2, OPD, CUDA, or the user simulator is missing:

```bash
bash examples/tau2_telecom/run_3080_smoke.sh all
```

The acceptance gate is stricter than process exit: inspect logs for finite loss, non-empty response masks, exact OPD
token alignment, at least one optimizer update, checkpoint creation, Pi session completion, and tau2 reward output.
The GiGPO log exposes singleton and zero-variance group fractions under `gigpo/*`.

## Dual-5090 formal runs

Run formal OPD with the default Qwen3-8B teacher:

```bash
bash examples/tau2_telecom/run_opd.sh
```

Run direct GiGPO from the base student:

```bash
bash examples/tau2_telecom/run_gigpo.sh
```

For OPD to GiGPO, point at the `hf_model` exported by the OPD actor checkpoint. This intentionally starts a fresh
GiGPO optimizer and avoids pretending that a one-student-GPU FSDP optimizer state can resume unchanged at world size
two:

```bash
OPD_HF_MODEL=/absolute/path/to/opd/checkpoint/actor/hf_model \
  bash examples/tau2_telecom/run_opd_then_gigpo.sh
```

Before each retained run, capture the exact software and hardware state:

```bash
python3 scripts/capture_pi_tau_manifest.py \
  --output artifacts/manifests/formal-opd.json \
  --student Qwen/Qwen3-0.6B \
  --teacher Qwen/Qwen3-8B \
  --task_split train
```

## Evidence status

Mac checks can validate Python compilation, JSONL routing, schema conversion, shell syntax, and Pi sidecar JavaScript
syntax. They cannot establish that tau2 simulations, vLLM rollout, OPD updates, GPU allocation, or checkpoint handoff
work. Record those as verified only after the 3080 gate succeeds, then repeat the formal profile on the dual 5090
machine.
