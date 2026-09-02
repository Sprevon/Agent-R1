#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PI_TAU_SIDECAR_ENTRYPOINT="${PI_TAU_SIDECAR_ENTRYPOINT:-$PROJECT_DIR/recipes/tau2_telecom/pi_sidecar/src/main.mjs}"
TAU2_BENCH_ROOT="${TAU2_BENCH_ROOT:-/root/autodl-tmp/code/tau2-bench-official}"
if [[ ! -d "$TAU2_BENCH_ROOT/.pi/skills" && -d "$PROJECT_DIR/../tau2-bench/.pi/skills" ]]; then
  TAU2_BENCH_ROOT="$PROJECT_DIR/../tau2-bench"
fi
export TAU2_BENCH_ROOT
export PI_CODING_AGENT_ENTRYPOINT="${PI_CODING_AGENT_ENTRYPOINT:-/root/autodl-tmp/code/pi/packages/coding-agent/dist/index.js}"
export PI_TAU_TRAINING_EXTENSION="${PI_TAU_TRAINING_EXTENSION:-$TAU2_BENCH_ROOT/.pi/extensions/agent-r1-training.ts}"
export PI_AGENT_DIR="${PI_AGENT_DIR:-$TAU2_BENCH_ROOT/.pi/agent}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in /root/envs/toolcall/bin/python /root/miniconda3/bin/python python3; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No usable Python interpreter found; set PYTHON_BIN explicitly." >&2
  exit 1
fi
export TAU2_PI_PYTHON="${TAU2_PI_PYTHON:-$PYTHON_BIN}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/recipes:$TAU2_BENCH_ROOT/src:/root/autodl-tmp/code/verl${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_PATH="${PI_TAU_CONFIG_PATH:-$PROJECT_DIR/recipes/tau2_telecom/base.yaml}"
TRAIN_PATH="${TAU2_TRAIN_PATH:-$PROJECT_DIR/data/tau2_telecom/train.parquet}"
VAL_PATH="${TAU2_VAL_PATH:-$PROJECT_DIR/data/tau2_telecom/test.parquet}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B}"
MAX_PROMPT_LEN="${TAU2_MAX_PROMPT_LEN:-8192}"
MAX_RESPONSE_LEN="${TAU2_MAX_RESPONSE_LEN:-1024}"
TRAIN_BATCH_SIZE="${TAU2_TRAIN_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${TAU2_PPO_MINI_BATCH_SIZE:-1}"
AGENT_FLOW_WORKERS="${AGENT_FLOW_WORKERS:-2}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 1
  fi
}

require_file "$CONFIG_PATH"
require_file "$TRAIN_PATH"
require_file "$VAL_PATH"
require_file "$PI_TAU_SIDECAR_ENTRYPOINT"
require_file "$PI_CODING_AGENT_ENTRYPOINT"
require_file "$PI_TAU_TRAINING_EXTENSION"

if [[ ! -d "$PROJECT_DIR/recipes/tau2_telecom/pi_sidecar/node_modules" ]]; then
  echo "Pi sidecar dependencies are missing." >&2
  echo "Run: bash scripts/install_pi_tau_deps.sh" >&2
  exit 1
fi

common_args=(
  "data.train_files=$TRAIN_PATH"
  "data.val_files=$VAL_PATH"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LEN"
  "data.max_response_length=$MAX_RESPONSE_LEN"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "++data.apply_chat_template_kwargs.enable_thinking=False"
  "actor_rollout_ref.model.path=$STUDENT_MODEL"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=${TAU2_DTYPE:-bfloat16}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=${TAU2_LR:-1e-6}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${TAU2_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
  "actor_rollout_ref.rollout.max_num_seqs=${TAU2_MAX_NUM_SEQS:-8}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${TAU2_MAX_NUM_BATCHED_TOKENS:-4096}"
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1))"
  "actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LEN"
  "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LEN"
  "actor_rollout_ref.rollout.temperature=${TAU2_ROLLOUT_TEMPERATURE:-1.0}"
  "actor_rollout_ref.rollout.top_p=${TAU2_ROLLOUT_TOP_P:-1.0}"
  "actor_rollout_ref.rollout.multi_turn.format=hermes"
  "actor_rollout_ref.rollout.agent.agent_flow_config_path=$CONFIG_PATH"
  "actor_rollout_ref.rollout.agent.num_workers=$AGENT_FLOW_WORKERS"
  "actor_rollout_ref.rollout.agent.default_agent_flow=pi_tau_telecom_agent"
  "actor_rollout_ref.rollout.trace.backend=null"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "critic.enable=False"
  "reward_model.enable=False"
  "algorithm.use_kl_in_reward=False"
  "trainer.critic_warmup=0"
  "trainer.logger=[\"console\"]"
  "trainer.nnodes=1"
  "trainer.max_actor_ckpt_to_keep=3"
)
