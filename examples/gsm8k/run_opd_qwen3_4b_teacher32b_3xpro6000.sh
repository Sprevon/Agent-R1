#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${VERL_ROOT:-}" ]]; then
  export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 1
  fi
}

OPD_PROFILE="${OPD_PROFILE:-smoke}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-32B}"
GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:-$HOME/data/gsm8k/train.parquet}"
GSM8K_VAL_PATH="${GSM8K_VAL_PATH:-$HOME/data/gsm8k/test.parquet}"
GSM8K_MAX_PROMPT_LEN="${GSM8K_MAX_PROMPT_LEN:-2048}"
STUDENT_GPUS_PER_NODE="${STUDENT_GPUS_PER_NODE:-2}"
TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-1}"
STUDENT_TP="${STUDENT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-1}"
PROJECT_NAME="${PROJECT_NAME:-AGENT_R1_OPD_3XPRO6000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/$PROJECT_NAME}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
RUN_TOKENIZER_CHECK="${RUN_TOKENIZER_CHECK:-true}"
TOKENIZER_LOCAL_FILES_ONLY="${TOKENIZER_LOCAL_FILES_ONLY:-false}"
PRINT_CONFIG_ONLY="${PRINT_CONFIG_ONLY:-false}"
STUDENT_PARAM_OFFLOAD="${STUDENT_PARAM_OFFLOAD:-True}"
STUDENT_OPTIMIZER_OFFLOAD="${STUDENT_OPTIMIZER_OFFLOAD:-True}"
RESUME_MODE="${RESUME_MODE:-auto}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\"]}"

case "$OPD_PROFILE" in
  smoke)
    : "${TRAIN_BATCH_SIZE:=2}"
    : "${PPO_MINI_BATCH_SIZE:=1}"
    : "${ROLLOUT_N:=2}"
    : "${MAX_RESPONSE_LEN:=512}"
    : "${AGENT_FLOW_WORKERS:=2}"
    : "${TOTAL_TRAINING_STEPS:=1}"
    : "${TOTAL_EPOCHS:=1}"
    : "${SAVE_FREQ:=1}"
    : "${TEST_FREQ:=-1}"
    : "${VAL_BEFORE_TRAIN:=False}"
    : "${STUDENT_GPU_MEMORY_UTILIZATION:=0.45}"
    : "${TEACHER_GPU_MEMORY_UTILIZATION:=0.75}"
    : "${STUDENT_MAX_NUM_SEQS:=4}"
    : "${TEACHER_MAX_NUM_SEQS:=4}"
    : "${MAX_NUM_BATCHED_TOKENS:=4096}"
    : "${ENFORCE_EAGER:=True}"
    : "${CUDAGRAPH_CAPTURE_SIZES:=null}"
    ;;
  capacity)
    : "${TRAIN_BATCH_SIZE:=8}"
    : "${PPO_MINI_BATCH_SIZE:=2}"
    : "${ROLLOUT_N:=4}"
    : "${MAX_RESPONSE_LEN:=2048}"
    : "${AGENT_FLOW_WORKERS:=8}"
    : "${TOTAL_TRAINING_STEPS:=3}"
    : "${TOTAL_EPOCHS:=1}"
    : "${SAVE_FREQ:=3}"
    : "${TEST_FREQ:=-1}"
    : "${VAL_BEFORE_TRAIN:=False}"
    : "${STUDENT_GPU_MEMORY_UTILIZATION:=0.55}"
    : "${TEACHER_GPU_MEMORY_UTILIZATION:=0.80}"
    : "${STUDENT_MAX_NUM_SEQS:=8}"
    : "${TEACHER_MAX_NUM_SEQS:=8}"
    : "${MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${ENFORCE_EAGER:=True}"
    : "${CUDAGRAPH_CAPTURE_SIZES:=null}"
    ;;
  pilot)
    : "${TRAIN_BATCH_SIZE:=32}"
    : "${PPO_MINI_BATCH_SIZE:=16}"
    : "${ROLLOUT_N:=4}"
    : "${MAX_RESPONSE_LEN:=4096}"
    : "${AGENT_FLOW_WORKERS:=32}"
    : "${TOTAL_TRAINING_STEPS:=10}"
    : "${TOTAL_EPOCHS:=1}"
    : "${SAVE_FREQ:=10}"
    : "${TEST_FREQ:=-1}"
    : "${VAL_BEFORE_TRAIN:=False}"
    : "${STUDENT_GPU_MEMORY_UTILIZATION:=0.65}"
    : "${TEACHER_GPU_MEMORY_UTILIZATION:=0.82}"
    : "${STUDENT_MAX_NUM_SEQS:=32}"
    : "${TEACHER_MAX_NUM_SEQS:=16}"
    : "${MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${ENFORCE_EAGER:=False}"
    : "${CUDAGRAPH_CAPTURE_SIZES:=[1,2,4,8,16]}"
    ;;
  full)
    : "${TRAIN_BATCH_SIZE:=32}"
    : "${PPO_MINI_BATCH_SIZE:=16}"
    : "${ROLLOUT_N:=4}"
    : "${MAX_RESPONSE_LEN:=4096}"
    : "${AGENT_FLOW_WORKERS:=32}"
    : "${TOTAL_TRAINING_STEPS:=500}"
    : "${TOTAL_EPOCHS:=3}"
    : "${SAVE_FREQ:=50}"
    : "${TEST_FREQ:=50}"
    : "${VAL_BEFORE_TRAIN:=True}"
    : "${STUDENT_GPU_MEMORY_UTILIZATION:=0.70}"
    : "${TEACHER_GPU_MEMORY_UTILIZATION:=0.85}"
    : "${STUDENT_MAX_NUM_SEQS:=32}"
    : "${TEACHER_MAX_NUM_SEQS:=16}"
    : "${MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${ENFORCE_EAGER:=False}"
    : "${CUDAGRAPH_CAPTURE_SIZES:=[1,2,4,8,16]}"
    ;;
  *)
    echo "Unknown OPD_PROFILE=$OPD_PROFILE; expected smoke, capacity, pilot, or full." >&2
    exit 2
    ;;
esac

REQUIRED_GPUS=$((STUDENT_GPUS_PER_NODE + TEACHER_GPUS_PER_NODE))
MAX_MODEL_LEN=$((GSM8K_MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1))
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-9216}"
EXPERIMENT_NAME="${EXP_NAME:-qwen3_4b_teacher32b_${OPD_PROFILE}}"
RUN_CHECKPOINT_DIR="${RUN_CHECKPOINT_DIR:-$CHECKPOINT_DIR/$EXPERIMENT_NAME}"

if (( PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE )); then
  echo "PPO_MINI_BATCH_SIZE must not exceed TRAIN_BATCH_SIZE." >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE." >&2
  exit 2
fi

if is_true "$PRINT_CONFIG_ONLY"; then
  RUN_PREFLIGHT=false
  RUN_TOKENIZER_CHECK=false
else
  require_file "$GSM8K_TRAIN_PATH"
  require_file "$GSM8K_VAL_PATH"
fi

if is_true "$RUN_PREFLIGHT"; then
  "$PYTHON_BIN" scripts/check_opd_3gpu_environment.py \
    --expected-gpus "$REQUIRED_GPUS" \
    --minimum-memory-mib 90000 \
    --storage-path "$RUN_CHECKPOINT_DIR"
fi

if is_true "$RUN_TOKENIZER_CHECK"; then
  tokenizer_args=(--student "$STUDENT_MODEL" --teacher "$TEACHER_MODEL")
  if is_true "$TOKENIZER_LOCAL_FILES_ONLY"; then
    tokenizer_args+=(--local-files-only)
  fi
  "$PYTHON_BIN" scripts/check_opd_tokenizers.py "${tokenizer_args[@]}"
fi

echo "Agent-R1 OPD launch summary"
echo "  profile: $OPD_PROFILE"
echo "  student: $STUDENT_MODEL ($STUDENT_GPUS_PER_NODE GPU, rollout TP=$STUDENT_TP)"
echo "  teacher: $TEACHER_MODEL ($TEACHER_GPUS_PER_NODE GPU, TP=$TEACHER_TP)"
echo "  prompts x samples: $TRAIN_BATCH_SIZE x $ROLLOUT_N"
echo "  max prompt/response/model length: $GSM8K_MAX_PROMPT_LEN/$MAX_RESPONSE_LEN/$MAX_MODEL_LEN"
echo "  steps/epochs: $TOTAL_TRAINING_STEPS/$TOTAL_EPOCHS"
echo "  checkpoint dir: $RUN_CHECKPOINT_DIR"

overrides=(
  "trainer.use_legacy_worker_impl=disable"
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=$GSM8K_TRAIN_PATH"
  "data.val_files=$GSM8K_VAL_PATH"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.max_prompt_length=$GSM8K_MAX_PROMPT_LEN"
  "data.max_response_length=$MAX_RESPONSE_LEN"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "data.shuffle=True"
  "data.seed=42"
  "actor_rollout_ref.model.path=$STUDENT_MODEL"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$STUDENT_PARAM_OFFLOAD"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$STUDENT_OPTIMIZER_OFFLOAD"
  "actor_rollout_ref.actor.optim.optimizer=AdamW"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=constant"
  "actor_rollout_ref.actor.optim.weight_decay=0.1"
  "actor_rollout_ref.actor.optim.betas=[0.9,0.98]"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.clip_ratio=0.2"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
  "actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_SAVE_CONTENTS"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$STUDENT_TP"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$STUDENT_GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER"
  "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CUDAGRAPH_CAPTURE_SIZES"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.max_num_seqs=$STUDENT_MAX_NUM_SEQS"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.prompt_length=$GSM8K_MAX_PROMPT_LEN"
  "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LEN"
  "actor_rollout_ref.rollout.temperature=0.8"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_k=1"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
  "actor_rollout_ref.rollout.agent.num_workers=$AGENT_FLOW_WORKERS"
  "actor_rollout_ref.rollout.agent.default_agent_flow=single_step_agent"
  "actor_rollout_ref.rollout.trace.backend=null"
  "critic.enable=False"
  "reward_model.enable=False"
  "custom_reward_function.path=recipes/gsm8k/reward_fn.py"
  "custom_reward_function.name=compute_score"
  "distillation.enabled=True"
  "distillation.n_gpus_per_node=$TEACHER_GPUS_PER_NODE"
  "distillation.nnodes=1"
  "distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL"
  "distillation.teacher_models.teacher_model.inference.name=vllm"
  "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP"
  "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION"
  "distillation.teacher_models.teacher_model.inference.enforce_eager=$ENFORCE_EAGER"
  "distillation.teacher_models.teacher_model.inference.cudagraph_capture_sizes=$CUDAGRAPH_CAPTURE_SIZES"
  "distillation.teacher_models.teacher_model.inference.max_num_seqs=$TEACHER_MAX_NUM_SEQS"
  "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  "distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_MODEL_LEN"
  "distillation.distillation_loss.loss_mode=k1"
  "distillation.distillation_loss.use_policy_gradient=True"
  "distillation.distillation_loss.use_task_rewards=False"
  "distillation.distillation_loss.distillation_loss_coef=1.0"
  "distillation.distillation_loss.loss_max_clamp=10.0"
  "distillation.distillation_loss.log_prob_min_clamp=-10.0"
  "distillation.distillation_loss.policy_loss_mode=vanilla"
  "distillation.distillation_loss.clip_ratio=0.2"
  "distillation.distillation_loss.clip_ratio_low=0.2"
  "distillation.distillation_loss.clip_ratio_high=0.28"
  "trainer.logger=$TRAINER_LOGGER"
  "trainer.project_name=$PROJECT_NAME"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=$STUDENT_GPUS_PER_NODE"
  "trainer.nnodes=1"
  "trainer.val_before_train=$VAL_BEFORE_TRAIN"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.default_local_dir=$RUN_CHECKPOINT_DIR"
  "trainer.resume_mode=$RESUME_MODE"
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
  "trainer.log_val_generations=0"
)

command=("$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo)
if is_true "$PRINT_CONFIG_ONLY"; then
  command+=(--cfg job)
fi
command+=("${overrides[@]}")
command+=("$@")
exec "${command[@]}"
