#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
STUDENT_GPUS_PER_NODE="${STUDENT_GPUS_PER_NODE:-1}"
TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-1}"

"$PYTHON_BIN" scripts/check_opd_tokenizers.py --student "$STUDENT_MODEL" --teacher "$TEACHER_MODEL"

exec "$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
  "${common_args[@]}" \
  trainer.use_legacy_worker_impl=disable \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.n=1 \
  distillation.enabled=True \
  distillation.n_gpus_per_node="$TEACHER_GPUS_PER_NODE" \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TAU2_TEACHER_GPU_MEMORY_UTILIZATION:-0.45}" \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1)) \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.distillation_loss_coef="${OPD_LOSS_COEF:-1.0}" \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
  trainer.project_name="${PROJECT_NAME:-TAU2_TELECOM_OPD}" \
  trainer.experiment_name="${EXP_NAME:-qwen3_0p6b_teacher8b_opd}" \
  trainer.n_gpus_per_node="$STUDENT_GPUS_PER_NODE" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  "$@"

