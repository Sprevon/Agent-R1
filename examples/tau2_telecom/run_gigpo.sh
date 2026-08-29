#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ROLLOUT_N="${TAU2_GIGPO_ROLLOUT_N:-4}"

exec "$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
  "${common_args[@]}" \
  algorithm.adv_estimator=gigpo \
  algorithm.norm_adv_by_std_in_grpo="${AGENT_R1_NORM_ADV_BY_STD_IN_GRPO:-True}" \
  ++algorithm.gigpo.step_advantage_w="${AGENT_R1_GIGPO_STEP_ADVANTAGE_W:-1.0}" \
  ++algorithm.gigpo.mode="${AGENT_R1_GIGPO_MODE:-mean_std_norm}" \
  ++algorithm.gigpo.enable_similarity="${AGENT_R1_GIGPO_ENABLE_SIMILARITY:-False}" \
  ++algorithm.gigpo.similarity_thresh="${AGENT_R1_GIGPO_SIMILARITY_THRESH:-0.95}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${AGENT_R1_GIGPO_KL_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
  trainer.project_name="${PROJECT_NAME:-TAU2_TELECOM_GIGPO}" \
  trainer.experiment_name="${EXP_NAME:-qwen3_0p6b_gigpo}" \
  trainer.n_gpus_per_node="${STUDENT_GPUS_PER_NODE:-2}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  "$@"
