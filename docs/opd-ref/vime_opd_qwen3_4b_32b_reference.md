# VIME OPD Reference: Qwen3-4B Student + Qwen3-32B Teacher

Source: https://github.com/vllm-project/vime/issues/327
Issue title: `[Spike] OPD E2E — Qwen3-4B Student + Qwen3-32B Teacher on GSM8K (vLLM teacher)`
Opened: 2026-07-08

> This file is a structured extraction of the public reference experiment, not a verbatim copy of the GitHub issue.

## 1. Experiment goal

- Student: `Qwen3-4B`
- Teacher: `Qwen3-32B`
- Task: GSM8K math reasoning
- Method: GRPO + On-Policy Distillation (OPD)
- Teacher backend: external vLLM service
- Student training backend: Megatron
- Student rollout backend: vLLM, colocated with training
- Training length: 500 rollout iterations

## 2. Reference hardware topology

- Node: single node
- GPUs: `8 × NVIDIA A800 80GB`

| GPU | Role |
|---|---|
| 0–3 | Qwen3-4B student training + student vLLM rollout |
| 4–7 | Qwen3-32B teacher vLLM |
| Student TP | 2 |
| Teacher TP | 4 |

Teacher server port: `13141`

## 3. Model and data layout

Reference paths in the issue:

- Teacher HF model: `/data/nfs_87/model/Qwen3-32B`
- Student HF model: `/data/nfs_87/model/Qwen3-4B`
- Student Megatron checkpoint: `/data/nfs_87/model/Qwen3-4B_torch_dist_padded`
- GSM8K train: `/data/nfs_87/data/gsm8k/train.parquet`
- GSM8K test: `/data/nfs_87/data/gsm8k/test.parquet`
- Eval set size: 1319

Qwen3-4B uses tied embeddings. The reference converts the HF checkpoint to `torch_dist` with:

```bash
python3 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /data/nfs_87/model/Qwen3-4B \
  --save /data/nfs_87/model/Qwen3-4B_torch_dist_padded \
  --padded-vocab-size 152064
```

Conversion uses TP=1; runtime training uses TP=2.

## 4. Teacher vLLM settings

```yaml
teacher:
  model: Qwen3-32B
  tensor_parallel_size: 4
  dtype: bfloat16
  gpu_memory_utilization: 0.85
  max_model_len: 8192
  host: 0.0.0.0
  port: 13141
```

Reference teacher GPU visibility:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
```

## 5. Student rollout / training settings

### Rollout

```yaml
rollout:
  num_rollout: 500
  rollout_batch_size: 32
  n_samples_per_prompt: 4
  rollout_max_response_len: 4096
  rollout_temperature: 0.8
  global_batch_size: 64
  shuffle: true
  reward_type: math
```

Each rollout iteration produces approximately:

```text
32 prompts × 4 samples = 128 trajectories
```

So the full 500-iteration reference run corresponds to about:

```text
500 × 128 = 64,000 trajectories
```

### Evaluation

```yaml
eval:
  interval: 50
  n_samples_per_prompt: 1
  max_response_len: 4096
  top_k: 1
```

### Parallelism / performance

```yaml
student_parallelism:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  context_parallel_size: 1
  expert_model_parallel_size: 1
  expert_tensor_parallel_size: 1
  sequence_parallel: true
  dynamic_batch_size: true
  max_tokens_per_gpu: 9216
```

### Student vLLM rollout engine

```yaml
student_vllm:
  rollout_num_gpus_per_engine: 2
  gpu_memory_utilization: 0.7
  max_num_seqs: 32
  max_cudagraph_capture_size: 16
```

## 6. OPD / GRPO settings

```yaml
algorithm:
  advantage_estimator: grpo
  use_opd: true
  opd_type: vllm
  opd_kl_coef: 1.0

  use_kl_loss: true
  kl_loss_coef: 0.0
  kl_loss_type: low_var_kl

  entropy_coef: 0.0
  eps_clip: 0.2
  eps_clip_high: 0.28
```

Teacher reward / OPD plumbing in the reference:

```yaml
reward:
  custom_rm_path: vime.rollout.on_policy_distillation.reward_func
  custom_reward_post_process_path: vime.rollout.on_policy_distillation.post_process_rewards
  rm_url: http://127.0.0.1:13141/inference/v1/generate
```

## 7. Optimizer

```yaml
optimizer:
  type: adam
  lr: 1.0e-6
  lr_decay_style: constant
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.98
```

## 8. Checkpoint settings

```yaml
checkpoint:
  save_interval: 50
  save_dir: /root/opd_checkpoints/qwen3-4b-opd
  megatron_to_hf_mode: bridge
```

## 9. Misc training settings

```yaml
misc:
  attention_dropout: 0.0
  hidden_dropout: 0.0
  accumulate_allreduce_grads_in_fp32: true
  attention_softmax_in_fp32: true
  attention_backend: flash
  actor_num_nodes: 1
  actor_num_gpus_per_node: 4
  colocate: true
  make_vocab_size_divisible_by: 128
```

## 10. OPD data flow

```text
Student rollout (vLLM)
  ↓
token sequence + student log-probs
  ↓
Teacher vLLM HTTP inference
  ↓
teacher log-probs for student-generated tokens
  ↓
post_process_rewards
  ↓
apply_opd_kl_to_advantages
  ↓
advantage -= opd_kl_coef × (student_logp - teacher_logp)
  ↓
GRPO policy update
```

## 11. Reference results

### GSM8K

| Model | Accuracy |
|---|---:|
| Qwen3-32B teacher | 88.6% |
| Qwen3-4B student before OPD | 78.8% |
| Qwen3-4B student after 500 OPD iterations | 85.6% |

- Absolute OPD gain: `+6.8 percentage points`
- Teacher–student gap before OPD: `9.8 pp`
- Teacher–student gap after OPD: `3.0 pp`
- Approximate gap closed: `70%`

### OPD health metric

```text
rollout/opd_reverse_kl
step 1:   0.216
step 499: 0.110
```

Approximate decrease: `49%`

### Speed on the reference machine

```text
8 × A800 80GB
~2 minutes / rollout iteration
```

## 12. Reference run procedure

1. Download Qwen3-4B and Qwen3-32B.
2. Prepare GSM8K train/test parquet files.
3. Convert Qwen3-4B HF checkpoint to Megatron `torch_dist`.
4. Start Qwen3-32B teacher as a vLLM service.
5. Start Ray for the student side.
6. Run Megatron student training with colocated vLLM rollout.
7. Run 500 rollout iterations.
8. Save every 50 iterations.
9. Export the resulting Megatron checkpoint back to HF format.
10. Evaluate on all 1319 GSM8K test samples.

## 13. Notes for adapting this reference to 3 × RTX PRO 6000 96GB

The reference itself is **8 × A800 80GB**. A possible reduced topology for your machine is:

```text
GPU 0:
  Qwen3-32B teacher
  vLLM
  TP=1

GPU 1-2:
  Qwen3-4B student
  Megatron training + vLLM rollout
```

This reduced 3-GPU topology is an adaptation, **not** the configuration reported in Issue #327.

Parameters that would likely need retuning:

- Teacher `tensor_parallel_size`: 4 → 1
- Student visible GPUs: 4 → 2
- `actor_num_gpus_per_node`: 4 → 2
- Potentially `rollout_batch_size`
- Potentially `vllm_max_num_seqs`
- Potentially `max_tokens_per_gpu`
- `gpu_memory_utilization`
- Wall-time expectation

Do not assume the reference's `~2 min/iteration` carries over directly.

## Original source

https://github.com/vllm-project/vime/issues/327
