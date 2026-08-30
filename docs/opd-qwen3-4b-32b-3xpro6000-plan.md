# Agent-R1 OPD 实验计划：Qwen3-4B Student + Qwen3-32B Teacher

## 1. 实验定位

本实验在 Agent-R1 `opd` 分支上实现 Qwen3-4B student 向 Qwen3-32B teacher 的纯 On-Policy Distillation（OPD），任务为 GSM8K，目标硬件为单机 `3 × NVIDIA RTX PRO 6000 96GB`。

本实验参考 VIME issue #327，但不是对其系统实现的逐项复刻：

- 参考实验使用 VIME、Megatron student 和外置 vLLM teacher；
- 本实验使用 Agent-R1、veRL FSDP student 和独立 Ray teacher resource pool；
- 参考硬件为 `8 × A800 80GB`，本实验为 `3 × RTX PRO 6000 96GB`。

因此可比较模型、数据、采样规模、优化器和 OPD 信号，但吞吐、显存占用、训练后准确率不能预先视为等价复现结果。

参考来源：<https://github.com/vllm-project/vime/issues/327>

## 2. 实验问题与成功标准

### 2.1 主要问题

1. 单张 96GB GPU 能否以 BF16 稳定承载 Qwen3-32B teacher vLLM，并返回完整 student 序列的 token log-prob？
2. 两张 96GB GPU 能否同时承载 Qwen3-4B FSDP 训练和两个 TP=1 student rollout replica？
3. Agent-R1 的 `k1 + policy-gradient` 蒸馏路径能否在 500 个训练 step 内降低 student-teacher reverse-KL，并提高 GSM8K 全量 greedy accuracy？

### 2.2 最低成功标准

- tokenizer vocab、added vocab 和 special tokens 完全一致；
- student rollout token ID 与 teacher 返回 token ID 逐位置一致；
- 完成至少一次 rollout、teacher log-prob、backward、optimizer update 和 checkpoint save；
- loss、log-prob、梯度范数均无 NaN/Inf；
- 10-step pilot 没有 OOM、Ray actor 丢失或 teacher 请求失败；
- 500-step 正式实验保留可恢复 checkpoint、配置、日志、评测输出和环境清单；
- 最终结论使用独立全量 1319 题 greedy 评测，不只引用训练过程中的 reward。

### 2.3 目标成功标准

- `actor/distillation/k1_mean` 或等价 reverse-KL 健康指标整体下降；
- 训练后 student 的全量 GSM8K accuracy 高于训练前 baseline；
- 记录实际 step time、峰值显存和总墙钟，以便判断三卡方案的成本效率。

VIME 报告的 `78.8% → 85.6%` 只作为外部参照，不作为 Agent-R1 适配实验的硬性验收线。

## 3. 固定项与适配项

### 3.1 与参考实验保持一致

| 项目 | 值 |
|---|---|
| Student | `Qwen/Qwen3-4B` |
| Teacher | `Qwen/Qwen3-32B` |
| 数据集 | GSM8K train/test |
| 训练 step | 500 |
| 每 step prompt 数 | 32 |
| 每 prompt 采样数 | 4 |
| 每 step trajectory 数 | 128 |
| 最大 response | 4096 tokens |
| rollout temperature | 0.8 |
| actor global mini-batch | 64 trajectories |
| optimizer | AdamW/Adam-compatible |
| learning rate | `1e-6` |
| beta | `[0.9, 0.98]` |
| weight decay | `0.1` |
| LR schedule | constant |
| OPD reverse-KL coefficient | `1.0` |
| save/eval interval | 50 steps |

Agent-R1 会在 actor update 前将 `ppo_mini_batch_size` 乘以 `rollout.n`。因此设置 `ppo_mini_batch_size=16`、`rollout.n=4`，得到 64 trajectories 的全局 mini-batch。

### 3.2 因三卡硬件必须适配

| 项目 | 三卡配置 | 原因 |
|---|---:|---|
| Student GPU | 2 | 兼顾 FSDP 训练与两个 rollout replica |
| Student rollout TP | 1 | 4B 单卡可放下，避免小模型跨 PCIe TP |
| Teacher GPU | 1 | 32B BF16 权重约 61GiB，可先尝试单卡 96GB |
| Teacher TP | 1 | 避免占用第二张 teacher 卡，保留 student 吞吐 |
| Student backend | FSDP | Agent-R1 OPD 分支的实际训练路径 |
| Teacher backend | Agent-R1 内置 vLLM pool | 分支不使用 VIME 的外置 `rm_url` |

备用拓扑仅在单卡 teacher OOM 或 teacher 明显成为瓶颈时启用：

```text
GPU 0: Qwen3-4B student, FSDP world size 1, rollout TP=1
GPU 1-2: Qwen3-32B teacher, TP=2
```

切换备用拓扑后必须重新执行 smoke 和 pilot，不能直接续跑正式实验。

## 4. OPD 数据与损失链路

```text
32 prompts
  -> student rollout.n=4
  -> 128 on-policy responses
  -> teacher 对 prompt + student response 做 forward
  -> teacher token log-probs
  -> 校验 teacher_ids == student sequence_ids
  -> k1 = student_logp - teacher_logp
  -> token advantage = -k1
  -> clipped policy-gradient update
```

正式配置使用：

```yaml
algorithm.adv_estimator: grpo
algorithm.use_kl_in_reward: false
actor_rollout_ref.actor.use_kl_loss: false
distillation.distillation_loss.loss_mode: k1
distillation.distillation_loss.use_policy_gradient: true
distillation.distillation_loss.use_task_rewards: false
distillation.distillation_loss.distillation_loss_coef: 1.0
```

当 `use_task_rewards=false` 时，pinned veRL 会把最终 task-reward policy loss 置零，并按系数 1.0 使用纯蒸馏信号。当前配置与参考实验的纯 OPD 一致；若以后要扫 `0.5/2.0`，需要先确认或修改 pinned veRL 的系数实现，不能只改命令行。

## 5. 分阶段实验矩阵

启动脚本：`examples/gsm8k/run_opd_qwen3_4b_teacher32b_3xpro6000.sh`

### E0：环境与资产预检

不启动训练，检查：

- 3 张可见 GPU，单卡显存至少 90,000 MiB；
- `nvidia-smi topo -m` 保存到实验记录；
- Python、CUDA、PyTorch、vLLM、Ray、Transformers 版本；
- `verl.__version__` 包含 `agentr1.opd`；
- `Role.TeacherModel`、teacher manager 和 distillation loss 可导入；
- 模型、数据、checkpoint 盘空间和系统内存；
- student/teacher tokenizer 一致。

```bash
export VERL_ROOT=/path/to/verl-at-5779c7c
export PYTHON_BIN=/path/to/python
export STUDENT_MODEL=/path/to/Qwen3-4B
export TEACHER_MODEL=/path/to/Qwen3-32B

$PYTHON_BIN scripts/check_opd_3gpu_environment.py \
  --expected-gpus 3 \
  --minimum-memory-mib 90000 \
  --storage-path /path/to/checkpoints

$PYTHON_BIN scripts/check_opd_tokenizers.py \
  --student "$STUDENT_MODEL" \
  --teacher "$TEACHER_MODEL" \
  --local-files-only
```

验收：两个脚本均输出 `PASS`，否则不进入 E1。

### E1：训练前独立评测

对 student 和 teacher 分别进行全量 1319 题 greedy 评测，保存逐题输出：

```bash
$PYTHON_BIN scripts/eval_gsm8k_vllm.py \
  --model "$STUDENT_MODEL" \
  --data "$GSM8K_VAL_PATH" \
  --tensor-parallel-size 1 \
  --output outputs/eval/qwen3-4b-baseline.jsonl

$PYTHON_BIN scripts/eval_gsm8k_vllm.py \
  --model "$TEACHER_MODEL" \
  --data "$GSM8K_VAL_PATH" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --output outputs/eval/qwen3-32b-teacher.jsonl
```

验收：输出包含 1319 条逐题记录、summary、正确数和 accuracy；评测器接受 prompt 要求的 `####` 格式以及参考方案常见的 `\\boxed{}` 格式，并保存实际 prompt template 设置。

### E2：1-step smoke

```bash
OPD_PROFILE=smoke \
bash examples/gsm8k/run_opd_qwen3_4b_teacher32b_3xpro6000.sh
```

默认缩小到 batch=2、n=2、response=512、1 step，并启用 eager mode。该阶段只验证完整链路，不用于报告性能。

验收：成功保存 `global_step_1`，日志无 OOM/NaN/token mismatch。

### E3：容量爬坡

```bash
OPD_PROFILE=capacity \
bash examples/gsm8k/run_opd_qwen3_4b_teacher32b_3xpro6000.sh
```

默认 batch=8、n=4、response=2048、3 steps。观察 teacher 单卡剩余 KV cache、student rollout 唤醒峰值和 optimizer update 峰值。

验收：连续 3 step；每阶段 GPU 峰值至少保留约 3GiB 安全余量。

### E4：正式负载 pilot

```bash
OPD_PROFILE=pilot \
bash examples/gsm8k/run_opd_qwen3_4b_teacher32b_3xpro6000.sh
```

使用正式 batch=32、n=4、response=4096，运行 10 step。前 5 step 视为 warmup，用后 5 step 中位数估算正式墙钟：

```text
estimated_hours = median_step_minutes * 500 / 60 * 1.15
```

验收：10 step 全部完成；step time 没有持续恶化；teacher 请求队列可排空；无 Ray worker 重启。

### E5：500-step 正式实验

```bash
OPD_PROFILE=full \
TRAINER_LOGGER='["console","wandb"]' \
bash examples/gsm8k/run_opd_qwen3_4b_teacher32b_3xpro6000.sh
```

正式默认值：

- `train_batch_size=32`
- `rollout.n=4`
- `ppo_mini_batch_size=16`
- `max_response_length=4096`
- `total_training_steps=500`
- `total_epochs=3`
- `save_freq=50`
- `test_freq=50`
- `val_before_train=true`
- 最多保留 3 个 actor checkpoint
- `resume_mode=auto`

正式实验不得跳过 E4。若 E4 修改了任何影响数学结果的配置，必须在实验记录中单列。

### E6：checkpoint 合并与最终评测

```bash
$PYTHON_BIN -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CHECKPOINT_DIR/global_step_500/actor" \
  --target_dir "$CHECKPOINT_DIR/global_step_500/hf_model"

$PYTHON_BIN scripts/eval_gsm8k_vllm.py \
  --model "$CHECKPOINT_DIR/global_step_500/hf_model" \
  --data "$GSM8K_VAL_PATH" \
  --tensor-parallel-size 1 \
  --output outputs/eval/qwen3-4b-opd-step500.jsonl
```

报告至少包含：baseline accuracy、teacher accuracy、step-500 accuracy、绝对提升、teacher-student gap closed、reverse-KL 走势和实际墙钟。

## 6. 显存与吞吐调节顺序

遇到 OOM 时按以下顺序处理，每次只改一类参数：

1. 确认 OOM 属于 student rollout、student update 还是 teacher；
2. 降低对应 vLLM `gpu_memory_utilization`；
3. 启用或保持 `enforce_eager=true`；
4. 降低 `max_num_seqs`；
5. 保持正式 response 上限不变，先降低 train batch 做定位；
6. student update OOM 时保持 param/optimizer offload；
7. teacher 权重加载或单请求上下文仍 OOM 时，切换 teacher TP=2 备用拓扑。

正式 run 不应通过量化 teacher 来省显存，因为这会改变 teacher token distribution，偏离参考实验。

## 7. 必须监控的指标

### 正确性

- student/teacher token ID mismatch 次数必须为 0；
- response mask 有效 token 数；
- checkpoint 是否包含 model、optimizer 和 extra state；
- resume 后 global step 是否连续。

### 蒸馏健康度

- `actor/distillation/loss`
- `actor/distillation/abs_loss`
- `actor/distillation/k1_mean`
- `actor/distillation/student_logprob`
- `actor/distillation/teacher_logprob`
- `actor/distillation/advantage_mean`
- `actor/distillation/advantage_abs_mean`

### 系统

- 每 step 的 rollout、teacher forward、old log-prob、update actor 耗时；
- 三张卡峰值显存、GPU 利用率和功耗；
- teacher vLLM KV cache capacity；
- Ray actor restart/error；
- checkpoint 保存耗时和磁盘增长。

## 8. 复现资产清单

每次正式实验保存：

- Agent-R1 commit；
- pinned veRL commit；
- 启动命令和完整 Hydra resolved config；
- `pip freeze`、Python/CUDA/driver/GPU topology；
- 模型目录或 revision；
- tokenizer 检查 JSON；
- GSM8K parquet SHA-256；
- console/W&B 日志；
- checkpoint；
- baseline、teacher、final 三份逐题 JSONL 和 summary；
- 人工记录的异常、重试与参数变更。

证据等级：只有实际保存了环境、命令、日志、checkpoint 和评测结果，才能声称该配置已在三卡 PRO 6000 上复现；配置解析或 4080 静态检查只属于准备证据。

## 9. 时间与租赁决策

不要按参考实验的 `2 min/step` 直接购买固定时长。完成 E4 后按实测中位数估算，并至少增加 15% 的评测、checkpoint 和波动余量。

| Pilot 实测 step time | 500-step 规划墙钟（含 15%） |
|---:|---:|
| 3 min | 约 29 h |
| 4 min | 约 38 h |
| 5 min | 约 48 h |
| 6 min | 约 58 h |

若 teacher 单卡吞吐导致墙钟过高，应先比较备用拓扑的 10-step pilot，再决定是否扩租；不要只依据瞬时 GPU utilization 判断。
