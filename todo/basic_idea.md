# Qwen3 Small-Agent Post-Training Plan

## P0：用 Qwen3-0.6B 跑通 τ-bench Telecom 的 OPD → GiGPO 闭环

> 文档状态：2026-08-29
>
> 当前阶段：工程可行性验证（smoke test），不是正式论文实验
>
> 当前目标：先证明模型、环境、蒸馏、RL 和 checkpoint 能组成可复现的完整链路

---

## 1. 当前决策

### 1.1 P0 固定配置

| 项目 | 当前选择 |
| --- | --- |
| Student | Qwen/Qwen3-0.6B |
| Teacher | Qwen/Qwen3-4B-Instruct-2507 |
| Environment | τ-bench Telecom |
| Trainer | Agent-R1 的 opd 分支 |
| Environment Adapter | PiTauEnv(AgentEnv) |
| Hardware | 2 × RTX 5090 32GB |
| Student mode | non-thinking |
| Primary reward | τ-bench 原生任务结果 |

P0 的核心问题不是：

> Qwen3-0.6B 能否达到论文级 Agent 性能？

而是：

> 在两张 RTX 5090 上，能否让 Qwen3-0.6B 完成 Telecom 多轮 rollout，并依次跑通 Base、OPD、GiGPO、OPD → GiGPO？

### 1.2 P0 暂不包含

以下内容先不进入首轮实现：

- FKL warmup
- 1.7B / 4B student scaling
- 8B / 14B teacher scaling
- LoRA
- teacher-side GiGPO
- reward shaping
- BFCL 和跨领域泛化
- Pi coding agent 的代码、文件和仓库任务
- GiGPO、StepPO、Process Reward
- 多 teacher、Memory Evolution、Skill Evolution

当前首先需要隔离并验证最短闭环。

---

## 2. 完整研究目标

长期研究问题是：

> 稀疏到稠密监督桥接，能否从可验证数学推理迁移到带用户模拟器、工具执行、状态变化和错误恢复的长程 Agent 环境？

完整目标流程为：

~~~text
Teacher qualification
        ↓
optional teacher-side RL
        ↓
FKL warmup
        ↓
OPD on student trajectories
        ↓
student-side GiGPO
~~~

即：

$$
\boxed{
\text{Qualified Teacher}
\rightarrow
\text{FKL}
\rightarrow
\text{OPD}
\rightarrow
\text{GiGPO}
}
$$

但是，当前 Agent-R1 并未提供可以直接使用的 FKL pipeline，P0 因此采用：

$$
\boxed{
\text{Base}
\quad vs \quad
\text{GiGPO}
\quad vs \quad
\text{OPD}
\quad vs \quad
\text{OPD}\rightarrow\text{GiGPO}
}
$$

FKL 在 P0 完成后单独实现和验证。

---

## 3. 核心假设

### 3.1 Small-agent exploration cold start

小模型直接进行 Agentic GiGPO 时，可能出现：

$$
\pi_{\text{small}}
\rightarrow
\text{invalid or low-quality trajectories}
\rightarrow
R(\tau)\approx 0
$$

同一个 task 的多条 rollout 可能满足：

$$
R_1 \approx R_2 \approx \cdots \approx R_G
$$

当 group 内奖励方差接近零时，GiGPO 几乎得不到有效的相对优势：

$$
A_i=
\frac{R_i-\mu_R}{\sigma_R+\epsilon}
$$

### 3.2 OPD 的作用

OPD 不让 teacher 重新执行一条独立 trajectory，而是在 student 实际访问的状态上提供监督：

$$
s_t\sim\pi_S
$$

~~~text
Student observes state
        ↓
Student samples action
        ↓
Environment executes the action
        ↓
Teacher scores the student action on the same state
        ↓
Student receives dense token-level supervision
~~~

P0 真正要验证的是：

> OPD 是否能减少无效工具调用和完全失败轨迹，使后续 GiGPO 获得非零的组内奖励方差？

### 3.3 必须直接观测的学习信号

仅报告最终 Success Rate 不足以证明上述机制。至少记录：

- group reward mean
- group reward standard deviation
- reward_std = 0 的 group 比例
- 每组至少存在一条成功轨迹的比例
- 有效非零 advantage 的 trajectory/token 比例
- teacher-student action log-prob gap
- valid tool call rate
- invalid tool call rate
- repeated tool call rate
- failure recovery rate
- average steps

---

## 4. 为什么 P0 选择 Qwen3-0.6B

### 4.1 优势

- 模型足够小，适合优先验证两卡资源编排。
- student actor、optimizer、rollout engine 的显存压力较低。
- 和 Qwen3 teacher 属于同一模型家族，tokenizer 和工具调用格式更容易对齐。
- 如果 0.6B 能产生少量有效 trajectory，能够快速检验 OPD 是否改善 GiGPO 的冷启动。

### 4.2 局限

Qwen3-0.6B 的 Telecom 能力可能过低，可能出现：

- 无法稳定遵循 Telecom policy；
- 工具选择和参数生成失败；
- 对长对话状态跟踪不足；
- teacher supervision 后仍然没有成功轨迹；
- GiGPO group reward 长期全零。

因此：

> Qwen3-0.6B 的失败不能直接否定 FKL/OPD/GiGPO 方法。

P0 失败时应先区分：

1. 工程链路失败；
2. 环境或 reward 错误；
3. teacher 不够强；
4. 0.6B student 能力下限过低；
5. OPD 确实没有形成可用的探索桥梁。

### 4.3 升级条件

完成 P0 后，如果满足以下条件，再将 student 升级为 Qwen/Qwen3-1.7B：

- PiTauEnv rollout 稳定；
- OPD loss 有限且能下降；
- checkpoint 可以正确保存、加载和继续训练；
- GiGPO 至少在部分 group 上产生非零 reward variance；
- 0.6B 的瓶颈主要来自能力，而不是代码或资源问题。

---

## 5. Teacher 选择与资格验证

### 5.1 P0 Teacher

~~~text
Qwen/Qwen3-4B-Instruct-2507
~~~

选择原因：

- Agent-R1 OPD 官方示例默认使用该 teacher；
- 模型是纯 non-thinking，便于和工具调用格式对齐；
- 相比 0.6B student 具有明显容量差距；
- 在单张 32GB GPU 上进行短上下文 teacher inference 更现实。

### 5.2 Teacher qualification gate

不能仅根据参数量假设 teacher 有效。

训练前必须在同一 Telecom validation split 上评测：

| 模型 | Success | Valid Tool | Invalid Tool | Avg Steps |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B |  |  |  |  |
| Qwen3-4B-Instruct-2507 |  |  |  |  |

只有 teacher 明显优于 student，OPD 才有合理前提。

如果 4B teacher 在 Telecom 上不够强，按以下顺序处理：

1. 检查 prompt、policy、tool schema 和 non-thinking 配置；
2. 检查 teacher 是否能正确调用 Telecom 工具；
3. 尝试更强的 Qwen3 teacher；
4. 必要时先对 teacher 做任务侧训练；
5. 增加 raw teacher 与 task-improved teacher 对照。

完整研究不能忽略 teacher quality。已有 sparse-to-dense 工作的关键结论之一，就是 raw teacher 和经过任务强化的 teacher 可能产生完全不同的迁移效果。

---

## 6. τ-bench Telecom 数据与环境

### 6.1 为什么选择 Telecom

Telecom 同时包含：

- domain policy
- agent tools
- structured task state
- multi-turn user interaction
- user simulator
- environment-side state mutation
- task-level evaluation

它比单轮数学题更适合验证：

- 长程状态分布偏移；
- 工具调用错误；
- 失败后的恢复；
- student 访问状态上的 teacher supervision；
- sparse outcome reward 对小模型 exploration 的限制。

### 6.2 数据版本必须固定

τ-bench 的任务、grader 和内部 API 会演化。所有实验必须记录：

~~~yaml
tau_repo_url: ...
tau_commit: ...
tau_version: ...
domain: telecom
task_split: ...
task_ids: [...]
user_simulator_model: ...
user_simulator_temperature: ...
user_simulator_seed_set: [...]
max_turns: ...
~~~

不同 τ-bench 版本的结果不能默认直接比较。

### 6.3 数据划分

必须区分：

- smoke/dev tasks：用于接口调试；
- train tasks：用于 OPD/GiGPO；
- validation tasks：用于训练过程选择；
- held-out test tasks：仅用于最终评测。

禁止在训练轨迹中使用最终 test task。

由于 user simulator 会引入随机性，正式评测时应：

- 对所有模型使用完全相同的 task IDs；
- 使用相同的 simulator model；
- 使用相同的 temperature；
- 使用配对的 seed set；
- 报告均值、方差和置信区间；
- 保存逐任务 trajectory，而不只保存聚合分数。

---

## 7. PiTauEnv 设计

### 7.1 继承关系

~~~python
@AgentEnv.register("pi_tau")
class PiTauEnv(AgentEnv):
    ...
~~~

PiTauEnv 只负责将 τ-bench 的任务和状态机适配为 Agent-R1 的：

~~~text
reset() -> Observation
step(Action) -> (Observation, reward, done, info)
~~~

不要在环境层实现训练算法。

### 7.2 reset 职责

reset() 至少完成：

1. 根据 task_id 加载 Telecom task；
2. 创建隔离的 Telecom backend/state；
3. 初始化 user simulator；
4. 加载 domain policy；
5. 生成第一轮 user message；
6. 返回完整 OpenAI-format messages；
7. 暴露当前可用的 tool schemas。

### 7.3 step 职责

step(action) 至少完成：

1. 解析 assistant 输出；
2. 区分 tool call、普通回复和 final answer；
3. 校验工具名与参数；
4. 执行 Telecom tool；
5. 更新环境数据库或会话状态；
6. 必要时推进 user simulator；
7. 调用 τ-bench evaluator；
8. 生成新的完整 conversation observation；
9. 返回 reward、done 和诊断信息。

建议 info 包含：

~~~python
{
    "task_id": ...,
    "turn": ...,
    "tool_name": ...,
    "tool_valid": ...,
    "argument_valid": ...,
    "state_changed": ...,
    "policy_violation": ...,
    "task_success": ...,
    "termination_reason": ...,
}
~~~

### 7.4 并发隔离

GiGPO 会对同一 task 同时采样多条 trajectory。每条 trajectory 必须拥有独立的：

- environment state
- user simulator state
- conversation history
- task database snapshot
- random seed

不同 rollout 之间不得共享可变 Telecom 状态。

### 7.5 Prompt 长度行为

Agent-R1 当前 AgentEnvLoop 在 prompt 超长时会提前停止，而不是自动安全截断。

因此 PiTauEnv 必须记录：

- 每一步 prompt tokens；
- 达到长度上限的 episode 数；
- 被长度限制提前终止的比例；
- observation 中哪些历史可以安全压缩。

P0 不先实现复杂 memory compression；先观察实际长度分布。

---

## 8. Reward 设计

### 8.1 P0 使用原生 outcome reward

首轮保持：

$$
R_{\mathrm{task}}=
\begin{cases}
1 & \text{task success}\\
0 & \text{task failure}
\end{cases}
$$

如果 τ-bench 原生 grader 返回更细粒度分数，应同时保存：

- raw grader result；
- 二值 success；
- action/policy correctness diagnostics。

### 8.2 P0 不加入正向 shaping

暂不加入：

~~~text
+0.05 valid tool call
+0.02 valid format
~~~

原因是这会混淆：

~~~text
OPD improvement
vs
reward engineering improvement
~~~

格式错误、非法工具和 policy violation 先作为诊断指标记录。只有确认纯 outcome reward 无法提供任何学习信号后，才设计单独的 shaping ablation。

---

## 9. Agent-R1 当前能力边界

### 9.1 分支边界

本地 Agent-R1 main 分支只公告 OPD 支持，实际实现位于独立 opd 分支。

实验必须固定：

~~~text
Agent-R1 branch/commit
OPD-compatible verl fork/commit
transformers version
vLLM version
PyTorch version
CUDA version
GPU driver
~~~

### 9.2 当前 OPD 路径

当前 OPD branch 已经能够：

- 在 student rollout 后计算 teacher log-prob；
- 将 teacher 信息对齐到 student 生成 token；
- 使用蒸馏 loss 更新 student；
- 为 teacher 分配独立 Ray resource pool；
- 保存能够继续训练的 student checkpoint。

但是必须实测多轮 AgentEnvLoop，不能由 GSM8K 示例直接推断 Telecom 已经兼容。

### 9.3 当前不应声称支持的内容

当前不能直接声称：

- forward_kl_topk 已可用；
- FKL warmup 已有 recipe；
- Telecom OPD 已跑通；
- LoRA OPD 已验证；
- 1 student GPU + 1 teacher GPU 一定可运行；
- OPD checkpoint 已能无缝进入两卡 GiGPO；
- Qwen3-0.6B 在 Telecom 上存在足够 exploration。

这些都是 P0 的待验证项。

---

## 10. P0 执行阶段

### Stage A — Runtime compatibility smoke

目标：确认 Qwen3 模型与 OPD 软件栈兼容。

验收：

- Qwen3-0.6B student 可以加载；
- Qwen3-4B-Instruct-2507 teacher 可以加载；
- tokenizer/chat template 一致可用；
- student 强制 non-thinking；
- teacher log-prob 能正确返回；
- 一次 optimizer update 完成；
- loss 非 NaN/Inf；
- 两张 GPU 无 OOM。

建议从以下约束开始：

~~~yaml
student_gpus: 1
teacher_gpus: 1
train_batch_size: 1-2
ppo_mini_batch_size: 1
max_prompt_length: 4096
max_response_length: 512
rollout_n: 1
student_vllm_gpu_memory_utilization: conservative
teacher_vllm_gpu_memory_utilization: conservative
param_offload: true
optimizer_offload: true
~~~

这里的数值是 bring-up 起点，不是正式超参数。

如果 Telecom prompt 经常超过 4096，再逐步升到 8192；不要一开始同时扩大 batch、context 和 rollout 数。

### Stage B — PiTauEnv deterministic smoke

目标：在不训练模型的情况下验证环境。

验收：

- 固定 task + 固定 seed 可以重复得到相同初始状态；
- 合法工具调用产生正确状态变化；
- 非法工具调用不会破坏环境；
- final answer 能正确结束 episode；
- reward 与 τ-bench evaluator 一致；
- 并发 rollout 状态互不污染；
- trajectory 中保存了完整 action/observation/reward；
- 超长 prompt 和异常终止有明确原因。

### Stage C — Base evaluation

固定评测：

~~~text
Qwen3-0.6B
Qwen3-4B-Instruct-2507
~~~

目的：

1. 保存 student baseline；
2. 判断 teacher 是否合格；
3. 检查 0.6B 是否至少能产生合法工具调用；
4. 得到 episode 长度和 context 分布；
5. 估计后续 rollout 成本。

### Stage D — OPD

P0 OPD 使用：

~~~yaml
student: Qwen/Qwen3-0.6B
teacher: Qwen/Qwen3-4B-Instruct-2507
rollout_n: 1
loss_mode: k1
use_policy_gradient: true
use_task_rewards: false
epochs: 1
~~~

核心数据流：

~~~text
Telecom task
    ↓
student rollout through PiTauEnv
    ↓
student prompt/action tokens
    ↓
teacher log-prob on aligned student sequence
    ↓
distillation update
    ↓
Qwen3-0.6B-OPD checkpoint
~~~

OPD 阶段必须记录：

- distillation loss；
- teacher/student log-prob；
- token mask 数量；
- prompt/response length；
- tool validity；
- task reward，但不把 task reward 混入首轮 distillation loss。

### Stage E — Direct GiGPO baseline

从原始 Qwen3-0.6B 启动：

~~~yaml
teacher: disabled
rollout_n: 4
reward: tau_outcome
~~~

P0 先使用 G=4，不要直接使用 G=8。

必须检查：

- group 内 task ID 是否相同；
- rollout seed 是否不同；
- reward 是否正确分组；
- advantage 是否存在非零值；
- 全零 group 比例；
- 是否出现 NaN；
- checkpoint 是否可恢复。

### Stage F — OPD → GiGPO

从：

~~~text
Qwen3-0.6B-OPD
~~~

启动和 Stage E 完全相同的 GiGPO 配置。

除初始化 checkpoint 外，保持：

- task IDs；
- seed set；
- rollout number；
- batch budget；
- max steps；
- sampling temperature；
- reward；
- validation protocol

一致。

这样才能回答：

> OPD 是否真的改善了后续 GiGPO 的探索和可学习性？

---

## 11. P0 实验矩阵

| ID | Model | Teacher | OPD | GiGPO | 目的 |
| --- | --- | --- | ---: | ---: | --- |
| T0 | Qwen3-4B-Instruct-2507 | - |  |  | teacher qualification |
| E0 | Qwen3-0.6B | - |  |  | student baseline |
| E1 | Qwen3-0.6B | - |  | ✓ | direct GiGPO baseline |
| E2 | Qwen3-0.6B | 4B-Instruct-2507 | ✓ |  | dense bridge endpoint |
| E3 | Qwen3-0.6B | 4B-Instruct-2507 | ✓ | ✓ | main P0 pipeline |

P0 不做 FKL，因此不能把 E3 命名为 FKL-OPD-GiGPO。

---

## 12. P0 验收标准

### 12.1 工程验收

- [ ] PiTauEnv.reset/step 测试通过
- [ ] Telecom tool schema 能进入 Qwen chat template
- [ ] non-thinking 输出格式稳定
- [ ] student rollout 可完成多轮 episode
- [ ] teacher log-prob 与 student token 对齐
- [ ] OPD 至少完成一个短训练 run
- [ ] OPD checkpoint 可重新加载
- [ ] direct GiGPO 至少完成一个短训练 run
- [ ] OPD checkpoint 可继续进入 GiGPO
- [ ] 训练无 NaN/Inf
- [ ] 两卡无持续 OOM
- [ ] trajectory、metrics、config 和版本信息均已落盘

### 12.2 学习信号验收

P0 不要求 0.6B 达到固定成功率，但至少需要回答：

- Base 是否能产生合法工具调用？
- Teacher 是否明显强于 student？
- OPD 是否降低 invalid tool call？
- OPD 是否改善 teacher-student log-prob gap？
- OPD 后是否出现更多非零 reward-variance group？
- OPD → GiGPO 是否优于 direct GiGPO 的早期学习信号？

如果所有 GiGPO group 始终全零，则 P0 不能宣称 RL 有效，只能宣称工程闭环是否完成。

---

## 13. 正式实验升级路线

### P1 — Qwen3-1.7B 主实验

P0 通过后，将 student 升级为：

~~~text
Qwen/Qwen3-1.7B
~~~

保持 Telecom、teacher、reward 和评测协议不变，重复：

~~~text
Base
GiGPO
OPD
OPD → GiGPO
~~~

Qwen3-1.7B 应作为更可能承担论文主结果的 student。

### P2 — 补充 FKL

在单独实现和验证 FKL 后增加：

~~~text
FKL
FKL → OPD
FKL → GiGPO
FKL → OPD → GiGPO
~~~

标准 Forward KL：

$$
L_{\mathrm{FKL}}
=
D_{KL}
\left(
\pi_T(\cdot|s_t^T)
\parallel
\pi_S(\cdot|s_t^T)
\right)
$$

其中：

$$
s_t^T\sim\pi_T
$$

如果只保存 teacher Top-K logits，必须明确这是近似 FKL，并定义：

- Top-K 概率如何重新归一化；
- tail mass 如何处理；
- teacher/student tokenizer 是否完全一致；
- padding 和 response mask；
- 是否只训练 action tokens；
- 与 full-distribution FKL 的误差验证。

不能把简单截断后的 Top-K cross entropy 直接写成精确 Forward KL。

### P3 — Teacher quality

在 1.7B student 上比较：

~~~text
raw/post-trained 4B teacher
task-improved 4B teacher
8B teacher
~~~

重点不是“teacher 越大越好”，而是：

> teacher 的 Telecom trajectory quality 和 state-conditional supervision quality 是否足够好？

### P4 — Student scaling

只保留必要规模：

~~~text
0.6B：pipeline floor
1.7B：main student
4B：scaling reference
~~~

所有规模至少比较：

~~~text
Base
GiGPO
OPD → GiGPO
~~~

### P5 — Generalization

Telecom 主结果稳定后，再考虑：

- τ-bench airline/retail；
- BFCL；
- 新工具 schema；
- 未见 policy；
- 不同 user simulator；
- Pi coding agent 的代码/文件/仓库任务。

---

## 14. 指标

### Capability

- Success Rate
- Average Reward
- Pass@1 / Pass@k
- policy compliance

### Agent behavior

- Average Steps
- Tool Calls / Task
- Valid Tool Call Rate
- Argument Validity
- Invalid Tool Call Rate
- Repeated Tool Call Rate
- Failure Recovery Rate
- Premature Termination Rate

### GiGPO trainability

- group reward mean/std
- zero-variance group ratio
- nonzero-advantage ratio
- policy entropy
- KL to initial policy
- positive trajectory rate

### OPD

- distillation loss
- teacher/student log-prob gap
- masked token count
- teacher confidence
- response length
- teacher throughput

### Efficiency

- TTFT
- Decode tok/s
- Wall-clock Time / Task
- Input/Output Tokens
- Peak GPU Memory
- GPU Seconds / Task
- Success / GPU-second

所有效率对比必须固定：

- GPU 型号；
- serving/training backend；
- precision；
- context limit；
- batch/concurrency；
- sampling parameters；
- task/seed set。

---

## 15. 两张 RTX 5090 的初始资源方案

### OPD

~~~text
GPU 0:
Qwen3-0.6B student
actor + rollout + training

GPU 1:
Qwen3-4B-Instruct-2507 teacher
vLLM inference
~~~

这只是目标资源布局。Agent-R1 OPD 官方示例默认使用 2 张 student GPU + 2 张 teacher GPU，因此一加一布局必须通过实际 smoke test。

优先调整顺序：

1. batch size；
2. prompt/response length；
3. vLLM memory utilization；
4. CPU parameter/optimizer offload；
5. agent workers；
6. rollout concurrency。

不要在首轮同时引入 LoRA，因为当前 OPD 路径没有现成的已验证 LoRA recipe。

### GiGPO

GiGPO 阶段关闭 teacher 后，两张 GPU 可重新分配给 student。

但是从一加一 OPD 切换到双卡 GiGPO 时，必须验证：

- checkpoint 格式；
- FSDP world size 变化；
- optimizer state 是否继续使用；
- 是否只加载 model weights 并重新初始化 optimizer；
- vLLM 权重同步；
- resume step 和 scheduler 行为。

不能只因为 checkpoint 文件存在，就称为“无缝进入 GiGPO”。

---

## 16. 主要风险与停止条件

| 风险 | 识别信号 | 处理 |
| --- | --- | --- |
| 0.6B 完全不会调用工具 | valid tool≈0 | 先查模板；仍失败则升 1.7B |
| Teacher 不合格 | teacher 与 student 接近 | 换更强 teacher 或先训练 teacher |
| OPD 栈与 Qwen3 不兼容 | load/log-prob/update 失败 | 固定并修复依赖，不启动正式实验 |
| 一加一 GPU OOM | bring-up 即 OOM | 缩短长度/减 batch/offload |
| Telecom 状态串扰 | 并发结果互相影响 | 修复环境隔离 |
| GiGPO 全零 advantage | zero-variance≈100% | 检查 reward，再判断是否升级 student |
| Prompt 经常超长 | episode 被提前终止 | 统计后再扩 context 或压缩 history |
| 结果不可复现 | seed/version 重跑漂移 | 固定 simulator 和 repo commit |

停止扩展实验矩阵的条件：

- teacher qualification 未通过；
- PiTauEnv deterministic/concurrency 测试未通过；
- OPD teacher token alignment 未验证；
- checkpoint 无法进入 GiGPO；
- 日志无法区分环境失败、格式失败和任务失败。

---

## 17. P0 最终交付物

P0 完成时至少应包含：

~~~text
PiTauEnv implementation
PiTauEnv tests
Telecom dataset adapter
pinned environment manifest
Base evaluation config/results
Teacher qualification results
OPD launch config/log/checkpoint
Direct GiGPO config/log/checkpoint
OPD→GiGPO config/log/checkpoint
trajectory samples
group reward variance report
GPU memory/runtime report
known failures
~~~

只有这些证据齐全，才可以写：

> Qwen3-0.6B 的 τ-bench Telecom OPD → GiGPO 工程闭环已经跑通。

在此之前，应统一表述为：

> 计划、实现中或静态验证通过。

---

## 18. 研究问题

### RQ0 — Can the pipeline run?

> Can Qwen3-0.6B complete a reproducible OPD → GiGPO training loop on τ-bench Telecom with 2 × RTX 5090?

### RQ1 — Does OPD improve GiGPO trainability?

$$
\text{GiGPO}
\quad vs \quad
\text{OPD}\rightarrow\text{GiGPO}
$$

重点观察 reward variance 和有效 advantage，而不只看最终 success。

### RQ2 — Does dense supervision help long-horizon recovery?

比较：

- invalid tool calls；
- repeated actions；
- policy violations；
- failure recovery；
- premature termination。

### RQ3 — Is FKL necessary before OPD?

在 P0 完成并实现 FKL 后比较：

$$
\text{OPD}
\quad vs \quad
\text{FKL}\rightarrow\text{OPD}
$$

### RQ4 — What is the performance-efficiency frontier?

$$
\text{Agent Performance}
\quad vs \quad
\text{Latency / GPU Cost}
$$

---

## 19. 当前推荐路线

~~~text
Step 1
Pin Agent-R1 OPD + verl + Qwen3 runtime

Step 2
Implement and test PiTauEnv

Step 3
Evaluate Qwen3-0.6B and Qwen3-4B-Instruct-2507

Step 4
Run Qwen3-0.6B OPD smoke

Step 5
Run Qwen3-0.6B direct GiGPO smoke

Step 6
Load OPD checkpoint and run GiGPO

Step 7
Audit reward variance, tool behavior and reproducibility

Step 8
Decide whether to upgrade the student to Qwen3-1.7B
~~~

当前唯一主配置：

$$
\boxed{
\text{Qwen3-0.6B}
\xleftarrow[\text{OPD}]{\text{Qwen3-4B-Instruct-2507}}
\text{Qwen3-0.6B-OPD}
\xrightarrow{\text{GiGPO}}
\text{Qwen3-0.6B-Agent}
}
$$

P0 的成功定义是：

> 用最小 student 跑通真实多轮 Agent 环境中的 dense-supervision-to-sparse-RL 闭环，并建立足够完整的证据链，为后续 1.7B 正式实验提供可靠基础。

---

## 20. 参考

- Qwen3-0.6B: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Qwen3-4B-Instruct-2507: <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
- Agent-R1 OPD branch: <https://github.com/AgentR1/Agent-R1/tree/opd>
- Agent-R1 OPD example: <https://github.com/AgentR1/Agent-R1/blob/opd/examples/gsm8k/run_opd.sh>
- τ-bench: <https://github.com/sierra-research/tau2-bench>
- Beyond GRPO and On-Policy Distillation: <https://arxiv.org/abs/2605.12483>
