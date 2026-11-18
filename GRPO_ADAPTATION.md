# Qwen3-VL-Reranker 的 GRPO 适配审计

更新日期：2026-11-18。本记录回答一个具体问题：已训练的 Qwen3-VL-Reranker-2B checkpoint-500 能否直接作为标准 token-level GRPO/DIVA-GRPO 的策略模型？

## 审计结论

**不能按当前 `generative_reranker` task type 直接运行标准 token-level GRPO。** checkpoint 底层确实来自 `Qwen3VLForConditionalGeneration`，但当前训练和推理路径不生成多 token 答案：它取最后位置的隐状态，使用 `yes` 与 `no` 词向量差得到一个 logit，再经 sigmoid 输出独立的 query-document 相关性分数。

ms-swift 的 reranker SFT 路径支持 pointwise/listwise loss；它在 vLLM 中把 `reranker` 和 `generative_reranker` 都映射为 `score/pooling` 任务。ms-swift GRPO 则通过 rollout engine 生成 completion、保存 token log-prob 并对多个 completion 计算组内 advantage。两条路径在输出形式和训练单位上不相同。

DIVA-GRPO 公开实现使用 Qwen2.5-VL-7B-Instruct、Ray/verl 和 vLLM，对每个问题采样多个自回归回答。因此，直接把 checkpoint-500 路径填入公开 DIVA 启动脚本不是有效的复现或迁移。

## 已确认的调用链

### 当前 reranker

```text
query/video + one candidate/image
  → Qwen3-VL chat template
  → Qwen3VLModel last hidden state at final position
  → dot(hidden, embedding["yes"] - embedding["no"])
  → sigmoid relevance score
  → sort all candidate scores
```

listwise SFT 把同一 query 的 1 个正例和 7 个负例分数组成多分类交叉熵。这与实际排序目标直接对齐，也已在官方 10k 上带来 +1.36pp Top-1。

### 标准 GRPO / DIVA-GRPO

```text
one multimodal prompt
  → sample K autoregressive completions
  → parse each completion and compute outcome reward
  → normalize rewards inside a rollout group
  → apply token-level PPO/GRPO ratios to generated tokens
```

DIVA 额外为原问题构造难度变体：同一变体的 rollout 共享 local `uid`，同一原题的所有变体共享 global `uid`。代码分别计算 local/global GRPO advantage，做 batch z-score 归一化和相对难度加权，然后组合两者。RRB 路径会对奖励范围过低的组缩放 advantage，避免小差异被过度放大。

## 三种可选适配路径

### A. 把 checkpoint 强制改为候选列表生成模型

在一个 prompt 中同时展开多个商品，要求生成 item_id 或选项编号，然后使用标准 token-level GRPO。这与 DIVA 的实现形式最接近，但 checkpoint-500 没有见过多文档列表输入或 item_id 生成目标，直接 rollout 很可能存在格式和任务冷启动。这条路线需要额外的生成式 bridge SFT，不再是对现有 reranker 的直接 RL。

### B. 对每个 query-document 采样 yes/no token

保留现有 pair prompt，对每个 pair 采样 `yes`/`no`。这在形式上可以变成 token-level GRPO，但优化的是独立二分类，会重新引入已在 pointwise Pilot 中观察到的负类偏置，也不直接表达候选间相对排序。不作为首选。

### C. 候选分数上的 categorical/bandit GRPO

对同一 query 的候选分数 logit 做 softmax，定义策略 `π(item | query, candidates)`；从该分布采样 K 个 item，选中真值奖励为 1，否则为 0，再用组内相对 advantage 优化所选 item 的 log-prob。

该路径能直接使用 checkpoint-500，不需要伪造多 token 生成任务，也可把不同 hard-negative 组合定义为 DIVA 难度变体。但它是候选选择上的 contextual-bandit GRPO，不是论文中的自回归 token-level GRPO。在单一正例和完整标签下，其期望梯度与 listwise 交叉熵方向相近但方差更大，必须与等计算量继续 listwise SFT 对照，否则无法证明 RL 本身的价值。

## 当前决策

先审计路径 C 的奖励方差，不直接训练。新增的 `scripts/audit_reranker_grpo_rollouts.py` 使用已保存的候选和预测分数：

1. 把 sigmoid yes/no 分数还原为 logit；
2. 对同一 query 的候选 logit 做带温度 softmax；
3. 针对多个 rollout group size 模拟采样；
4. 报告正例策略概率、策略熵、组内奖励零方差率和有效 advantage 比例。

如果大部分组的奖励都全 0 或全 1，普通 GRPO 没有可用组内信号，应停止路径 C；如果仅通过人为提高温度才获得方差，需要额外证明该探索不会损害当前已良好校准的排序。

## 奖励方差审计结果

2026-11-18 在固定 2,000-query item-disjoint 开发预测上完成审计。主配置 `temperature=1.0, group_size=8` 的结果为：

- 1,270 个候选命中 query 的正例平均策略概率为 `0.564079`；
- 预期有效组比例为 `0.636479`，超过预注册的 `0.50` 门槛；
- 模拟有效组比例为 `0.638287`，与解析值仅差 `0.001809`，低于 `0.03` 门槛；
- 平均绝对 advantage 为 `0.503746`，组内学习信号非退化。

因此路径 C 通过了“是否值得实现最小训练 Pilot”的准入门槛。这不是精度收益结论：36.35% 的候选命中组仍为零方差，而候选未命中的 730 个 query 必然没有正奖励。真实训练只使用每组必含真值的训练侧 1 正 + 7 负 hard-negative 数据，不使用这 2,000 条开发候选训练。

## 最小训练 Pilot 实现

`scripts/reranker_score_policy_grpo_plugin.py` 通过 ms-swift `--external_plugins` 注册 `score_policy_grpo` loss。对每个正例在首位的 8 候选组，它对 logits 做温度 1.0 的 softmax，独立采样 8 个候选动作，以是否采到第 0 位真值作为二值奖励，然后用组内标准化 advantage 优化动作 log-prob。零方差组贡献 0 梯度。

这是每批次重新 rollout 的单次 on-policy 更新；在 importance ratio 等于 1 的当前参数点，局部梯度与未截断 GRPO ratio objective 一致。Pilot 不重用 rollout、不引入参考模型 KL，因此继续准确称为 categorical score-policy/contextual-bandit GRPO，不称为 token-level GRPO。

配对入口 `scripts/run_reranker_score_policy_grpo_pilot.sh` 从已合并的 checkpoint-500 同时启动两个单卡 arm：

- 固定的 100 个训练侧 query，两个 arm 共用同一 JSONL 和 hash manifest；
- score-policy GRPO 与继续 listwise 同为 100 step、学习率 `1e-6`、LoRA rank 8；
- 两者写入独立目录，不修改 checkpoint-500；
- score-policy arm 保存每批 rollout 统计，实测 active-group rate 低于 0.50 或 loss 非有限时入口直接失败。

此 Pilot 的功能门槛是可训练、有效组比例不低于 0.50，且固定 item-disjoint 开发集不出现灾难性回退。是否有算法收益必须再比较 checkpoint-500、等 step listwise 和 score-policy 三者的配对预测；不能只因为训练 loss 下降就进入大规模实验。

## Pilot 通过门槛

奖励方差审计先在训练侧监控或 item-disjoint 开发预测上运行，不使用官方 10k 调参。路径 C 的最小训练 Pilot 只有同时满足以下条件才启动：

- 主配置预先固定为 `temperature=1.0` 和 `group_size=8`；候选命中 query 上 `expected_active_group_rate >= 0.50`，且模拟值与解析值的绝对差不超过 0.03；
- `temperature=0.5/2.0` 只用于诊断分数过尖或过平，不得在主配置失败后事后改用它们启动训练；
- 奖励为严格的 item_id 命中，格式或解释不产生额外奖励；
- 保留 checkpoint-500 不变，新训练使用独立输出目录；
- 同时运行等样本/等 step 的 listwise 继训对照；
- 只在固定 item-disjoint 开发集比较 Base SFT、继续 listwise 和 score-policy GRPO。

## 来源与实现映射

- Qwen3-VL-Reranker 官方 `Qwen3VLReranker.compute_scores`：<https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py>
- ms-swift Reranker 训练说明：<https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Reranker.md>
- ms-swift 模型注册与 `generative_reranker` head patch：<https://github.com/modelscope/ms-swift/blob/main/swift/model/register.py>
- ms-swift vLLM task mapping：<https://github.com/modelscope/ms-swift/blob/main/swift/infer_engine/vllm_engine.py>
- ms-swift GRPO trainer/rollout：<https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/grpo_trainer.py>
- DIVA-GRPO 论文：<https://arxiv.org/abs/2603.01106>
- DIVA-GRPO 官方代码：<https://github.com/Siaaaaaa1/DIVA-GRPO>
