# 已验证脚本

2026-11-25：新增 [精排文本去重A/B](RERANKER_TEXT_ABLATION.md)，固定模型/候选/路由，仅比较原始文本与OCR/ASR完全重复段去重，支持空闲多卡。


首次正式提交后的CPU溯源与内部错误分解：[POST_SUBMISSION_AUDIT.md](POST_SUBMISSION_AUDIT.md)。新增审计入口，不修改冻结推理代码。


2026-11-24最新状态：正式50k七卡召回与精排已由用户确认完成；XLSX核验及正式提交待确认。见 [结果记录](../进展更新-2026-11-24-正式50k推理完成.md)。以下入口状态按各自日期理解。


最新入口：[正式50k推理与提交](OFFICIAL_SUBMISSION.md)。测试数据已由用户确认到位；新正式入口待远端GPU Pilot验证。


当前入口：[真实增量Pilot与正式数据准备](EVALUATION_READINESS.md)。动态Top-25离线结果已确认；新GPU入口待远端验证。


最新主线：[新召回动态Top-25](RETRIEVER_DYNAMIC_TOP25.md)。七卡Top-10三组对照已由用户完成；动态实验入口待远端执行。


2026-11-23当前主线：[三套召回的固定精排对照](RETRIEVER_RERANK_COMPARISON.md)，扩大训练暂停；新对照入口尚待远端GPU验证。


当前主线：[召回 embedding 1k 扩大实验](EMBEDDING_SCALE.md)。100-query Pilot 已由用户远端验证通过；1k 扩大入口尚待执行。


这些脚本根据 2026-11-04 至 2026-11-05 在远端机器实际运行成功的命令整理。本地目录不包含验证集、模型或官方仓库，需把 `scripts/` 同步到远端项目目录后运行。

远端环境：

- 数据检查：`conda activate ecom-retrieval`
- 模型推理：`conda activate ecom-infer`
- 数据：`/code/ecommerce-retrieval/data/eval/test_small_package`
- 模型：`/code/ecommerce-retrieval/models/Qwen3-VL-Embedding-2B`
- 官方代码：`/code/ecommerce-retrieval/Qwen3-VL-Embedding`

运行方法：

```bash
python scripts/audit_eval_data.py
python scripts/verify_images.py
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_embedding.py
```

`smoke_embedding.py` 会把不同宽高的帧按比例缩放并填充为 `512×512`。它还会要求预处理结果包含 `pixel_values_videos`，防止官方实现捕获视觉预处理异常后静默改为编码 `NULL` 文本。

已确认结果：验证集包含 20,000 个商品、10,000 个视频、10,000 条标签和 60,097 张 JPEG；JSON、ID、标签引用、图片路径和图片解码检查全部通过。首个标签样本的真实多帧 embedding 为 2,048 维，有限且归一化，单次前向约 0.745 秒，峰值显存约 4.1 GiB。

## 下一阶段：完整候选池基线

以下两份脚本已经完成本地语法检查，尚待在远端数据和 GPU 上首次运行：

- `build_item_index.py`：把 20,000 个商品分片编码；每 512 个商品保存一个原子分片，重跑时跳过已验证分片。
- `evaluate_retrieval.py`：固定抽取 100 个视频，并在完整 20,000 候选池上比较纯文本、纯视觉和多模态输入；每个视频结果单独原子保存，可以断点续跑。

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval

CUDA_VISIBLE_DEVICES=0 python scripts/build_item_index.py \
  --batch-size 16 \
  --shard-size 512

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_retrieval.py \
  --num-queries 100 \
  --modes text vision multimodal
```

商品索引默认写入 `artifacts/zero_shot_v0/items/`，试运行结果默认写入 `artifacts/zero_shot_v0/eval100/`。最终汇总位于 `eval100/summary.json`，逐视频 Top-100 和精确排名位于 `eval100/results/`。

文本输入按字段独立限额，避免长 OCR 把 ASR 完全截掉。奇数帧视频会复制最后一帧作为时间维填充；所有帧保持宽高比并填充为 `512×512`。任何视觉预处理异常都会中止，不会接受官方实现的静默 `NULL` embedding。

## 训练前诊断

`analyze_rerank_errors.py` 对已经完成的精排结果做标签侧错误分解，并可成对比较两个嵌套候选实验。`run_error_analysis_top25.sh` 按“Top-10 错误分解 → Top-25 全新零训练精排 → 成对对照”的固定顺序运行。详情见 [RERANK.md](RERANK.md)。

## 训练数据准备

- `locate_training_data.py`：在本地数据目录中查找已解压且同时包含商品/视频元数据的根目录，不进入巨型图片树。
- `audit_training_data.py`：流式检查训练 JSONL、标签引用、重复/冲突、文本覆盖和图片路径；默认随机检查各 10,000 条路径，可用 `--full-path-check` 全量检查且不会把全部路径留在内存。
- `prepare_training_pairs.py`：在不修改原始文件的前提下去除完全重复标签行，并按显式 item_id 排除无法训练的样本，保存清洗清单与哈希。
- `split_training_data.py`：按 item_id 做稳定哈希划分，只输出 ID 清单，不复制媒体。该版本确保商品 ID 不重叠，但尚未合并近重复图片组，因此不能直接视为最终开发集。
- `hash_item_images.py`：多进程、分片、可断点续跑地计算商品图文件 SHA-256 和版本固定的 DCT 感知哈希。
- `propose_item_duplicate_groups.py`：用精确哈希和 Hamming 距离提出重复候选簇；输出明确标记为待核验，不会自动改变开发集。
- `mine_hard_negatives.py`：使用冻结训练商品索引，为训练视频生成纯视觉和多模态 Top-100 安全负例；强制排除正例及同一候选簇商品，按视频分片原子保存并可断点续跑。
- `run_mine_hard_negatives_8gpu.sh`：将训练视频结果分片分配到 8 张 GPU，全部 worker 完成后统一校验并发布 manifest。
- `prepare_reranker_sft.py`：从完整 hard-negative 候选中确定性抽取视频组和分层负例，物化与评估一致的 512×512 无损媒体，并输出 ms-swift Qwen3-VL-Reranker 数据格式。
- `run_reranker_sft_pilot.sh`：在独立 `ecom-train` 环境中对 100 组 Pilot 数据执行单卡 Qwen3-VL-Reranker-2B LoRA pointwise 前向/反向和评估链路验证。
- `run_reranker_listwise_pilot.sh`：在正式 4,500/500 query 数据上运行单卡 50-step LoRA listwise 目标函数 Pilot，用训练前后 MRR/NDCG 判断是否修复 pointwise 的负类塌缩。
- `run_reranker_listwise_perf_pilot.sh`：对正式 4,500 个训练 query 运行隔离评估和保存耗时的 1/2/4/8 卡 listwise 性能 Pilot，并采集逐卡 GPU 与训练进程 CPU 指标。
- `summarize_reranker_listwise_perf.py`：汇总四组性能 Pilot 的 query/s、等效 pair/s、资源指标和 8/1 加速比，按 4 倍吞吐及 60% 利用率门槛给出是否通过。
- `run_reranker_listwise_train_8gpu.sh`：按冻结的正式 v1 协议运行 8 卡 listwise LoRA 训练；使用 global batch 8、学习率 `1e-5` 和 500 step，在 step 0/100/200/300/400/500 评估并每 100 step 保存，同时记录 GPU/CPU 指标并保护已有 GPU 上下文。
- `run_item_disjoint_retrieval.sh`：为开发商品补建冻结索引，拼接训练/开发两个互斥商品索引，从开发视频中固定均匀抽样并生成纯视觉/多模态 Top-100，最后冻结两路 Top-10 并集。
- `export_reranker_checkpoints.sh`：按 ms-swift 官方 `swift export --adapters ... --merge_lora true` 接口，将 checkpoint-300/400/500 导出为独立可推理模型。
- `run_item_disjoint_rerank_4gpu.sh`：在相同 item-disjoint 候选上用 4 张 GPU 并行比较 Base、checkpoint-300/400/500，并先做每模型 100-query gate。
- `audit_reranker_grpo_rollouts.py`：不训练模型；把已保存的 reranker sigmoid 分数还原为 logit，在固定候选内模拟 categorical score-policy GRPO，报告不同温度/group size 下的奖励零方差率和有效 advantage 比例。
- `prepare_score_policy_grpo_pilot.py`：从训练侧 1 正 + 7 负 query group 中确定性选取 100 组，产生两个对照 arm 共用的 JSONL 和 hash manifest。
- `reranker_score_policy_grpo_plugin.py`：通过 ms-swift external plugin 注册 categorical score-policy/contextual-bandit GRPO loss，并按 batch 记录奖励与 advantage 统计。
- `run_reranker_score_policy_grpo_pilot.sh`：从合并后的 checkpoint-500 并行运行固定 100 query/100 step 的 score-policy 与等 step listwise 单卡对照。
- `summarize_score_policy_grpo_stats.py`：汇总实际训练 rollout 的 active-group rate、奖励、advantage、策略熵和有限 loss gate。
- `compare_reranker_predictions.py`：汇总固定候选上的 Top-1、MRR、NDCG、候选内准确率及相对 Base 的救回、伤害和净增正确数。
- `evaluate_dynamic_top25_router.py`：将 Top-25 同次打分限制回 Top-10 作为无漂移基线，使用固定 calibration/test 哈希拆分评估 10%/25%/50%/75%/100% 扩容预算。
- `evaluate_incremental_top25_router.py`：使用独立 Top-10 运行的路由分数和已有候选分数，只合并 Top-25 新增候选分数，离线验证缓存式两阶段排序的准确率语义。
- `rerank_dynamic_top25_incremental.py`：真实执行单次视频加载、Top-10 打分和低置信 query 新增候选打分，分别记录 Top-10、增量及端到端延迟；强制仅运行 100–500 query Pilot。

远端命令和继续训练的门槛见 [TRAINING.md](TRAINING.md)。
