# 精排训练执行顺序

训练归档：`viewfs://hadoop-lt-cluster/home/mmu/wulei/data/competition/all_train_data.zip`。验证集只用于最终评估，禁止用于训练、hard-negative 挖掘或阈值选择。

## 1. 定位与审计

先同步最新脚本，然后定位可能已解压的数据：

```bash
python scripts/locate_training_data.py \
  --search-root /code/ecommerce-retrieval/data \
  | tee artifacts/train_data_locations.json
```

从输出中选择训练集根目录。它必须同时包含 `item_info.jsonl`、`photo_info.jsonl` 和 `label.jsonl`（训练集）或 `label.json`（验证集），且不能误选验证集的 `test_small_package`。先运行随机路径审计：

```bash
TRAIN_ROOT=/替换为实际训练集根目录
python scripts/audit_training_data.py \
  --data-root "$TRAIN_ROOT" \
  --allow-missing-item-images 1 \
  --output artifacts/train_v0/audit_sample.json
```

元数据和抽样路径全部通过后再执行全路径检查：

```bash
python scripts/audit_training_data.py \
  --data-root "$TRAIN_ROOT" \
  --full-path-check \
  --allow-missing-item-images 1 \
  --output artifacts/train_v0/audit_full.json
```

脚本发现任何关键错误会在保存 JSON 报告后以非零状态退出。不要在错误未解释前继续训练。

## 2. 初始 item-disjoint 划分

```bash
python scripts/prepare_training_pairs.py \
  --labels "$TRAIN_ROOT/label.jsonl" \
  --exclude-item-id 20304330443907 \
  --output-labels artifacts/train_v0/clean_labels.jsonl \
  --output-manifest artifacts/train_v0/clean_labels_manifest.json

python scripts/split_training_data.py \
  --labels artifacts/train_v0/clean_labels.jsonl \
  --dev-item-ratio 0.02 \
  --seed 20261110 \
  --output-dir artifacts/train_v0/split_item_disjoint
```

此步骤不会复制图片。它保证 train/dev 的 item_id 不重叠，但尚未处理不同 item_id 的近重复图片，因此当前只作为初始划分。近重复分组完成前不得把它作为最终开发集，也不得据此选择最终模型。

## 3. 商品图片重复候选

先在独立目录做 1,000 张功能预跑：

```bash
conda activate ecom-infer
python scripts/hash_item_images.py \
  --data-root "$TRAIN_ROOT" \
  --exclude-item-id 20304330443907 \
  --workers 16 \
  --limit 1000 \
  --output-dir artifacts/train_v0/item_hashes_pilot
```

预跑 `error_count=0` 后，去掉 `--limit` 并使用全新输出目录 `artifacts/train_v0/item_hashes_full`。脚本按 10,000 商品原子分片，可断点续跑；每张图保存文件 SHA-256、尺寸及版本固定的 63-bit DCT 感知哈希。

完成后只生成候选簇，不直接修改划分：

```bash
python scripts/propose_item_duplicate_groups.py \
  --hash-dir artifacts/train_v0/item_hashes_full \
  --hamming-threshold 3 \
  --output-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --output-summary artifacts/train_v0/proposed_item_duplicate_summary.json
```

感知哈希候选簇必须先做大小分布检查和抽样视觉核验。未经核验时状态为 `proposed_not_confirmed`，不得用于最终划分。

全量候选生成后，自动跳过字节级完全相同的组，对“文件不同但感知哈希相似”的候选生成确定性抽查拼图：

```bash
python scripts/sample_item_duplicate_groups.py \
  --data-root "$TRAIN_ROOT" \
  --groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --output-dir artifacts/train_v0/duplicate_visual_review \
  --sample-size 100 \
  --largest-count 20
```

抽样由 20 个最大感知候选簇和 80 个固定 seed 随机簇组成。查看输出目录中的 `review_sheet_*.jpg`；审核结论仅允许记为 `same_or_near_duplicate`、`different` 或 `uncertain`。

若视觉核验不可行，采用保守隔离策略：不删除任何原始数据，把全部候选簇商品固定留在训练侧，开发集只从 singleton 商品中确定性抽取；训练 hard negatives 时禁止同一候选簇成员互为负例：

```bash
python scripts/split_training_data.py \
  --labels artifacts/train_v0/clean_labels.jsonl \
  --dev-item-ratio 0.02 \
  --seed 20261110 \
  --quarantine-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --output-dir artifacts/train_v0/split_quarantine_duplicates
```

该策略不声称感知候选已被人工确认，而是通过“不进入 dev”消除已发现候选造成的跨集合泄漏。输出额外包含 `quarantined_item_ids.txt` 和 `quarantined_photo_ids.txt`。

## 后续门槛

只有 `audit_full.json` 的关键错误全部为 0，且近重复分组后的最终划分固定，才继续：冻结召回器生成 hard negatives → 1～5 万条 SFT 冒烟 → 固定 Top-10 评估 → 扩大训练 → 动态 Top-25。GRPO 不在第一阶段范围内。

## 4. 冻结召回器的训练商品索引

最终划分固定后，只对 `train_item_ids.txt` 中的商品建索引。这一步是冻结模型的离线推理，不更新任何参数。8 张 GPU 按索引分片互不重叠地并行处理，中断后重跑会校验并跳过已完成分片：

```bash
TRAIN_ROOT=/code/ecommerce-retrieval/data/train/all_data_package
nohup bash scripts/run_train_item_index_8gpu.sh \
  "$TRAIN_ROOT" \
  artifacts/train_v0/split_quarantine_duplicates/train_item_ids.txt \
  artifacts/train_v0/train_item_index \
  > artifacts/train_v0/train_item_index_launcher.log 2>&1 &
echo $! | tee artifacts/train_v0/train_item_index_launcher.pid
```

查看整体调度和单卡进度：

```bash
tail -f artifacts/train_v0/train_item_index_launcher.log
tail -f artifacts/train_v0/train_item_index/logs/worker_0.log
```

完成标志是 `artifacts/train_v0/train_item_index/manifest.json` 存在且 `complete=true`。后续 hard-negative 脚本必须同时使用 `train_photo_ids.txt`、该商品索引和 `proposed_item_duplicate_groups.jsonl`：正例及同候选簇商品都不得输出为负例。

## 5. 冻结召回器生成 hard-negative 候选

本阶段不直接生成 SFT 对话，而是为每个训练视频保存两路可审计的 Top-100 安全负例：纯视觉召回与多模态召回。脚本在完整训练商品池上打分，并强制屏蔽正例及正例所在的全部感知候选簇成员；官方验证集不参与候选生成。

先在单卡上运行 1,000 个视频的 Pilot。Pilot 仍使用完整 764,049 商品索引，只限制查询视频数量：

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval

CUDA_VISIBLE_DEVICES=0 python -u scripts/mine_hard_negatives.py \
  --data-root /code/ecommerce-retrieval/data/train/all_data_package \
  --labels artifacts/train_v0/clean_labels.jsonl \
  --photo-ids artifacts/train_v0/split_quarantine_duplicates/train_photo_ids.txt \
  --item-index artifacts/train_v0/train_item_index \
  --duplicate-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --output-dir artifacts/train_v0/hard_negative_candidates_pilot \
  --limit 1000 \
  --shard-size 1000
```

Pilot 完成后必须看到 `hard_negative_candidates_pilot/manifest.json` 的 `complete=true`、恰好 1 个结果分片和 1,000 行结果。结果分片会在恢复时重新校验 photo_id 顺序、两路 Top-K 长度、候选唯一性，以及正例/同候选簇成员均未混入负例。

2026-11-11 实际 Pilot 已通过：完整商品池上的 1,000 个视频写入 `part-000000-001000.jsonl`，用时 93.5 秒，随后回读 `partition 0/1 complete: 1000 photos` 和 `hard-negative candidates complete`。

Pilot 通过后使用全新输出目录启动 8 卡全量任务：

```bash
nohup bash scripts/run_mine_hard_negatives_8gpu.sh \
  /code/ecommerce-retrieval/data/train/all_data_package \
  artifacts/train_v0/split_quarantine_duplicates \
  artifacts/train_v0/train_item_index \
  artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  artifacts/train_v0/hard_negative_candidates \
  > artifacts/train_v0/hard_negative_candidates_launcher.log 2>&1 &

echo $! | tee artifacts/train_v0/hard_negative_candidates_launcher.pid
```

每个 worker 只加载自己负责的视频元数据，但会把完整商品矩阵加载到对应 GPU，以保持与冻结召回协议一致的精确全池检索。输出按 1,000 个视频形成原子 JSONL 分片，8 个 worker 按分片编号互不重叠，可安全断点续跑。监控：

```bash
tail -f artifacts/train_v0/hard_negative_candidates/logs/worker_0.log
find artifacts/train_v0/hard_negative_candidates/shards \
  -name 'part-*.jsonl' | wc -l
```

最终预计 983 个分片。只有 launcher 日志出现 `hard-negative mining complete`，且 `hard_negative_candidates/manifest.json` 中 `complete=true`、分片数为 983，才视为完成。改变模型、索引、清洗标签、划分、重复簇、Top-K 或分片大小时必须使用新的输出目录。

这些 Top-100 是带来源分数的候选池，不是最终 SFT 样本。下一步需从两路高排名错误、不同排名区间及少量随机负例中确定性采样，构造 1～5 万条 SFT 冒烟集，并保存候选来源和版本哈希。

## 后续全量任务的性能门槛

单卡功能 Pilot 通过不等于可以直接启动 8 卡全量。后续索引、候选生成、SFT 和评估任务在全量前必须增加代表性 1/2/4/8 卡性能 Pilot：至少连续运行到文件缓存和初始化影响消失，记录总吞吐、逐卡 GPU SM 利用率/显存/功耗、每进程 CPU 占用和落盘速度。

- 稳态 GPU SM 利用率目标为 60% 以上；长期低利用率时先检查 CPU 图像预处理、线程过量、存储读取、同步点和 batch size。
- 8 卡吞吐低于单卡 4 倍时停止扩容，不按单卡小样本结果线性估算全量耗时。
- 优先验证批量查询和异步数据预取，并限制每个 worker 的 CPU 线程，避免多个进程各自占用十多个核导致争用。
- 改变 batch size、候选打包或数值协议可能造成 bf16 分数漂移，必须使用新输出目录，不能混用旧分片。
- 在性能门槛通过前，不启动百万样本级全量任务。

## 6. 构造精排 SFT 冒烟数据

完整 hard-negative 候选必须先满足：launcher 日志出现 `hard-negative mining complete`，manifest 为 `complete=true` 且列出 983 个分片。数据构造采用 5,000 个确定性训练视频组，每组 1 个正例和 7 个负例，展开后为 40,000 个 pointwise yes/no pair，位于第一阶段 1～5 万 pair 的范围内。

每组负例固定覆盖：纯视觉/多模态 Top-10、两路 11～50、两路 51～100，以及一个从训练商品池稳定哈希抽取的随机商品。若两路候选重合，先去重，再按联合原始排名补齐。正例和正例所在感知候选簇仍然禁止作为负例。

先运行 100 视频数据 Pilot。该步骤只做 CPU 图片处理，不占 GPU：

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval

python scripts/prepare_reranker_sft.py \
  --data-root /code/ecommerce-retrieval/data/train/all_data_package \
  --photo-ids artifacts/train_v0/split_quarantine_duplicates/train_photo_ids.txt \
  --train-item-ids artifacts/train_v0/split_quarantine_duplicates/train_item_ids.txt \
  --candidates artifacts/train_v0/hard_negative_candidates \
  --duplicate-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --output-dir artifacts/train_v0/reranker_sft_smoke_pilot \
  --num-queries 100 \
  --eval-queries 10 \
  --negatives-per-query 7 \
  --media-workers 16
```

Pilot 通过标准：`manifest.json` 中 `complete=true`、`effective_pointwise_pairs=800`，`swift_train.jsonl` 为 90 行、`swift_smoke_eval.jsonl` 为 10 行；每行包含 1 个 `<video>` 查询、1 个正例商品和 7 个负例商品，所有媒体路径存在。训练视频帧和商品图均先按既有灰底 letterbox 规则转为 `512×512` 无损 PNG，奇数视频帧通过重复最后一个路径补齐。

Pilot 通过后使用新目录构造正式 40,000 pair 冒烟集：

```bash
python scripts/prepare_reranker_sft.py \
  --data-root /code/ecommerce-retrieval/data/train/all_data_package \
  --photo-ids artifacts/train_v0/split_quarantine_duplicates/train_photo_ids.txt \
  --train-item-ids artifacts/train_v0/split_quarantine_duplicates/train_item_ids.txt \
  --candidates artifacts/train_v0/hard_negative_candidates \
  --duplicate-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --output-dir artifacts/train_v0/reranker_sft_smoke_40k \
  --num-queries 5000 \
  --eval-queries 500 \
  --negatives-per-query 7 \
  --media-workers 32
```

输出 `samples.jsonl` 保存 photo_id、正例及每个负例的来源路、分层区间、原始排名和分数；`swift_train.jsonl`/`swift_smoke_eval.jsonl` 使用 ms-swift 多模态 reranker 的 `messages`、`videos`、`positive_messages/positive_images`、`negative_messages/negative_images` 格式。500 条 query holdout 只用于监控训练管线和损失，不是 item-disjoint 的最终选模开发集。

训练框架采用 ms-swift 的 Qwen3-VL-Reranker pointwise loss。官方格式和示例：

- https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/Reranker.md
- https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/qwen3/qwen3_vl_reranker.sh

安装和训练必须使用新的 `ecom-train` 环境，不升级或破坏已验证的 `ecom-infer`。在真正启动 8 卡训练前，先对 100 组数据做单卡前向/反向、显存和 token 长度 Pilot，再按本文件的 1/2/4/8 卡性能门槛确定 batch、CPU worker 和并行配置。
