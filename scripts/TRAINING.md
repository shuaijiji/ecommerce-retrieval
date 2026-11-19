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

## 7. 单卡精排训练 Pilot

正式 40,000 pair 数据发布前必须回读 `manifest.json`：`complete=true`、`effective_pointwise_pairs=40000`，且 train/eval query 数分别为 4,500/500。已确认后先创建独立环境：

```bash
conda create -n ecom-train python=3.11 -y
conda activate ecom-train

python -m pip install --upgrade pip
python -m pip install "transformers>=4.57" "qwen_vl_utils>=0.0.14" "ms-swift>=4.0" packaging ninja
python -m pip install flash-attn --no-build-isolation
```

安装后先记录实际版本，不依赖未锁定的“最新版”复现：

```bash
python -c 'import torch,transformers,swift,qwen_vl_utils; print(torch.__version__, torch.version.cuda); print(transformers.__version__); print(swift.__version__)'
```

先运行单卡 10 step 功能 Pilot，使用 100 组、800 pair 数据，不直接读取正式 40k 目录：

```bash
cd /code/ecommerce-retrieval
bash scripts/run_reranker_sft_pilot.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_pilot \
  artifacts/train_v0/reranker_sft_lora_pilot_v1 \
  2>&1 | tee artifacts/train_v0/reranker_sft_lora_pilot_v1.launch.log
```

脚本固定 `MAX_POSITIVE_SAMPLES=1`、`MAX_NEGATIVE_SAMPLES=7`。ms-swift 会把每个 query 展开为 8 个 pointwise pair，因此 `per_device_train_batch_size=1` 时实际每步是 8 对，不得误解为 1 对。Pilot 采用 LoRA rank 8/alpha 32、bf16、FlashAttention、padding-free 和 4,096 token 上限，与官方 reranker 脚手架保持一致。

通过标准：模型、数据和媒体全部成功加载；训练完成 10 step；初始和 step 5/10 评估成功；`loss`、`eval_loss`、梯度范数均为有限数；无 OOM/NaN/媒体解码或 token 截断导致的数据大量丢弃；并记录 `nvidia-smi` 的峰值显存、SM 利用率和 10 step 总耗时。功能 Pilot 通过后才设计更长的 1/2/4/8 卡性能 Pilot，不直接启动正式训练。

## 8. Listwise 排序与性能 Pilot

pointwise 10-step Pilot 的 MRR/NDCG 从 0.6833/0.7655 退化到 0.6333/0.7286，因此不扩容该配置。正式 4,500/500 query 上的 listwise 50-step Pilot 已通过排序方向门槛：step 0/25/50 MRR 为 0.78295714/0.78590238/0.79066905，NDCG 为 0.83696853/0.83918828/0.84271104。500 个 query 仅用于训练管线监控，不能替代 item-disjoint 开发集。

性能 Pilot 使用正式 4,500 个训练 query，但关闭验证和 checkpoint 保存，避免把每轮约 130 秒的 500-query 评估计入训练吞吐。固定每卡 batch size 1、gradient accumulation 1、200 step；因此 N 卡每个 optimizer step 处理 N 个 query，等效 pointwise pair/s 固定按 query/s × 8 折算。1/2/4/8 卡必须分别使用全新输出目录：

```bash
conda activate ecom-train-v2
cd /code/ecommerce-retrieval

bash scripts/run_reranker_listwise_perf_pilot.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_listwise_perf_1gpu \
  1

bash scripts/run_reranker_listwise_perf_pilot.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_listwise_perf_2gpu \
  2

bash scripts/run_reranker_listwise_perf_pilot.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_listwise_perf_4gpu \
  4

bash scripts/run_reranker_listwise_perf_pilot.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_listwise_perf_8gpu \
  8
```

脚本通过 `NPROC_PER_NODE` 启动 ms-swift DDP，每次保存 `train.log`、`run_info.txt`、`gpu_metrics.csv` 和 `cpu_metrics.csv`。运行期间不应并发其他 GPU 任务。四组均成功后汇总：

```bash
python scripts/summarize_reranker_listwise_perf.py \
  artifacts/train_v0/reranker_listwise_perf_1gpu \
  artifacts/train_v0/reranker_listwise_perf_2gpu \
  artifacts/train_v0/reranker_listwise_perf_4gpu \
  artifacts/train_v0/reranker_listwise_perf_8gpu \
  --output artifacts/train_v0/reranker_listwise_perf_summary.json
```

汇总器以 `N × max_steps × per-device batch / train_runtime` 计算 query/s，以 query/s × 8 计算等效 pair/s；资源监控只截取进程结束前与 `train_runtime` 等长的纯训练窗口，排除模型加载和数据初始化，再报告逐卡 GPU 利用率中位数、峰值显存、功耗及训练进程 CPU/RSS。正式训练门槛为：8 卡 query/s 至少达到单卡 4 倍，且 8 张卡的利用率中位数均不低于 60%。任一条件不满足时先分析 DDP 通信、CPU 图片处理、数据读取和 worker 争用，不启动正式训练。

若 batch size 1 的吞吐扩展通过但 GPU 利用率不达标，先在无外部占用的单张 GPU 上扫描每卡 batch size 1/2/4。脚本支持通过 `PILOT_GPU_LIST` 指定物理卡，通过 `PER_DEVICE_TRAIN_BATCH_SIZE`、`DATALOADER_NUM_WORKERS` 和 `DATASET_NUM_PROC` 覆盖默认值。batch 扫描必须固定 GPU、步数、数据和其他参数，按 query/s、利用率及峰值显存共同选择配置；出现 OOM 时保留失败目录，不覆盖或删除已有产物。

`PILOT_GPU_LIST` 使用 `nvidia-smi` 展示的物理索引。脚本会把索引解析为 GPU UUID 后设置 `CUDA_VISIBLE_DEVICES`，避免 CUDA 枚举顺序与 `nvidia-smi` 索引不一致；运行结束还会检查每张选中卡是否实际出现超过 1 GiB 显存和非零计算利用率，未命中目标 GPU 时不得发布为成功结果。

### 2026-11-17 实测结论

每卡 batch size 1 的 1/2/4/8 卡吞吐为 1.413/2.494/4.629/8.741 query/s，8 卡相对单卡为 6.186×；吞吐扩展门槛通过。严格自动 gate 因逐卡利用率中位数未达到 60% 而仍为 false，且早期单卡 batch 扫描存在 CUDA ordinal 与物理索引监控不一致，不能使用对应资源行。batch 1/2/4 的相对吞吐约为 1.653/1.946/1.991 query/s，batch 4 已明显进入收益平台。

当前停止追加性能 Pilot。后续若启动正式 listwise 训练，优先使用已完成多卡扩展验证的每卡 batch 1，不根据无效的单卡物理资源行改变配置。拟定正式入口为 8 卡、global batch 8、学习率 `1e-5`、最多 500 step，step 0/100/200/300/400/500 评估并每 100 step 保存；本段仅记录拟采用方案，不表示任务已启动或性能总 gate 已无条件通过。

## 9. 正式 8 卡 Listwise 训练

正式 v1 入口冻结为 8 卡、每卡 batch size 1、global batch 8、学习率 `1e-5` 和 500 step。启动前脚本会回读 40k manifest、把物理 GPU 索引转换为 UUID，并检查每张目标卡已有显存占用。它不会终止或覆盖未知 GPU 进程。

先前 GPU 0 曾存在约 15.4 GiB 的容器外/不可见计算上下文，因此首次运行预检查预计可能主动停止。先通过平台确认这些上下文的所有者及共享是否安全；只有明确允许共享时才设置 `ALLOW_BUSY_GPUS=1`，不能仅为了绕过检查而设置。

```bash
conda activate ecom-train-v2
cd /code/ecommerce-retrieval

bash scripts/run_reranker_listwise_train_8gpu.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_listwise_train_v1
```

正常启动后查看训练和资源日志：

```bash
tail -f artifacts/train_v0/reranker_listwise_train_v1/train.log
tail -f artifacts/train_v0/reranker_listwise_train_v1/gpu_metrics.csv
```

脚本在 step 0/100/200/300/400/500 评估，在 step 100/200/300/400/500 保存，并保留这些 checkpoint。500 个 eval query 仍只是训练侧管线监控集；不得只凭该集合选择最终 checkpoint 或宣称泛化提升。训练完成后必须另行在 item-disjoint 内部开发集和冻结 Top-10 协议上评估各 checkpoint。

若训练中断，指定原输出目录和实际 checkpoint 路径恢复；checkpoint 通常位于 `model/v0-*/checkpoint-N`：

```bash
RESUME_FROM_CHECKPOINT=artifacts/train_v0/reranker_listwise_train_v1/model/v0-时间戳/checkpoint-200 \
bash scripts/run_reranker_listwise_train_8gpu.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_listwise_train_v1
```

## 10. Item-disjoint Checkpoint 选择

正式 500-step 训练已完成。训练侧 500-query 监控集在 step 0/100/200/300/400/500 的 MRR 为 `0.780943/0.786692/0.794919/0.800182/0.807254/0.803947`，NDCG 为 `0.835454/0.839789/0.846066/0.849902/0.855148/0.852769`。checkpoint-400 在该监控集上领先，但该集合不是 item-disjoint，不能据此最终选模。

评估使用 `split_quarantine_duplicates` 的开发视频和开发商品。现有训练商品索引不包含开发正例，因此先为开发商品建立独立索引，再在检索时拼接训练/开发两个互斥索引；候选生成不读取标签补真值。先运行 100-query 全链路 Pilot：

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval

NUM_QUERIES=100 EVAL_GPU=1 \
bash scripts/run_item_disjoint_retrieval.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/item_disjoint_eval_pilot
```

合并导出 checkpoint-300/400/500。脚本使用物理 GPU 1 并转换为 UUID；每个目标目录必须同时包含 `config.json` 和 safetensors 才会视为完成：

```bash
conda activate ecom-train-v2

EXPORT_GPU=1 bash scripts/export_reranker_checkpoints.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_listwise_train_v1 \
  artifacts/train_v0/reranker_listwise_train_v1_merged
```

在固定 Pilot 候选上并行比较 Base 与三个 checkpoint。默认使用物理 GPU 1/2/3/4，避开有未知上下文的 GPU 0：

```bash
RERANK_GPU_LIST=1,2,3,4 \
bash scripts/run_item_disjoint_rerank_4gpu.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/item_disjoint_eval_pilot/candidates_top10.jsonl \
  artifacts/train_v0/clean_labels.jsonl \
  artifacts/train_v0/reranker_listwise_train_v1_merged \
  artifacts/train_v0/item_disjoint_eval_pilot/rerank
```

Pilot 的四个模型均通过媒体、分数、完整候选和结果行数 gate 后，使用全新目录固定 2,000 条均匀抽样开发 Query：

```bash
conda activate ecom-infer

NUM_QUERIES=2000 EVAL_GPU=1 \
bash scripts/run_item_disjoint_retrieval.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/item_disjoint_eval_v1

conda activate ecom-train-v2

RERANK_GPU_LIST=1,2,3,4 \
bash scripts/run_item_disjoint_rerank_4gpu.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/item_disjoint_eval_v1/candidates_top10.jsonl \
  artifacts/train_v0/clean_labels.jsonl \
  artifacts/train_v0/reranker_listwise_train_v1_merged \
  artifacts/train_v0/item_disjoint_eval_v1/rerank
```

最终比较结果为 `rerank/comparison.json`。选择规则以完整系统 Top-1 为主，同时检查 MRR/NDCG、相对 Base 的净增正确数和伤害数。只有 item-disjoint 结果确定唯一 checkpoint 后，才允许对该 checkpoint 运行一次官方 10,000 条冻结 Top-10 评估；禁止用官方验证集反复比较 checkpoint。

## 11. Listwise 最终结果

100-query Pilot 通过后，2,000-query item-disjoint 开发评估的 Base/checkpoint-300/400/500 Top-1 分别为 `0.4585/0.4655/0.4650/0.4660`。checkpoint-500 在预先规定的主指标上唯一领先，因此被固定为唯一官方最终评估对象。

官方冻结 Top-10 10,000 条结果：

- Base/checkpoint-500 完整系统 Top-1 为 `0.7936/0.8072`，绝对提升 `0.0136`；
- 候选内准确率为 `0.877973/0.893019`；
- MRR 为 `0.837659/0.846199`，NDCG 为 `0.854215/0.860652`；
- 配对救回/伤害/净增为 `282/146/+136`，双侧精确检验 `p≈4.69×10⁻¹¹`；
- 平均延迟为 `1.1658/1.2185` 秒/视频，checkpoint-500 增加约 4.52%；
- 候选 Recall 固定为 `0.9039`，剩余错误为 961 条候选未命中和 967 条候选内排序错误。

最终预测文件 SHA-256 为 `7708702d8a69427ce7c74fabda4ad03a380f0ff9a739e006504348e41c511ddf`，远程结果目录为 `artifacts/train_v0/reranker_listwise_official_10k_checkpoint500`。官方验证集从此封存，不得用于后续 GRPO 选参或 checkpoint 筛选。

## 12. GRPO 下一步

checkpoint-500 是后续 RL 的固定 SFT 起点。先审计 Qwen3-VL-Reranker-2B 的 yes/no 打分头、ms-swift rollout 接口和 DIVA-GRPO 公开 Qwen2.5-VL 生成式路径的差异，在此之前不直接运行公开 8 卡脚本。

普通 GRPO 的首个实现门槛为：只使用训练侧 hard negatives；真值 item_id 不变；输出必须严格可解析；正确性是主奖励，格式奖励不得掩盖选择错误；100-query Pilot 必须证明奖励有方差、advantage 非退化且无标签泄漏。普通 GRPO 通过固定 item-disjoint 开发门槛后，再实现 DIVA-GRPO 的难度变体、local/global advantage、难度加权和奖励范围缩放。

代码审计确认，当前 `generative_reranker` 在 ms-swift/vLLM 中映射为 `score/pooling` 任务，而标准 GRPO 需要 `generate` rollout 及 token log-prob，因此 checkpoint-500 不能直接套用标准 token-level GRPO/DIVA-GRPO 启动脚本。详细映射见 `GRPO_ADAPTATION.md`。

首先不训练，在已保存的 item-disjoint checkpoint-500 分数上审计 categorical score-policy 的奖励方差：

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval

PYTHONPATH=scripts python scripts/audit_reranker_grpo_rollouts.py \
  --candidates artifacts/train_v0/item_disjoint_eval_v1/candidates_top10.jsonl \
  --predictions artifacts/train_v0/item_disjoint_eval_v1/rerank/checkpoint_500/predictions_2000.jsonl \
  --labels artifacts/train_v0/clean_labels.jsonl \
  --group-sizes 4,8,16 \
  --temperatures 0.5,1.0,2.0 \
  --replicates 64 \
  --output artifacts/train_v0/grpo_rollout_audit_checkpoint500.json
```

该命令不加载 GPU 模型、不更新权重，也不读取官方 10k。主配置预先固定为 `temperature=1.0, group_size=8`；只有当候选命中 query 上 `expected_active_group_rate >= 0.50`，且模拟与解析 active rate 相差不超过 0.03，才继续实现 score-policy GRPO 最小训练 Pilot。`temperature=0.5/2.0` 仅作诊断，不用于事后更换主配置。训练 Pilot 必须同时保留等 step 的 listwise 继训对照。

审计已通过：主配置的解析/模拟 active-group rate 为 `0.636479/0.638287`，差值 `0.001809`，平均绝对 advantage 为 `0.503746`。运行最小配对训练 Pilot：

```bash
conda activate ecom-train-v2
cd /code/ecommerce-retrieval

POLICY_GPU=1 CONTROL_GPU=2 \
bash scripts/run_reranker_score_policy_grpo_pilot.sh \
  /code/ecommerce-retrieval \
  artifacts/train_v0/reranker_sft_smoke_40k \
  artifacts/train_v0/reranker_score_policy_grpo_pilot_v1
```

入口从 `reranker_listwise_train_v1_merged/checkpoint-500-merged` 出发，两个 arm 共用固定 100-query JSONL、LoRA rank 8、学习率 `1e-6` 和 100 step。默认并行占用两张空闲卡，已有显存超过 1 GiB 时拒绝启动。score-policy 的实际 rollout 汇总保存在 `score_policy/rollout_summary.json`；active-group rate 低于 0.50 或 loss 出现非有限值时脚本失败。训练成功只通过功能门槛，下一步需合并两个 adapter，只在固定 item-disjoint 开发集与原 checkpoint-500 做配对对比，不再读取官方 10k。

Pilot 已完成。score-policy 实际 active-group rate 为 `0.69`，loss 全部有限，通过功能门槛。2,000-query item-disjoint 上 checkpoint-500/继续 listwise/score-policy 的 Top-1 为 `0.4660/0.4665/0.4665`，MRR 为 `0.52469/0.52575/0.52498`，NDCG 为 `0.55161/0.55245/0.55181`。score-policy 未超过等 step listwise，因此不扩大该路径，不通过改温度、group size 或 seed 进行事后追试，也不据此启动 DIVA-GRPO。完整结果见 `进展更新-2026-11-19.md`。
