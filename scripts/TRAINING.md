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

## 后续门槛

只有 `audit_full.json` 的关键错误全部为 0，且近重复分组后的最终划分固定，才继续：冻结召回器生成 hard negatives → 1～5 万条 SFT 冒烟 → 固定 Top-10 评估 → 扩大训练 → 动态 Top-25。GRPO 不在第一阶段范围内。
