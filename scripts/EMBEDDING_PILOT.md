# 召回 embedding 对比学习 Pilot

2026-11-19 正式转入召回模型训练。固定 item-disjoint 2,000 query 的两路 Top-100 并集 Recall 基线为 83.3%（1,666/2,000），334 条未召回。官方 10k 保持封存，动态 Top-25 与精排 checkpoint-500 暂不改变。

**当前交付是代码，尚未执行真实数据审计和 GPU 训练。** 本地只有代码与记录，没有训练数据、模型或 PyTorch。远端由用户从 GitHub 拉取执行；下面各阶段顺序运行，失败就停止并保留日志，不能把功能 Pilot 当作泛化收益。

## 1. 同步并全量审计负例

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
mkdir -p backups
cp -a scripts "backups/scripts-$(date +%Y%m%d-%H%M%S)"
cp -a /code/ecommerce-retrieval-source/scripts/. scripts/
conda activate ecom-infer
mkdir -p artifacts/train_v0/embedding_logs
python -u scripts/prepare_embedding_pilot.py \
  --data-root data/train/all_data_package \
  --split-dir artifacts/train_v0/split_quarantine_duplicates \
  --labels artifacts/train_v0/clean_labels.jsonl \
  --candidates artifacts/train_v0/hard_negative_candidates \
  --duplicate-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
  --num-queries 100 --negatives 4 --seed 20261119 \
  --output-dir artifacts/train_v0/embedding_data_pilot_v1 \
  > artifacts/train_v0/embedding_logs/data_audit.log 2>&1
```

完整扫描 manifest 声明的全部分片，逐 query 核对真实标签、训练商品归属、正例/同重复簇排除、两路候选唯一性、排名、有限分数、查询顺序和完整覆盖。检查 train/dev 不相交及重复簇隔离；核对挖掘时标签/划分/重复簇哈希。任何异常拒绝发布 manifest。扫描及分片哈希会产生较多磁盘读取，但不占 GPU。

随后按固定 seed 从训练侧选 100 query（参数允许 100–500），从两路按排名轮流取 4 个去重 hard negatives；不取随机负例、不引入跨 query 的 in-batch negatives。输出 `samples.jsonl` 保留 query 原始帧/文本、商品图/完整标题和负例来源/原始排名/分数；逐张解码所需媒体。`manifest.json` 记录全部来源与分片哈希。

排除范围是 **已知 item ID 与已发现感知哈希重复候选簇**。这些簇仍未被确认为语义同款，不能证明全部不同主图的同款都被覆盖；训练前应核验本次 `samples.jsonl` 中正负商品，疑似同款应补进重复组并重新挖掘/审计，不能当成正确负例继续训练。不得把代码审计通过表述成“全库没有假负例”。

## 2. 单卡训练功能 Pilot

在现有 `ecom-train-v2` 环境检查 `torch/transformers/peft/qwen_vl_utils` 可导入，不修改已验证的 `ecom-infer` 环境。上游源码必须是 `/code/ecommerce-retrieval/Qwen3-VL-Embedding`。选择一张空闲物理卡；程序拒绝已有超过 1 GiB 占用的卡。

```bash
conda activate ecom-train-v2
cd /code/ecommerce-retrieval
python -c 'import torch, transformers, peft, qwen_vl_utils; print(torch.__version__, transformers.__version__, peft.__version__)'
CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader) \
python -u scripts/train_embedding_pilot.py \
  --repo Qwen3-VL-Embedding --model models/Qwen3-VL-Embedding-2B \
  --dataset artifacts/train_v0/embedding_data_pilot_v1 \
  --output-dir artifacts/train_v0/embedding_pilot_v1 \
  --steps 100 --lr 1e-5 --temperature 0.07 --seed 20261119 \
  > artifacts/train_v0/embedding_logs/train.log 2>&1
```

固定协议：共享双塔权重，语言层 q/v attention LoRA rank=8、alpha=16，视觉 backbone 冻结，单 query + 1 正例 + 4 显式负例，归一化向量点积、temperature=0.07 的 InfoNCE。查询为视频帧和文本；商品侧遵循现有索引的主图/完整标题输入协议。沿用 512 图像配置、视频 letterbox/奇数帧补齐、4096 max length。商品图沿用现有索引入口的原图协议，未擅自改成精排的 PNG 预处理。

直接调用底层模型而非带 `no_grad` 的官方包装器 forward。检查视觉张量及 grid 数量，拒绝 NULL 降级；每步检查有限 loss、有限非零梯度。输出逐步 loss/margin/梯度范数，训练前后在同一批全部 query 上用 eval 模式测量平均 loss、正例减最难负例 cosine margin，记录峰值 allocated/reserved 显存。两项必须严格改善才通过 `functional_gate` 并导出 `merged/`；否则保留 adapter 和日志，停止。

这里只验证训练链路，不把训练集 margin 改善当成泛化。100 step 是事先固定配置，不按开发集选择 step/seed/温度。

## 3. 固定 2,000 query 全池召回

只在功能门槛通过后执行。**这一阶段重建约 78 万商品的全部 train/dev 索引，成本远高于训练 Pilot。** 默认单卡、复用已验证建库实现；未验证多卡性能前不自动启动八卡扩容。

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval
EVAL_GPU=1 bash scripts/run_embedding_pilot_eval.sh \
  /code/ecommerce-retrieval \
  /code/ecommerce-retrieval/artifacts/train_v0/embedding_pilot_v1 \
  /code/ecommerce-retrieval/artifacts/train_v0/embedding_pilot_eval_v1
```

两套商品索引都使用 `merged/`，保存真实 safetensors 权重哈希，查询端强制检查哈希一致；不能只比较 `config.json`（不同 checkpoint 可能配置相同）。固定使用旧 `item_disjoint_eval_v1/retrieval/selected_photo_ids.json`，不重新抽样、不复用旧商品向量。

`comparison.json` 报告视觉、多模态、两路并集 Recall@10/25/50/100，以及配对救回/丢失数。扩容门槛预注册为：Top-100 并集 Recall > 83.3%，且相对基线的双侧精确配对检验 p < 0.05。若仅数值超过而未显著，不扩大训练、不重新接入动态 Top-25。主门槛通过后才规划扩大训练及性能 Pilot；官方 10k 不在本流程中。

## 本地验证

`python3 -m unittest discover -s scripts -p test_embedding_pilot.py -v`；新脚本及修改脚本语法检查。PyTorch/PEFT 前反向、媒体预处理、显存、实际索引构建与召回仍待远端验证。
