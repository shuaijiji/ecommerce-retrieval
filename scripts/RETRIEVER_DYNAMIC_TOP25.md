# 新召回上的动态 Top-25（2026-11-23）

## 已确认结果与本轮目标

固定 checkpoint-500、Top-10候选对照已由用户在七卡运行完成。完整2,000-query原始/pilot100/train1k的Top1=.4660/.4845/.4900，Top3=.5755/.6000/.6090。pilot100相对原始Top1救回53/伤害16、净增37、p=9.095e-6；train1k相对原始救回142/伤害94、净增48、p=.00215。train1k对pilot100仅净增11条Top1、p=.4850，Top3净增18、p=.3296。因此继续保留pilot100作本轮主方案，train1k作对照，不宣称1k稳定胜出。

目标：固定新召回候选后，验证旧方案的low_top1_score + 25%预算能否保留Top-25精排收益。复用刚完成的七卡Top-10缓存，不训练、不重建索引、不运行公开10k或正式50k。

## 执行前固定协议

- 主方案pilot100，对照train1k；本轮不根据test结果重新选择主方案或更换预算、seed、特征。
- 固定原有sha256(20261119:photo_id)排序，前1,000 calibration、后1,000 test。该内部开发集已反复使用，不当作新的未见测试集。
- 固定low_top1_score与25%预算。每组阈值仅从其缓存Top-10 calibration首名分数计算；同分稳定哈希，test自然扩容比例允许偏离25%。不使用标签或Top-25分数拟合阈值，不要求calibration恰好250条同分抽中。
- **GPU运行前**把两组阈值、缓存候选/预测哈希、预算和seed写入experiment_config.json，结束后再次比对阈值一致。
- 使用同一checkpoint-500、batch size4等既有评分协议。新增两组完整Top-25并集打分，共4,000 query；Top-10分数不重跑。先每worker100条功能检查，再全分片，复用已完成结果。
- 缓存语义：旧Top-10商品始终保留旧分数，只从Top-25预测取新增商品分数；按合并分数重排。完整Top-25重跑仅作对照，同时报告旧商品两次分数漂移分布。
- 主指标test完整系统Top1相对缓存Top-10的净增和配对p（净增>0且p<.05为预设收益检查）；次指标Top3，同时报告候选Recall、MRR/NDCG、救回/伤害、实际扩容率、平均候选数，以及保留全量增量Top-25净收益比例。无全量正净收益时保留比例为null。
- test结果不用于事后换策略、调预算或提高/降低门槛。离线通过后再准备100-query真实新增候选推理与延迟Pilot；本轮不能声称端到端加速。

## 更新脚本

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
python - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
src=Path('/code/ecommerce-retrieval-source/scripts')
dst=Path('/code/ecommerce-retrieval/scripts')
backup=dst.parent/'backups'/datetime.now().strftime('scripts-dynamic-%Y%m%d-%H%M%S-%f')
shutil.copytree(dst,backup)
shutil.copytree(src,dst,dirs_exist_ok=True)
print('同步完成，备份：',backup)
PY
```

## 七卡运行

选用1–7须当前仍空闲。检查失败时停止，不抢占其他任务。卡数可设1–8，不在同一输出目录改变卡数。缓存Top-10位置明确使用用户刚完成的七卡目录。

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval
mkdir -p artifacts/train_v0/retriever_rerank_logs
nohup python -u scripts/run_retriever_rerank_comparison.py \
  --root /code/ecommerce-retrieval \
  --gpus 1,2,3,4,5,6,7 \
  --top-k 25 --arms pilot100 train1k \
  --top10-root /code/ecommerce-retrieval/artifacts/train_v0/retriever_rerank_top10_7gpu_v1 \
  --output-dir /code/ecommerce-retrieval/artifacts/train_v0/retriever_dynamic_top25_7gpu_v1 \
  > artifacts/train_v0/retriever_rerank_logs/dynamic_top25_7gpu.log 2>&1 &
echo "PID=$!"
```

如只想先完成候选及阈值登记，添加 `--prepare-only`；用相同参数移除该flag后可继续启动GPU，不重新拟合成不同配置。

## 进度与结果

```bash
tail -n 20 artifacts/train_v0/retriever_rerank_logs/dynamic_top25_7gpu.log
tail -n 5 artifacts/train_v0/retriever_dynamic_top25_7gpu_v1/logs/pilot100_worker_0_full.log
```

首次生成候选/校验哈希需要时间；某stage尚未开始时对应日志不存在。完成后：

```bash
cat artifacts/train_v0/retriever_dynamic_top25_7gpu_v1/comparison.json
```

重点读取 `arms.pilot100` 中 `test_top10`、`test_dynamic_incremental`、`test_all_incremental_top25`、`test_top1_benefit_gate`。`router.threshold` 是执行前登记的可独立逐query使用的数值规则。train1k按相同方式报告，保持对照地位。

若两组分析因模型/预处理/标签/候选来源不一致失败，保留所有已打分结果，先检查，不绕过校验。脚本与测试已完成本地验证；这轮真实GPU结果仍待用户运行。
