# 召回扩大实验 v1：固定 1,000 query

## 已取得的证据（用户远端回报，2026-11-20）

982,797 条训练候选审计完成，100 query × 4 负例的 100-step Pilot 成功。平均 loss 1.977394 → 1.718308，最难负例 margin -0.050624 → -0.025234，峰值 allocated/reserved 4.632/4.813 GiB，合并模型已导出。

固定 item-disjoint 2,000 query 上：

| 路由 | @10 基线→Pilot | @25 | @50 | @100 |
|---|---|---|---|---|
| 视觉 | .5660→.6020 | .6595→.6835 | .7175→.7390 | .7700→.7915 |
| 多模态 | .5460→.6075 | .6400→.7070 | .7095→.7695 | .7760→.8210 |
| 并集 | .6350→.6725 | .7285→.7595 | .7820→.8155 | .8330→.8595 |

并集 @100 救回/丢失 55/2，净增 53，配对 p=2.2954e-14。原始召回收益门槛通过。原有 `embedding_data_pilot_v1`、`embedding_pilot_v1`、`embedding_pilot_eval_v1` 保留，不覆盖。

## 本轮预注册

- 从原始 `models/Qwen3-VL-Embedding-2B` 重新训练；不从 100-query Pilot 续训。
- 按相同 seed=20261119 的稳定哈希选择 1,000 个训练 query（包含原 100-query 样本），4 个两路高排名显式负例；重新做完整安全审计、媒体解码与哈希记录。
- 单卡每步一个 query、1,000 step（一轮），LoRA rank=8/alpha=16、lr=1e-5、temperature=.07；不添加隐式批内负例，不刷新挖掘池。
- 固定 100 个训练侧 query 作 before/after 功能监控；它们参与训练，不能用它们声称泛化。监控集不会使用 item-disjoint 评估标签。
- 先在同一 1k 数据上跑 100-step 性能 Pilot，忽略前 20 step 的预热影响；只记录性能，不导出或比较开发集结果。正式训练重新从原始模型开始。
- 仅使用单卡。当前不做百万样本训练或多卡扩容；后者仍需 1/2/4/8 卡性能门槛。
- 正式训练完成并通过功能门槛后，只评估最终 checkpoint 一次；原始两路并集 @100 显著提升仍是基础门槛。本轮扩大训练的增量收益还要相对 100-query Pilot 的 85.95% 做配对比较，不因只超过 83.3% 就认定扩大成功。
- 此 2,000-query 集合已多次用于开发，配对 p 是固定集合上的证据，不当作未见测试成绩。官方公开 10k 保持封存；主办方正式 50k 不在本轮运行。

## 1. 更新代码并准备 1k 数据

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
python - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
src=Path('/code/ecommerce-retrieval-source/scripts')
dst=Path('/code/ecommerce-retrieval/scripts')
backup=dst.parent/'backups'/datetime.now().strftime('scripts-scale-%Y%m%d-%H%M%S-%f')
shutil.copytree(dst,backup)
shutil.copytree(src,dst,dirs_exist_ok=True)
print('脚本同步完成，备份：',backup)
PY
conda activate ecom-infer
bash scripts/run_embedding_scale_v1.sh data
```

这一步完整扫描负例池，日志为 `artifacts/train_v0/embedding_scale_logs/data.log`。完成时输出 dataset ready 路径；失败保留目录，先处理原因，不盲目覆盖。已知重复簇只能排除已发现的同款/近重复风险；对疑似同款的负例仍需人工核验。

## 2. 单卡性能 Pilot

```bash
conda activate ecom-train-v2
TRAIN_GPU=1 bash scripts/run_embedding_scale_v1.sh perf
cat artifacts/train_v0/embedding_perf_1k_v1/performance.json
```

日志 `embedding_scale_logs/perf.log`；GPU 资源采样为 `perf_gpu_*.csv`。性能阶段前置 100-query 基线测量不计入纯训练时间；训练 step 带 Unix 时间戳，可对齐资源 CSV。检查稳态 query/s、step 用时、峰值显存和 GPU 利用率是否稳定，100 step 完成、无 OOM/NaN/零梯度后再运行正式阶段。脚本自动检查性能报告、显存余量、数据/模型/上游源码哈希和超参数一致。若吞吐或利用率异常，先检查 CPU、图片读取和预处理，不追加更多 GPU。

## 3. 正式 1,000 step

```bash
TRAIN_GPU=1 bash scripts/run_embedding_scale_v1.sh train
cat artifacts/train_v0/embedding_train_1k_v1/summary.json
cat artifacts/train_v0/embedding_train_1k_v1/performance.json
```

仅导出最终 adapter/merged，不自动恢复失败任务；重跑必须用新的输出目录或先明确失败原因。日志为 `embedding_scale_logs/train.log`。

## 4. 全部商品重新建库与固定召回

此阶段耗时明显长于训练，复用已验证单卡建库协议。

```bash
conda activate ecom-infer
EVAL_GPU=1 bash scripts/run_embedding_pilot_eval.sh \
  /code/ecommerce-retrieval \
  /code/ecommerce-retrieval/artifacts/train_v0/embedding_train_1k_v1 \
  /code/ecommerce-retrieval/artifacts/train_v0/embedding_eval_1k_v1

python scripts/compare_embedding_recall.py \
  --baseline artifacts/train_v0/embedding_pilot_eval_v1/retrieval \
  --baseline-recall 0.8595 \
  --trained artifacts/train_v0/embedding_eval_1k_v1/retrieval \
  --output artifacts/train_v0/embedding_eval_1k_v1/comparison_vs_pilot.json
```

比较两份报告。`comparison.json` 对原始 83.3% 基线，`comparison_vs_pilot.json` 对 85.95% Pilot。若未超过 Pilot，不追调开发集，不宣称扩大有效；保留较优模型，分析训练侧问题。若扩大有明确收益，再准备固定 checkpoint-500 的新候选 Top1/Top3 精排评估和 calibration 侧动态 Top-25 阈值校准。
