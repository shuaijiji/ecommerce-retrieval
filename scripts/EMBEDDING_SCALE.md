# 召回扩大实验 v1：固定 1,000 query

2026-11-23结果：1k模型并集Recall@100=87.10%，相对100-query Pilot的85.95%配对p=.1315，未通过扩大增量主门槛，暂停继续训练。用户已同意进入 [三套召回的固定精排对照](RETRIEVER_RERANK_COMPARISON.md)。


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

## 5. 单卡建库切换到八卡（2026-11-20）

用户要求八卡利用。可复用 `embedding_eval_1k_v1/train_item_index` 中已完成分片，分片划分/模型/权重哈希/batch size/预处理协议保持相同，不重新训练模型。仅八卡并行商品建库，最终固定2,000-query召回保持单卡。

先拉取代码，用专用工具核对并停止原 shell launcher 及其子进程。以下 PID=4137568 来自用户实际启动命令；若 PID 已退出或不匹配，工具会拒绝，不猜测新 PID，不直接使用宽泛 pkill。完成的原子 `.pt` 保留，未完成的 `.tmp` 文件不作为有效分片使用。

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
python /code/ecommerce-retrieval-source/scripts/stop_embedding_eval.py \
  --pid 4137568 \
  --output-dir /code/ecommerce-retrieval/artifacts/train_v0/embedding_eval_1k_v1
```

确认打印 `Verified launcher and workers stopped` 后，按前文 Python 备份/完整覆盖方式同步 scripts（不要只复制新入口而遗漏公共脚本），保持 `ecom-infer` 环境，不更改 PyTorch/Transformers 版本。

```bash
cd /code/ecommerce-retrieval
conda activate ecom-infer
nohup python -u scripts/run_embedding_index_8gpu.py \
  --root /code/ecommerce-retrieval \
  --model /code/ecommerce-retrieval/artifacts/train_v0/embedding_train_1k_v1/merged \
  --output-dir /code/ecommerce-retrieval/artifacts/train_v0/embedding_eval_1k_v1 \
  --gpus 0,1,2,3,4,5,6,7 \
  > artifacts/train_v0/embedding_scale_logs/eval_8gpu_launcher.log 2>&1 &
```

启动会拒绝仍在运行的旧索引 writer，解析物理 GPU UUID，检查每张卡的已有显存（>1GiB 拒绝）。不终止其他 GPU 占用，也没有忙卡绕过开关。八张卡需均可用；否则先确认占用来源和释放情况。

先按1/2/4/8卡各运行每卡最多4个新分片；这些是实际索引分片，会继续复用。性能报告记录所有worker实际编码时间窗口上的整体吞吐和逐卡利用率中位数；GPU CSV同时记录显存与功耗。八卡速度至少单卡4倍且每卡利用率中位数≥60%后才自动启动全量八卡 train/dev 商品建库；失败保留所有完成分片并停止，不放宽门槛追跑。限制每worker CPU线程数，避免8进程抢占过多线程。

```bash
tail -n 30 artifacts/train_v0/embedding_scale_logs/eval_8gpu_launcher.log
ls artifacts/train_v0/embedding_eval_1k_v1/logs/eight_gpu_*/
find artifacts/train_v0/embedding_eval_1k_v1/train_item_index -name 'part-*.pt' | wc -l
```

训练商品索引预计1493个分片；全局进度看有效分片数，不把各worker日志中的全局商品位置相加。worker按分片编号取模分配，日志中的 encoded位置不是全局完成比例。资源与worker日志位于本轮独立 `logs/eight_gpu_<时间>/`。只有完整分片校验通过才发布manifest并继续召回、生成 comparison.json。该入口尚未在远端实际执行，八卡速度必须以此次性能结果为准。
