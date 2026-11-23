# 三套召回的固定精排对照（2026-11-23）

## 决策与证据

用户远端结果：1k 模型对原始召回的 Top-100 并集 Recall 从83.30%提高到87.10%，救回/伤害166/90，净增76，p=2.35e-6；对100-query Pilot从85.95%提高到87.10%，救回/伤害118/95，净增23，p=.1315。扩大训练未通过预设主指标的增量显著性门槛，停止扩大训练，不改用@10/@25的显著性追认主门槛通过。

对100-query Pilot，1k 纯视觉@10/25/50/100为57.15/66.35/71.85/77.60%，分别降低3.05/2.00/2.05/1.55pp；多模态为65.50/73.05/78.15/83.05%，分别提高4.75/2.35/1.20/.95pp。保留两套模型，下一步以完整系统精排收益判断实用价值，不宣称扩大训练已稳定胜出。

用户已同意以下精排对照。代码已做本地合成数据验证，真实GPU执行尚未发生。

## 固定实验口径

- 同一 item-disjoint 2,000 query，完整分母，不只评估召回命中的子集。
- original=`item_disjoint_eval_v1/retrieval`；pilot100=`embedding_pilot_eval_v1/retrieval`；train1k=`embedding_eval_1k_v1/retrieval`。
- 各自视觉/多模态每路Top-10并集作为候选，候选生成不加载标签。不补真值、不改变商品池、不再建库。
- 同一 `reranker_listwise_train_v1_merged/checkpoint-500-merged`，batch size=4，其余沿用已验证的 rerank_multimodal.py 配置；原始召回组也重新打分，保证本次代码/环境/数值协议一致。旧预测可保留作历史结果，但不混入本次对照。
- 主指标完整系统Top1，次指标Top3，附候选Recall、MRR/NDCG、平均候选数、平均逐视频打分耗时。三组两两报告Top1/Top3的救回、伤害、净增和双侧精确配对p；不把探索性多组比较当成独立测试成绩，不自动晋升模型。
- 暂不加动态Top-25，不重新训练精排，不运行公开10k或正式50k。

## 同步代码

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
python - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
src=Path('/code/ecommerce-retrieval-source/scripts')
dst=Path('/code/ecommerce-retrieval/scripts')
backup=dst.parent/'backups'/datetime.now().strftime('scripts-rerank-%Y%m%d-%H%M%S-%f')
shutil.copytree(dst,backup)
shutil.copytree(src,dst,dirs_exist_ok=True)
print('同步完成，备份：',backup)
PY
```

## 运行

先确认GPU空闲。默认5/6/7只是此前截图显示空闲的卡，不代表现在仍可用；启动时每卡已有显存>1GiB会拒绝。可用 `--gpus 1,5,6,7` 指定四卡，或八卡均空闲时指定0–7，不抢占其他任务。入口把同一query划分到相同worker，在每卡顺序跑三组。每个分片先100条媒体/分数功能Pilot，再完整分片并复用已完成结果；每组2,000条，3组共6,000条精排。

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval
mkdir -p artifacts/train_v0/retriever_rerank_logs
nohup python -u scripts/run_retriever_rerank_comparison.py \
  --root /code/ecommerce-retrieval --gpus 5,6,7 \
  > artifacts/train_v0/retriever_rerank_logs/launcher.log 2>&1 &
echo "PID=$!"
```

如仅想先检查三套候选，可添加 `--prepare-only`，不启动GPU。分卡数固定在输出配置中；更改卡数时用新的 `--output-dir`，避免改变分片边界。保持分卡数、代码、模型和数据不变时，重跑会复用本次各worker的已完成query结果。启动器对输出目录加互斥锁，防止重复运行。

```bash
tail -n 20 artifacts/train_v0/retriever_rerank_logs/launcher.log
tail -n 5 artifacts/train_v0/retriever_rerank_top10_v1/logs/original_worker_0_full.log
```

开始后按launcher日志的arm/worker/stage找对应日志；未开始的full日志不存在是正常的。结束后：

```bash
cat artifacts/train_v0/retriever_rerank_top10_v1/comparison.json
```

报告核对三组query覆盖、每个预测完整候选排名、有限有序分数、相同reranker权重/预处理/环境和所有分片来源哈希。候选外真值按Top1/Top3失败计算。对1k训练是否改善最终系统Top1/Top3作判断后，再决定是否做新的动态Top-25验证；不直接沿用旧阈值宣称上线收益。
