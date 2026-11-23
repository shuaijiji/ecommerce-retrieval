# 真实增量Pilot与正式评测数据准备（2026-11-23）

## 当前结论

用户远端动态Top-25报告：主方案pilot100在固定test1000条上，Top1 .489→.498（救回12/伤害3，p=.035156），Top3 .604→.616（救回20/伤害8，p=.035698）。扩容252/1000、平均候选14.669→20.391，保留全量增量Top-25的90% Top1净收益。calibration阈值固定为.87109375，同分概率.5，seed20261119。

train1k对照test Top1 .492→.502，Top3 .613→.624，扩容239/1000，平均候选15.372→21.061，只保留全量Top-25的50%净收益。本轮仍保持pilot100主方案，不根据test换模型/预算。

这证明内部开发集的离线排序收益，不能当作正式50k成绩或端到端延迟。主办方正式50k数据是否存在、与已有公开10k的关系尚未确认。后续不追加训练，完成下面两个可独立执行的工程步骤。

## 1. 更新脚本

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
python - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
src=Path('/code/ecommerce-retrieval-source/scripts')
dst=Path('/code/ecommerce-retrieval/scripts')
backup=dst.parent/'backups'/datetime.now().strftime('scripts-readiness-%Y%m%d-%H%M%S-%f')
shutil.copytree(dst,backup)
shutil.copytree(src,dst,dirs_exist_ok=True)
print('同步完成，备份：',backup)
PY
```

## 2. 盘点本地正式评测候选数据（CPU）

```bash
cd /code/ecommerce-retrieval
mkdir -p artifacts/submission_readiness
python scripts/locate_evaluation_data.py \
  --search-root /code/ecommerce-retrieval/data \
  > artifacts/submission_readiness/data_inventory.json
cat artifacts/submission_readiness/data_inventory.json
```

逐行统计每个包含photo_info.jsonl和item_info.jsonl目录的视频/商品数、重复或非法ID、缺少媒体字段数、少量媒体路径存在性；只记录label文件是否存在，不读取答案，不解压下载、不占GPU。大型训练metadata需扫描一次，因此会有CPU/磁盘开销。

看到 `matches_expected_video_count_only=true` 只说明视频数匹配50k，**不能自动认定为主办方正式集**。仍须对照主办方分发说明确认身份、候选商品范围、媒体完整性与预测ID集合，再启动正式推理。当前没有证据时不会把10k拼凑/复制成50k，也不会启动新的正式商品建库。

如果报告未找到50k，不代表其他挂载盘也没有；需要用户给出实际数据位置，再以该路径运行同一个公开本地盘点脚本。此脚本只对指定根目录向下最多6层扫描，跳过图片树、模型和实验产物。

## 3. 主方案100-query真实新增候选打分（GPU）

```bash
conda activate ecom-infer
cd /code/ecommerce-retrieval
python -u scripts/run_trained_retriever_latency_pilot.py --gpu 1
```

使用一张当前空闲卡，可把 `--gpu 1` 改为任意空闲物理卡。单卡是为了测量单条视频的串行延迟，这个100条工程Pilot无需七卡。已有显存超过1GiB会拒绝，不终止其他任务。

入口从已完成的 `retriever_dynamic_top25_7gpu_v1/comparison.json` 读取pilot100阈值，并核对执行前experiment_config里的阈值、两套候选哈希、reranker权重/配置、数据metadata、预处理与上游代码、torch/transformers版本。不能误用原始召回的旧阈值.85546875。

实际调用固定checkpoint-500：每条先打Top-10，保存分数；符合阈值时仅打Top-25新增候选；合并分数后输出前三名。固定query seed20261105、100条，不因正确率/扩容率改变种子或阈值。记录Top1/Top3、路由率、候选数、显存、平均/P95/P99时延；100条只证明工程链路，不重做选模。首次冷启动的样本也计入，不把这组小样本尾分位当成服务SLA。

延迟范围为**视频/商品媒体处理+精排Top-10+条件新增候选精排**，不含embedding查询编码、检索、模型加载；不能直接作为正式50k完整链路吞吐。

```bash
tail -n 10 artifacts/train_v0/retriever_dynamic_latency_pilot100_v1/run.log
cat artifacts/train_v0/retriever_dynamic_latency_pilot100_v1/pilot_summary.json
```

输出目录已存在时入口拒绝重跑，先检查已有结果/错误；不删除失败目录来抹去证据。成功后保留source.json、config.json、逐query结果和汇总。

## 正式评测下一步

拿到数据盘点及主办方身份确认后，固定pilot100 embedding + checkpoint-500 +当前路由，准备正式商品池索引与多卡视频推理、预测完整性检查和XLSX输出。XLSX需photo_id文本和item_id_list字符串数组，最多3个不同商品；DataAgent仅格式校验，文件还需按主办方指定渠道提交。主办方出分时间未知。

100-query延迟Pilot可与数据准备独立开展，**不作为寻找正式数据、准备提交格式的阻塞条件**。当前仅代码准备和本地验证，尚无真实增量GPU结果，也未生成正式提交文件。
