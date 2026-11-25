# 固定候选的精排文本去重对照

2026-11-25。缓存排名融合已失败：calibration权重0/.1/.25/.5/1的Top1分别49.2%/46.7%/42.6%/39.9%/33.3%，选中0，原test维持49.8%/61.6%。停止融合权重搜索，保留正式提交。

本轮只比较 **原始OCR/ASR vs 完全重复段去重**，两组重新精排同一批2,000条开发query。代码与本地测试已完成，真实GPU结果尚待服务器执行。

## 固定内容与唯一处理变化

- checkpoint-500、instruction、视频帧、商品图片/标题、batch4、512图像配置、10240 token上限和每字段预算不变。
- 正式Pilot召回的现有Top10/Top25候选、旧Top10分数计算的扩容决定均冻结，去重组的新分数不参与路由。
- 使用旧策略实际会打分的候选：不扩容query仅原Top10并集，扩容query为原Top25并集；两组候选ID和顺序完全相同。准备候选时不使用标签或真值排名选择商品。
- 仅对photo_ocr/photo_asr按竖线或换行切段，以空白规整后的完整段是否相等判断重复，保留首次出现。保留大小写、标点、品牌、型号、数字和不同规格；不做模糊去重、关键词抽取、文本重写或句内删词。
- 在原有字段截断前去重，可能让原来被重复段挤到预算外的尾部内容进入输入。这是本次处理的一部分；字段预算本身不扩大，不删除其他文本字段，不修改商品标题。
- 为两组各生成仅含这2k视频的metadata；通过符号链接读取同一套原始媒体和商品metadata，不复制/改写原数据，也不改已有推理脚本。

## 为什么重新跑原始组

当前固定候选一次打分，与此前Top10缓存加新增Top25分数的批次划分可能不同，浮点分数也可能有漂移。因此主对照必须是本轮raw vs dedup；另报告cached_vs_raw用于定位批次/运行环境漂移，不能把相对旧缓存的收益全算作去重收益。

旧2k已反复用于开发，保留原calibration/test划分仅为可比性，不作为新的未见确认集。配对p值只作描述；本轮不自动替换正式提交、不改标签、不训练、不重新建库、不重跑正式50k。只有方向有效且Top3无明显退化，才值得安排独立确认或进一步精排模型对照。

## 同步新文件

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
cp /code/ecommerce-retrieval-source/scripts/reranker_text_dedup.py scripts/
cp /code/ecommerce-retrieval-source/scripts/run_reranker_text_ablation.py scripts/
conda activate ecom-infer
```

只同步这两个新入口。prepare会核对旧推理/预处理脚本、模型、商品/视频metadata、候选和预测缓存的原始哈希；不匹配时检查来源，不覆盖旧脚本绕过核验。

## 1. CPU准备并冻结

```bash
python -u scripts/run_reranker_text_ablation.py prepare \
  --gpus 1,2,3,4,5,6,7
```

默认目录：`artifacts/train_v0/reranker_text_ablation_v1`，必须不存在。输出plan.json、两组metadata及原图链接、text_preview.jsonl、frozen_routing.jsonl、候选分片和旧baseline逐条分类。准备会扫描原metadata并校验模型文件，耗时取决于I/O；此阶段不调用GPU。

若8卡全部空闲，可在prepare和run中均指定 `--gpus 0,1,2,3,4,5,6,7`。预先选定worker数，不能在同一输出目录改变卡数；可更换为同数量的空闲卡。

## 2. 空闲卡执行两组精排

```bash
mkdir -p artifacts/train_v0/reranker_text_ablation_logs
nohup python -u scripts/run_reranker_text_ablation.py run \
  --gpus 1,2,3,4,5,6,7 \
  > artifacts/train_v0/reranker_text_ablation_logs/launcher.log 2>&1 &
```

会检查所选卡显存占用（>1GiB则停止），每卡顺序跑raw/dedup；每组先100条工程Pilot，再完成该卡分片。100条结果在同配置下复用；Pilot仅检查预处理、视觉张量、有限分数和token上限，不按Pilot准确率筛选方案。1–8卡均支持，最多两组各2k精排，无新召回或训练。

这次是两组精排，耗时应以worker日志为准，不能套用之前单组或纯CPU审计耗时。

## 进度和结果

```bash
tail -f artifacts/train_v0/reranker_text_ablation_logs/launcher.log
```

实际逐query进度在：

```bash
tail -n 2 artifacts/train_v0/reranker_text_ablation_v1/logs/*_full.log
find artifacts/train_v0/reranker_text_ablation_v1/raw -path '*/results/*.json' | wc -l
find artifacts/train_v0/reranker_text_ablation_v1/dedup -path '*/results/*.json' | wc -l
```

raw/dedup各2,000条即完成推理，入口会自动合并并检查候选覆盖、模型/评分协议和实际query_text与冻结preview一致，输出：

```bash
cat artifacts/train_v0/reranker_text_ablation_v1/comparison.json
```

重点看metrics.test.top1/top3（baseline是本轮raw，trained是dedup）、gained/lost/net，以及cached_vs_raw；同时记录text_changed、平均时延与峰值显存。paired_cases.jsonl保存全部逐条Top3便于复盘。

中断后重跑同一个run可复用已完成结果；不要重跑prepare覆盖计划。只重新汇总可执行 `python scripts/run_reranker_text_ablation.py report`。计划、模型或输入变化会停止，另建输出目录重新准备。


## 已完成结果：不采用去重

用户已回报真实comparison.json，计划SHA-256为53e471540bacca60a0ab1ba28c54845947cc470ecb267fee0d41ef6e1e557b83。结果保存在records/2026-11-25/reranker_text_ablation_reported.json，系用户终端转录，非助手远端直读。

| 分组 | raw Top1 | dedup Top1 | 救回/伤害 | raw Top3 | dedup Top3 |
|---|---:|---:|---|---:|---:|
| calibration1000 | 49.2% | 49.3% | 1 / 0 | 61.2% | 61.2% |
| 原test1000 | 49.8% | 49.6% | 0 / 2 | 61.6% | 61.6% |
| 全部2000 | 49.50% | 49.45% | 1 / 2 | 61.4% | 61.4% |

test Top1净-2、p=.5；这不是“显著退化”的证据，也没有支持采用该处理的收益。原方案保留，停止去重规则搜索。两组候选覆盖相同（全部70.6%、test71.4%）。原缓存与raw的Top1逐条一致；Top3全部有2救回/1伤害，说明微小批次/运行漂移确实存在，比较dedup时必须用本轮raw。

208条（10.4%）渲染文本发生变化，另1792条未变化。总体的1救回/2伤害不能在未核验前全归到208条。新增独立只读汇总脚本，无需改动已被plan冻结的实验入口，也不重新推理：

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
cp /code/ecommerce-retrieval-source/scripts/summarize_text_ablation_subsets.py scripts/
python scripts/summarize_text_ablation_subsets.py
```

结果保存到原实验目录changed_text_summary.json，分别报告calibration/test/all中的changed/unchanged子集命中率、救回/伤害、未变文本的预测漂移。验证原计划/preview/split哈希、全部ID覆盖和逐条Top3能重现既有comparison指标。子集为空时指标为null，避免混淆0命中率与无样本。

该补充仅用于归因，不再选新去重方案或改阈值。后续实验应转向固定候选的精排模型能力/训练对照，并使用独立确认集，而非继续依据这2k试文本小规则。
