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
