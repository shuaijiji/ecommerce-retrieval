# 固定候选：官方8B精排与微调2B对照

2026-11-25。本轮从模型替换入手：官方Qwen3-VL-Reranker-8B（未做本项目微调）对照现有listwise checkpoint-500（2B、已做项目微调）。**模型大小和项目微调状态同时不同，这不是纯参数规模消融**；8B胜出才值得独立确认，未胜出也不能证明8B微调无效。

官方来源：[模型说明](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B)、[固定修订](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B/tree/b212dc8c91a8164aef1ea2de9c1a867611e75c04)。官方提供8B的图文/视频精排能力；该赛题收益与H100实际显存、耗时仍需本轮验证。沿用现有Transformers入口，不安装Sentence Transformers或vLLM，不升级运行环境。

## 固定协议

- 复用已完成文本A/B实验的 **raw** 2B结果，不采用去重结果。
- 同一2k query、同一原始文本、帧、商品图片/标题、batch4、512图像配置、token上限10240、instruction、seed。
- 冻结2B旧分数生成的候选与扩容决定，8B分数不参与路由。不直接将2B的分数阈值应用于8B。
- 仅重排原策略已打分的候选，两组候选覆盖不变。原test候选覆盖率71.4%是这批内部数据的精排上界，不等于正式测试集上界，内部49.8%也不能直接与榜单70.875%比较。
- 官方8B权重、tokenizer/processor文件均记录本地哈希，固定HF revision `b212dc8c91a8164aef1ea2de9c1a867611e75c04`。原有推理脚本不改动。
- 下载、CPU冻结 → 单卡100条工程Pilot → 空闲多卡2k精排 → 配对报告。Pilot不使用准确率挑模型或调参数。
- 原2k已多次用于开发，calibration/test仍只用于可比性；p值是描述性结果，无自动晋升，正式50k不重跑。方向有效后另做独立确认。

## 1. 同步与下载（不使用GPU）

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
cp /code/ecommerce-retrieval-source/scripts/download_qwen_reranker_8b.py scripts/
cp /code/ecommerce-retrieval-source/scripts/run_reranker_8b_comparison.py scripts/
conda activate ecom-infer
python -u scripts/download_qwen_reranker_8b.py
```

下载来自官方HF仓库，四个权重分片总计约17.5GB，另有tokenizer/processor小文件；本入口不下载或执行仓库Python代码。默认存到 `models/Qwen3-VL-Reranker-8B-b212dc8`，不要覆盖已有2B模型。下载中断可重跑；成功后写download_receipt.json，再次运行会核对文件哈希并复用。需要服务器能够访问该仓库；网络错误不表示GPU问题。

## 2. CPU准备

```bash
python -u scripts/run_reranker_8b_comparison.py prepare \
  --gpus 1,2,3,4,5,6,7
```

默认输出目录 `artifacts/train_v0/reranker_8b_comparison_v1` 必须不存在。校验上一轮原始组计划、候选、预测、源数据、代码、2B模型及8B下载回执，冻结输入哈希并划分多卡分片。保留原calibration/test各1000条。只复用原始组，不复制媒体或重建索引。

若0–7共8卡全部空闲，prepare和run均改为 `--gpus 0,1,2,3,4,5,6,7`。同一计划不能改变worker数量，可以更换为同数量的空闲卡。

## 3. 100条功能预跑，再完成2k

```bash
mkdir -p artifacts/train_v0/reranker_8b_logs
nohup python -u scripts/run_reranker_8b_comparison.py run \
  --gpus 1,2,3,4,5,6,7 \
  > artifacts/train_v0/reranker_8b_logs/launcher.log 2>&1 &
```

所选卡>1GiB显存占用即停止；先在首卡跑100条，验证模型加载、原有多模态接口、视觉张量、无静默截断及有限分数。通过后才启动完整多卡任务。各worker继续使用旧入口自身的100条工程检查后完成分片；分片内100条结果复用。全局Pilot与worker分片配置不同，分开保存，不能交叉复用。

不预先承诺8B耗时/显存。若OOM或接口不兼容，保留日志定位，不自动改batch、精度或升级环境，以免改变对照协议。复用旧2B结果的前提是其余评分协议一致；运行环境版本变化会拒绝比较。

## 进度

开始校验会读取模型文件哈希，worker日志尚未生成是正常阶段；先看launcher：

```bash
tail -f artifacts/train_v0/reranker_8b_logs/launcher.log
```

单卡功能Pilot进度：

```bash
tail -n 5 artifacts/train_v0/reranker_8b_comparison_v1/logs/pilot.log
```

进入全量后：

```bash
tail -n 2 artifacts/train_v0/reranker_8b_comparison_v1/logs/worker_*_full.log
find artifacts/train_v0/reranker_8b_comparison_v1/base8b -path '*/results/*.json' | wc -l
```

每条仍是一个query，最终全量2,000条（不含独立Pilot目录）。同配置中断后重跑run复用已有结果，不重跑prepare。

## 结果与判断

入口完成后自动生成：

```bash
cat artifacts/train_v0/reranker_8b_comparison_v1/comparison.json
```

重点看metrics.test.top1/top3、gained/lost/net、candidate_recall，以及给定正例在候选内的排序正确率。字段baseline为微调2B，trained为官方8B；字段名沿用比较工具，**trained不表示本轮训练了8B**。同时记录平均单query秒数、峰值显存、paired_cases.jsonl和预测哈希。

只重新汇总可执行 `python scripts/run_reranker_8b_comparison.py report`。报告核对两组候选覆盖、运行协议（除模型路径/配置/权重以外必须相同）、逐条原始文本与帧路径一致，以及实际8B权重与固定下载回执一致。

有明确方向性改善，再构建独立item-disjoint确认集，随后才决定8B微调或正式50k推理；不因单个小p值直接更换提交。当前正式XLSX和已训练checkpoint继续保留。
