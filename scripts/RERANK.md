# 零训练精排流水线

2026-11-05 完成本地实现与合成回归测试；2026-11-06 完成远端 GPU 预跑和 10,000 条全量运行。本地仍没有数据、召回产物和模型，真实指标以 `进展更新-2026-11-06.md` 记录的远端输出为准。

## 输入与输出

传入完整召回输出目录，该目录应包含 `results/vision/*.json` 和 `results/multimodal/*.json`，各 10,000 条；字段兼容现有 `evaluate_retrieval.py`。不猜测实际完整结果目录名称。

1. `rerank_pipeline.py candidates`：Top-10 并集去重，每个视频输出 photo_id、vision_top1、候选 item_id、vision_rank/similarity、multimodal_rank/similarity。两路排名从完整已保存 Top-100 中取；超出保存范围记 null。不会读取源结果的 positive_item_id、positive_score、rank，也不读取 label.json。
2. `rrf`：等权 RRF，k=60；使用两路已保存排名，缺失路贡献 0，同分按 item_id 升序。参数预先固定，不根据验证标签搜索参数。
3. `rerank_multimodal.py`：Qwen3-VL-Reranker-2B 对每个视频—商品独立计算相关性。默认每批 1 个候选，全候选排序；不将召回分数作为模型输入。官方 yes/no 分数仅用作排序，不视为校准后的正确率。
4. 固定种子打乱全部候选，不读取标签，前 100 条用于流程预跑；全量重用这 100 条结果。预跑检查图像/视频 tensor 及数量、有限分数、输入未被截断、排序非全局恒定，记录输入帧路径、文本、token 数、耗时和显存。预跑评估失败则 shell 停止；不以验证集效果选择超参数。
5. 全量只在同配置预跑通过后执行，逐视频原子保存。改变模型文件、上游代码、预处理、输入或 batch size 时要求新输出目录。

模型使用单卡 bf16/SDPA。视频按既有协议 512×512 等比例填充、奇数帧复制末帧；商品图也填充至 512×512，标题最多 500 字。视频文本沿用现有字段限额。超 token 预算直接报错，不接受静默截断或 NULL 视觉退化。

## 远端执行

将压缩包上传至 `/code/ecommerce-retrieval/` 后：

```bash
cd /code/ecommerce-retrieval
tar -xzf ecommerce-rerank-scripts-2026-11-05.tar.gz
conda activate ecom-infer

# 此模型与已有 Embedding 模型不同，需单独下载。
modelscope download --model qwen/Qwen3-VL-Reranker-2B \
  --local_dir /code/ecommerce-retrieval/models/Qwen3-VL-Reranker-2B

# 替换为真实的完整 10,000 条召回结果目录。
bash scripts/run_rerank.sh /实际完整召回结果目录
```

既有官方仓库需含 `src/models/qwen3_vl_reranker.py`，依赖环境需能导入该类；不要自动升级已验证推理环境。首次应检查 pilot.log、vlm/pilot_pass.json、vlm/results/ 下的分数和输入，以及 vlm_pilot_metrics.json。shell 默认在功能预跑和评估成功后继续全量；如需人工查看再继续，可按 shell 中的命令分步运行。

默认输出 `artifacts/rerank_v0/`：candidates.jsonl、rrf.jsonl、rrf_metrics.json、vlm_pilot_metrics.json、vlm_full_metrics.json，及逐视频完整分数。可通过 ECOM_DATA_ROOT、ECOM_RERANK_MODEL 覆盖路径。

## 指标口径

- 完整系统 Top-1 = 预测正确视频数 / 所有评估视频数，候选缺失真值仍计错误。
- 候选内精排准确率 = 预测正确视频数 / 真值在候选内的视频数。
- 候选 Recall = 真值在候选内的视频数 / 所有评估视频数。完整 Top-10 并集要求回读为 90.39%，不匹配则停止；100 条子集报告自己的实际 Recall。
- 视觉正确改错率 = 原视觉 Top-1 正确但精排错误的视频数 / 原视觉 Top-1 正确视频数。同时报告其占全部视频的比例。
- 单视频耗时 = 该视频所有候选的图片读取、预处理、GPU 推理及排序时间，GPU 同步计时；不含初始化、结果落盘和初始召回。报告均值/P50/P95/P99及按候选数分组均值，首条冷启动保留。RRF 仅统计融合排序时间，不能与 GPU 端到端时间视作同口径。

完整集应满足 system_top1 = candidate_recall × candidate_rerank_accuracy。历史纯视觉 Top-1 为 64.09%，应在新报告中回读核对。只有拿到真实运行输出才能报告新准确率或实际速度。

官方实现参考：https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py

## 错误分解与 Top-25 对照

训练前固定先运行以下对照；它不会改动 Top-10 结果，也不会用标签生成候选：

```bash
bash scripts/run_error_analysis_top25.sh \
  /实际完整召回结果目录 \
  artifacts/rerank_v0 \
  artifacts/rerank_top25_v0
```

脚本首先对已完成的 Top-10 全量结果做逐例错误分解，再在全新目录中生成“纯视觉 Top-25 ∪ 多模态 Top-25”候选，以同一个冻结的零训练 Reranker 完成功能预跑和 10,000 条评估。默认 batch size 为 4，可用 `ECOM_RERANK_BATCH_SIZE` 覆盖；改变 batch size 必须使用新的输出目录。

主要产物：

- `rerank_v0/analysis/error_summary.json`：Top-10 的候选未命中、候选内选错、正确三分法，以及视觉首名到精排结果的转移。
- `rerank_v0/analysis/error_cases.jsonl`：逐视频真值/预测、两路真值排名、分数间隔、候选数、帧数和文本字段是否存在。
- `rerank_top25_v0/analysis/error_summary.json`：同口径 Top-25 错误分解。
- `rerank_top25_v0/analysis/top10_vs_top25_summary.json`：成对候选召回与 Top-1 净变化、新增正确、退化错误及每个新增候选命中的净转化。
- `rerank_top25_v0/analysis/top10_vs_top25_cases.jsonl`：每条视频的成对状态，便于抽样复核。

Top-25 候选 Recall 必须回读为 93.62%，否则流程停止。Top-25 采用全新推理，不复用 Top-10 分数，从而保留完整、可比的端到端延迟口径。只有全量成对结果证明新增正确明显超过退化错误，才据此判断扩容价值；否则优先分析候选内 1,103 条错误，而不是直接训练。
