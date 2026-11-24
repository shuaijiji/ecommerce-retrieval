# 首次提交后的核验与错误分解

2026-11-24：用户截图回报排名13/19，Top1 **70.8750%**、Top3 **82.9300%**；第一名截图为83.1125%/93.0100%，分别相差12.2375/10.0800个百分点。截图是用户提供的成绩快照，非实时榜单查询。

## 已核验与仍待核验

下载文件、GitHub `submissions/2026-11-24/submission_top3.xlsx` 和导出manifest的SHA-256一致：

`77f205be74c06c2c2d8abb281e4e58c4b39225f04b222acc8ff1829f004d8a64`

本地回读：2,019,786字节，一个可见sheet，两列，50,000个唯一文本视频ID，每行3个不同的文本商品ID，无公式、宏或外部链接。服务器预测逐条比对与正式提交回执仍待核验。manifest中的提交/格式服务状态是导出时快照，不能据此否认用户后来提交并获得成绩。

另有评分口径疑问：若50,000条全部等权计分，70.8750%对应35,437.5条命中，不能为整数。**不能由此认定实际评分40k**。建议用户向主办方确认（本流程不会代发）：

> 请确认此次有效评分视频数、是否过滤或加权，以及本次提交对应的文件标识。文档标注50k视频，但Top1 70.8750%按50k等权换算不是整数命中数。

## 先用已有结果拆错误

以下为已保存报告的代数分解，尚待服务器逐条重建。仅属于内部2,000条开发集，不能解释正式测试集每条错误或直接预测榜单收益。

| Top1错误来源 | 冻结Pilot方案 | 1k对照 |
|---|---:|---:|
| 两路Top100均未召回 | 281 | 258 |
| Top100召回、Top25截断 | 200 | 185 |
| Top25含正确商品，但未触发扩容 | 107 | 96 |
| 已打分，正确商品排2–3 | 237 | 241 |
| 已打分，正确商品排4及以后 | 185 | 217 |
| 错误合计 | 1,010 | 997 |

Pilot已打分候选内精排错误共422条；仅改精排的当前候选覆盖上界70.60%，全部Top25上界75.95%，Top100上界85.95%。这些上界均来自内部开发集。该2k已反复用于开发；后续改模型需增加独立确认集，不能作为新的未见测试集。

## 服务器执行：仅CPU审计

同步只增加两个审计入口，保留冻结推理代码，避免覆盖plan记录的代码哈希：

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
python - <<'PYCODE'
from pathlib import Path
import shutil
src=Path('/code/ecommerce-retrieval-source/scripts')
dst=Path('/code/ecommerce-retrieval/scripts')
for name in ('audit_submission_readonly.py','audit_retriever_error_budget.py'):
    target=dst/name
    if target.exists() and target.read_bytes()!= (src/name).read_bytes():
        raise RuntimeError(f'审计脚本已存在且不同，请先检查：{target}')
    shutil.copy2(src/name,target)
print('审计入口同步完成')
PYCODE
conda activate ecom-infer
mkdir -p artifacts/post_submission_audit
python -u scripts/audit_submission_readonly.py \
  --xlsx artifacts/official_submission_v1/submission_top3.xlsx \
  --official-root artifacts/official_submission_v1 \
  --output artifacts/post_submission_audit/submission_audit.json
python -u scripts/audit_retriever_error_budget.py
```

第一个脚本核对XLSX、manifest、冻结模型/代码和50k预测逐条一致；第二个核对现有2k候选/预测/精排协议/阈值来源，重建校准与test分组的错误，检查复现原test Top1/Top3。需要已有模型和结果文件，但不加载模型、不用GPU。

第二个脚本输出：

- `artifacts/train_v0/error_budget_after_submission_v1/summary.json`
- `pilot100_cases.jsonl`、`train1k_cases.jsonl`：全部逐条分解。
- `pilot100_review_samples.jsonl`、`train1k_review_samples.jsonl`：每个错误桶固定抽10条，附文本、帧路径、正例/预测标题与图片路径，供人工检查。

输出目录必须不存在；失败后保留证据，检查原因后用 `--output-dir` 指定新目录重跑。审计会扫描训练metadata以补充抽样资料，耗时取决于磁盘。无新商品建库、无模型训练、无正式测试标签读取。

完成后回传两个summary及抽样文件，再依据真实错例确定下一轮：候选保持一致的更强精排对照，或视觉/多模态联合训练。新方案先过训练侧与独立确认集，再生成下一次正式提交。
