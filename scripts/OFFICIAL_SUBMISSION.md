# 正式50k推理与提交文件

2026-11-24状态：用户已确认正式50k召回与精排JSON各50,000条、后台任务Done。XLSX及manifest、full_summary仍待回读；未确认DataAgent校验或正式提交。结果见 [本轮实验记录](../进展更新-2026-11-24-正式50k推理完成.md)。以下保留执行说明供复现。


## 数据与冻结方案

用户已从主办方的 `viewfs://hadoop-lt-cluster/home/mmu/wulei/data/competition/test_data.zip` 下载并解压，盘点目录 `/code/ecommerce-retrieval/data/test/extracted/test_large_package`：50,000个唯一视频、100,000个唯一商品，无label，无重复/非法ID、无缺少媒体字段，抽样媒体路径存在。

数据规格来源：[训练和评测数据](https://docs.corp.kuaishou.com/d/home/fcABDgdJxN3Qy-PdXvJDh9T4i)。提交规格来源：[命题25评测说明](https://docs.corp.kuaishou.com/d/home/fcAAgNywX_QY7NHRRTQ5HknjR)。正式商品池仅用此包的全部100,000商品，不能混入训练/开发商品池。

冻结方案：

- 召回：`artifacts/train_v0/embedding_pilot_v1/merged`，即100-query Pilot，**不是1k模型**。
- 视觉/多模态共享该模型，各检索Top-25；Top-10并集先精排。
- 精排：`reranker_listwise_train_v1_merged/checkpoint-500-merged`，batch4、512图像配置、max length10240；复用既有query文本/帧协议和商品标题500字符协议，不允许静默NULL预处理或截断。
- 路由：固定 `score < .87109375`；相等时seed20261119稳定哈希概率.5。不在正式视频上再拟合阈值或强制25%配额。
- 触发扩容才给Top-25中的新增商品打分，Top-10分数保持不变；最终按分数、item_id确定性排序，输出前三名。
- 不训练、不读取label、不计算准确率、不在正式集上选模型。与提供label的开发评估入口分离。

最新用户真实100-query增量Pilot已完成：扩容18条，Top1/Top3不变且无新增伤害，平均精排时延1.481→1.872s（+26.4%），P95=5.087s，P99=5.610s，峰值4.64GiB；这些时间不含embedding和检索。下一步是正式测试数据上的工程功能Pilot，不追加模型实验。

## 同步

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
python - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
src=Path('/code/ecommerce-retrieval-source/scripts')
dst=Path('/code/ecommerce-retrieval/scripts')
backup=dst.parent/'backups'/datetime.now().strftime('scripts-official-%Y%m%d-%H%M%S-%f')
shutil.copytree(dst,backup)
shutil.copytree(src,dst,dirs_exist_ok=True)
print('同步完成，备份：',backup)
PY
conda activate ecom-infer
```

所有stage默认数据目录如上、输出目录 `artifacts/official_submission_v1`、GPU物理索引1–7。每阶段可显式传`--gpus`；卡数须与prepare冻结的worker数相同，可以换成同数量的空闲卡。GPU0若仍有其他占用继续避开，不设置忙卡绕过参数。切换为不同卡数需新输出目录及重新prepare。

## 1. 全媒体审计并冻结配置（CPU）

```bash
python -u scripts/run_official_submission.py prepare
```

严格验证50k/100k元数据，检查所有媒体路径在数据根目录内且文件存在，以8个CPU线程解码全部视频帧与商品图片，损坏/缺失立即停止。然后核对模型权重与已完成的pilot100开发评估一致、阈值与执行前登记一致，保存plan.json及data_audit.json。冻结数据、代码、模型配置与权重哈希、worker数和100-query确定性Pilot ID。准备时不创建GPU推理任务。

## 2. 正式商品池建库（多卡）

```bash
python -u scripts/run_official_submission.py index
```

只编码正式100,000个商品，复用build_item_index.py的既有batch16、512分片、输入协议。先按1/2/4/7卡每卡最多2个新分片测吞吐，性能分片用于真实索引并继续复用；七卡吞吐需至少达到单卡3.5倍（所选卡数的一半），否则停止并保留分片。GPU CSV记录利用率/显存/功耗供诊断，不自动抢占更多卡或修改batch来绕过门槛。

门槛通过后完成全部分片并校验manifest，保存index_complete.json。固定配置下中断后可重跑index，复用已完成分片与已通过的性能门槛。100k商品约196分片，远小于此前764k建库，但不据此承诺耗时，以日志为准。

## 3. 正式数据100-query全链路Pilot

```bash
python -u scripts/run_official_submission.py pilot
cat artifacts/official_submission_v1/pilot_summary.json
```

跨固定workers选100个确定性视频，先用embedding两路检索，再卸载embedding进程、启动reranker，真实Top-10加条件新增候选打分。结果写入按photo_id保存的retrieval/与predictions/，全部结果验证通过后生成pilot_pass.json。检查媒体、有限分数、候选完整性、实际路由和显存；**没有准确率/Top1分数，因为不读取真值**。

summary包含实际检索/精排平均耗时、扩容率、峰值显存、阶段wall time，以及根据逐query耗时的全量线性估算（明确排除模型初始化/导出，不保证实际多卡墙钟）。结果不能用于调阈值或更换模型。100条Pilot的有效结果会在全量任务中复用。

前三步建议在后台串行运行，失败即停止：

```bash
mkdir -p artifacts/official_submission_logs
nohup bash -c 'set -e
python -u scripts/run_official_submission.py prepare
python -u scripts/run_official_submission.py index
python -u scripts/run_official_submission.py pilot' \
  > artifacts/official_submission_logs/prepare_index_pilot.log 2>&1 &
echo "PID=$!"
```

```bash
tail -n 20 artifacts/official_submission_logs/prepare_index_pilot.log
```

详细worker和GPU采样见 `artifacts/official_submission_v1/logs/<时间>/`，文件名带index/pilot/full阶段及worker编号。

## 4. 全量50,000视频

仅在正式数据Pilot成功后运行；脚本强制校验pilot、索引、计划哈希一致，不允许跳过。

```bash
nohup python -u scripts/run_official_submission.py full \
  > artifacts/official_submission_logs/full.log 2>&1 &
echo "PID=$!"
```

七卡按排序后的photo_id取模分配，各卡使用完整正式商品库；检索和精排分别一组GPU进程，避免同时驻留两个模型。每条原子保存结果，重跑先校验并复用。错误时停止任务组、保留已完成结果；不自动丢视频、不补空数组掩盖失败。输出目录锁防止重复launcher。

```bash
tail -n 20 artifacts/official_submission_logs/full.log
find artifacts/official_submission_v1/retrieval -name '*.json' | wc -l
find artifacts/official_submission_v1/predictions -name '*.json' | wc -l
```

全量结束自动汇总full_summary.json并导出提交XLSX。注意一个阶段的逐worker `n/total`是该worker自己的进度，不把全局photo_id位置当成已完成总数。

## 5. 导出与提交

完整覆盖50,000视频后才导出：`artifacts/official_submission_v1/submission_top3.xlsx`。如仅需重新检查和导出，可运行：

```bash
python scripts/run_official_submission.py export
cat artifacts/official_submission_v1/submission_top3.manifest.json
```

导出再次验证：预测ID集合与测试集完全相同、每条检索候选属于正式商品池、Top-10缓存分数不变、扩容决策与固定阈值一致、最终候选完整、分数有限有序、Top-3唯一。存在缺失/额外视频或失败结果时拒绝导出。

服务器上的独立导出器使用Python标准库构造OOXML，不依赖服务器安装桌面表格运行时；仅一个可见sheet、严格两列photo_id/item_id_list、全部单元格用字符串，数据列文本格式，最多三个不同商品，无公式/宏/外部链接。保存后重新读取ZIP/XML逐格验证与原始字符串完全一致、文件≤10MiB，避免Excel舍入长ID。测试覆盖长ID、前导零和部分结果拒绝。

**此时只是本地文件生成，并未正式提交或获取分数。** 用户将文件下载上传DataAgent，运行competition-evaluator-25（市场ID10781）的格式校验，再通过主办方指定渠道提交同一个Excel及参赛人标识。主办方出分时间仍未知。

当前交付是代码，CPU合成检查已通过；真实正式数据GPU Pilot和全量预测尚未执行，不报告未知的正式准确率。
