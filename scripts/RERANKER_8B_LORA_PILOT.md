# 8B领域排序训练 Pilot

2026-11-25。官方未做项目微调的8B未胜过当前微调2B：原test Top1 49.8%→48.8%，Top3 61.6%→62.5%，没有明确替换收益。保留现有正式提交，开始同一官方8B基座的listwise LoRA功能/方向Pilot。

本轮先回答：8B是否能在已审计的项目排序数据上正常学习；再区分相对官方8B的训练收益与相对现有2B的系统收益。真实训练与评测结果尚待服务器执行。

## 固定训练协议

- 基座：已经下载并完成推理对照的 `models/Qwen3-VL-Reranker-8B-b212dc8`，不继续微调2B，也不从功能检查的adapter续训。
- 数据：复用 `artifacts/train_v0/reranker_sft_smoke_40k` 中已审计的4,500 train / 500 smoke-eval query。固定seed20261125，以query ID稳定抽取 **400 train + 100 monitor**，不重新挖负例。
- 每query仍为1个正例和7个负例；核对canonical样本与图片ID及顺序、真实标签、已知重复组排除，并检查训练/监控query及所有正负例ID不包含2k开发集的正例或其已知重复组。
- 100条monitor仅为训练侧query holdout，可能与训练集存在商品重叠；不是独立item-disjoint确认集，不用于挑checkpoint。已知重复分组仍不能证明覆盖所有语义同款。
- LoRA rank8/alpha32、all-linear、lr=1e-5、temperature1、bf16、FlashAttention、padding-free、max_length4096、每卡batch1、梯度累积1、gradient checkpointing。
- 沿用训练环境 **ecom-train-v2**（已记录ms-swift4.6.1），冻结实际torch/transformers/peft/FlashAttention版本及listwise loss源码哈希。不升级ecom-infer或自动安装包。
- 固定 **50 optimizer steps**，step0/25/50做监控，保存25/50但只导出预先指定的step50。全局query batch=卡数；7卡名义暴露350组、8卡400组，不能称为跑完完整4,500条训练集。
- 先从上述子集中取100 train / 20 monitor做单卡5-step smoke，仅检查功能；丢弃其adapter，再从官方基座重新启动50步多卡Pilot。

训练组被展开成8个pair，每卡batch1指一组query，不是一对图文。使用strict=true和delete长度策略：异常行/超长编码在严格预处理下报错，不静默删样本或截断。对应行为已核对 [ms-swift v4.6.1预处理实现](https://github.com/modelscope/ms-swift/blob/v4.6.1/swift/dataset/preprocessor/core.py)。缺失媒体、已知假负例或开发正例泄漏均在准备阶段停止。

## 1. 同步并在训练环境准备

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
cp /code/ecommerce-retrieval-source/scripts/reranker_8b_lora_utils.py scripts/
cp /code/ecommerce-retrieval-source/scripts/run_reranker_8b_lora_pilot.py scripts/
conda activate ecom-train-v2

python -u scripts/run_reranker_8b_lora_pilot.py prepare \
  --gpus 1,2,3,4,5,6,7
```

默认新目录 `artifacts/train_v0/reranker_8b_lora_pilot_v1` 必须不存在。校验训练源manifest全部输出哈希、原8B对照计划、模型下载回执、数据身份和隔离，再保存抽样数据、query ID、环境版本、计划和文件哈希。本阶段不启动GPU训练，但会导入训练依赖确认注册的listwise loss可用。

若8卡全空闲，prepare与train都可用 `--gpus 0,1,2,3,4,5,6,7`。同一Pilot不可改变训练卡数。默认master port29525，可在prepare时显式指定 `--master-port`。若需新输出目录，所有阶段一致添加 `--output-dir`。

## 2. 启动5步功能检查和50步训练

```bash
mkdir -p artifacts/train_v0/reranker_8b_lora_logs
nohup python -u scripts/run_reranker_8b_lora_pilot.py train \
  --gpus 1,2,3,4,5,6,7 \
  > artifacts/train_v0/reranker_8b_lora_logs/launcher.log 2>&1 &
```

入口检查所选卡空闲（>1GiB占用即停止），首卡先完成5-step smoke；有限loss/非零有限梯度、初末eval、checkpoint及GPU活动均通过，才用所选卡从原基座启动50步。不会自动降低精度、减少负例、改batch或换优化器绕过问题。

```bash
tail -f artifacts/train_v0/reranker_8b_lora_logs/launcher.log
```

阶段日志：

```bash
tail -n 15 artifacts/train_v0/reranker_8b_lora_pilot_v1/smoke/train.log
tail -n 15 artifacts/train_v0/reranker_8b_lora_pilot_v1/train/train.log
```

每阶段有gpu_metrics.csv，记录所选卡利用率、显存和功耗。训练尚未进入下一阶段时，对应日志不存在。无需同时启动其他训练/推理任务。

如果训练意外中断，不自动从部分checkpoint续训改变Pilot暴露量；保留日志，定位后用新的Pilot目录。重复train只会复核已成功且有summary的阶段，不覆盖部分训练目录。

## 3. 读取训练功能和方向报告

```bash
cat artifacts/train_v0/reranker_8b_lora_pilot_v1/train/summary.json
```

也可在相同目录运行 `python scripts/run_reranker_8b_lora_pilot.py report` 重新汇总，无GPU推理。

- functional_gate：完成50步；loss、梯度、评估指标有限且存在非零梯度；有step0和50的eval_loss；最终adapter与LoRA/base配置相符；全部所选GPU确有活动。
- direction_gate：固定100条monitor的eval_loss下降，且至少一项实际记录的MRR/NDCG提高。缺失排序指标标记未通过，不把缺失当成功。
- 不直接用第一步与最后一步训练loss比较不同query；核心看同一monitor的前后变化。门槛只判断训练链路/方向，不代表泛化或官方收益。

如果方向未过，停止导出/2k评估，先看报告；不自动挑step25、改学习率或换seed追试。

## 4. 通过后仅导出固定step50

仍在ecom-train-v2：

```bash
python -u scripts/run_reranker_8b_lora_pilot.py export --gpus 1
```

校验两个门槛、最终adapter哈希和base来源，使用一张空闲卡执行既有swift export合并。输出 `merged/` 和export_receipt.json。核对架构、词表大小和多模态token ID不变，并恢复官方基座原始tokenizer/processor文件，固定输入处理，仅让模型权重变化。

## 5. 固定候选比较三个模型

切回已验证的推理环境：

```bash
conda activate ecom-infer
python -u scripts/run_reranker_8b_lora_pilot.py prepare-eval \
  --gpus 1,2,3,4,5,6,7

nohup python -u scripts/run_reranker_8b_lora_pilot.py eval \
  --gpus 1,2,3,4,5,6,7 \
  > artifacts/train_v0/reranker_8b_lora_logs/eval_launcher.log 2>&1 &
```

新增推理只有Pilot训练后8B；复用现有微调2B raw和官方8B结果。候选、原始文本、帧、instruction、batch4、512图像配置、推理token上限10240和旧路由决策不变。训练4096上限与既有2B训练协议一致，不能把训练与推理上限混为一谈。

每卡执行旧入口的工程检查后完成2k分片，最终输出：

```bash
cat artifacts/train_v0/reranker_8b_lora_pilot_v1/eval/comparison_vs_base8b.json
cat artifacts/train_v0/reranker_8b_lora_pilot_v1/eval/comparison.json
```

第一份：训练前8B vs 50步LoRA后8B，直接观察本轮领域适配效果。
第二份：现有微调2B vs Pilot训练后8B，观察系统是否值得继续投入；两者训练暴露量不同，不能视为纯参数规模对照。

GPU评测进度在 `eval/logs/pilot.log`、`eval/logs/worker_*_full.log`，总launcher在reranker_8b_lora_logs/eval_launcher.log。原2k已经反复用于开发，所有p值仅作描述；独立item-disjoint确认前不换正式提交、不重跑正式50k、不扩大训练。
