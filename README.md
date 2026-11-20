# 电商商品识别

- [Embedding Pilot 执行说明](scripts/EMBEDDING_PILOT.md)
- [本地方法与进展记录](项目记录.md)
- [训练数据、开发集与训练入口方法](方法记录-训练数据与开发集.md)
- [Docs 方法与进展记录（2026-11-04）](https://docs.corp.kuaishou.com/d/home/fcAD9w17I0oHAJMzki28S8xM4)

截至 2026-11-20：100-query/100-step 召回 embedding 对比学习 Pilot 已通过。用户远端回报固定 item-disjoint 2,000 条上，两路 Top-100 并集 Recall 从 **83.30% 提高到 85.95%**，救回/丢失 55/2、净增 53，配对 p≈2.30e-14；平均训练侧 loss 下降、margin 改善，峰值 allocated 显存 4.63 GiB。下一轮固定为 **1,000 query / 1,000 step**，先做单卡 100-step 性能 Pilot，再训练和重建商品索引，同时比较原始基线与 100-query Pilot。扩大实验代码已准备，尚未远端执行。见 [扩大实验执行说明](scripts/EMBEDDING_SCALE.md)。最终系统 Top1/Top3 仍待新候选精排验证，公开 10k 保持封存。

- [本次验证集进展详情](进展更新-2026-11-04.md)
- [验证集审计和推理进展](进展更新-2026-11-05.md)
- [零训练多模态精排结果](进展更新-2026-11-06.md)
- [Top-25 受控对照与训练准备](进展更新-2026-11-10.md)
- [训练商品索引完成与 hard-negative 候选生成](进展更新-2026-11-11.md)
- [hard-negative 全量完成与精排 SFT 数据构造](进展更新-2026-11-13.md)
- [Qwen3-VL-Reranker 单卡训练链路通过](进展更新-2026-11-16.md)
- [Listwise 排序门槛与多卡吞吐测试](进展更新-2026-11-17.md)
- [Listwise 官方 10k 最终结果与 GRPO 下一步](进展更新-2026-11-18.md)
- [Score-policy GRPO 最小 Pilot 结果](进展更新-2026-11-19.md)
- [Item-disjoint 召回曲线与 Top-25 受控扩容](进展更新-2026-11-19-召回扩容.md)
- [Qwen3-VL-Reranker 的 GRPO 适配审计](GRPO_ADAPTATION.md)
- [已验证脚本](scripts/README.md)

远端为 8×H100 80GB。这里是本地记录目录，不是远端训练目录，文件尚未同步到远端。

精排阶段：候选生成、固定 RRF、逐候选多模态打分、100 条预跑与全量评估已在远端完成，见 [运行说明](scripts/RERANK.md) 和 [零训练精排结果](进展更新-2026-11-06.md)。本地只保存代码与记录，不包含远端数据、模型或评估产物。

2026-11-10：Top-25 对照和分数漂移受控分解已完成，结论见上述进展记录。已开始编写训练集定位、审计和 item-disjoint 划分脚本。

2026-11-11：训练商品索引和 hard-negative Pilot 已完成；8 卡全量候选生成已启动。当前产物是两路 Top-100 安全负例候选池，不是最终 SFT 样本。

2026-11-13：hard-negative 全量 983 个分片、40,000 pair 的 Qwen3-VL-Reranker pointwise SFT 数据及最终 manifest 均已通过；开始单卡训练功能 Pilot。

2026-11-16：pointwise 10-step Pilot 虽通过功能链路，但 MRR/NDCG 退化且出现负类偏置，因此不扩容；正式 4,500/500 query 上的 listwise 50-step Pilot 排序指标单调改善，下一步进行隔离评估耗时的 1/2/4/8 卡吞吐与资源利用测试。

2026-11-17：listwise 1/2/4/8 卡吞吐测试完成，8 卡达到单卡 6.19×。随后正式 8 卡 listwise v1 完成 500 step，五个 checkpoint 均已保存；step 400 的训练侧监控排序指标最高。已进入 item-disjoint 内部选模入口准备，官方验证集尚未用于 checkpoint 选择。

2026-11-18：2,000-query item-disjoint 开发评估中 checkpoint-500 以 46.60% Top-1 唯一入选。官方冻结 Top-10 10k 上达到 80.72% Top-1，相对 Base 提升 1.36pp，配对结果为救回 282/伤害 146。官方集从此封存，后续 GRPO 只使用训练侧数据和 item-disjoint 开发集调试。
