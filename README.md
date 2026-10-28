# cf_Rec — LGMRec / LGMRecOpt

## 当前发布：用户指定的 Recall@10 最佳版

Baby seed999：完整 VALID Recall@10=**0.066458264**，Recall@20=0.103411760，NDCG@20=0.045481086。200轮，最佳 epoch=194（从0起，仍按完整 VALID R20 选模）。本次发布第二模块 v1 存档，不包含后续 β 版本。

此次实验期间覆盖过服务器代码，结果需固定源码复跑确认；不能视为固定版本的正式复现。发布存档已核对与启动源码指纹一致。无 TEST 结果、无已保存权重。详情见 [本次发布说明](docs/20261028-123000-改动说明.md)。


按官方 LGMRec 目录组织：https://github.com/georgeguo-cn/LGMRec 。LGMRecOpt 已独立实现，仅继承通用接口 GeneralRecommender，不导入或继承 LGMRec 类。算法仍来自 LGMRec 优化研究，不称为独立 TRD-HRec。

## 结构

```text
cf_Rec/
  common/                  通用模型接口、训练器
  configs/dataset/         baby、sports、clothing
  configs/model/           两个模型各自的配置
  configs/overall.yaml     公共训练设置
  data/                    数据位置，初始为空
  img/                     论文框架图
  log/                     新运行产物
  models/LGMRec.py          原始基线
  models/LGMRecopt.py       独立优化模型及内部算子
  utils/                   数据、配置、评估、产物身份
  main.py
  requirement.txt
  train.sh
  README.md
```

models/ 严格只有上述两个 Python 文件。保留全部历史实验；2026-10-20 清理的源码快照和验证产物放在 log/cleanup-audit-20261020-144353/，更早的验证产物仍在原仓库。

## 当前实现与阅读顺序

当前默认实现为 `lgmrec-opt-masked-neighbor-v1-group`，architecture revision 为 6。LGMRecOpt 在同一模型内衔接两个模块：输入端的遮蔽训练与邻居补全，以及原有的行为锚定超图推荐。`sparse_anchor_nohcl` 仍是实验类别标签，精确实现身份以运行摘要中的 `implementation_version` 为准。

模块一保持 LightGCN 协同编码 + 图文局部编码 + TRAIN 锚点群组超图 + 双侧行为残差；超图传播值仍为协同物品表示，原行为残差系数仍为 0.2。损失仅 BPR + 原有表示正则，不增加对比、重构或一致性损失。

模块二每个 epoch 为图像和文本独立采样整行缺失掩码，默认每模态概率 `modality_mask_rate=0.3`。补全使用模块一共享的 TRAIN 二值成员支持，仅在补全时排除 self；只对缺失行取该模态当前可观测邻居的原始特征算术均值，无可用邻居时保持零。观测行原值不变，补全发生在线性投影之前，补出的值不再递归作为 donor。训练掩码不读取 VALID/TEST 标签，也不改写磁盘原始特征。

模型按用户提供的 LGMRec 实现组织：初始化 → 构图与编码 → 输入补全 → `forward` → 损失 → 预测。

1. `__init__`、`get_norm_adj_mat`：读取配置，初始化参数与 TRAIN 图。
2. `cge`、`mge`、输入准备函数：完成协同/模态编码与每轮随机遮蔽、可观测邻居补全。
3. `forward`：局部融合 → 图文加权的锚点超图 → 原始物品 ID 的行为残差。图文特征决定超图权重，超图传播值来自协同物品表示。
4. `bpr_loss`、`reg_loss`、`calculate_loss`、`full_sort_predict`：训练目标与推荐评分；其后是 epoch 和缓存生命周期钩子。
5. 文件下半部分的构图函数、`MSCABehaviorView` 和 `SparseGroupHypergraph`：模型内部的图算子。

`utils/topk_evaluator.py` 负责全目录排名、缺失场景、覆盖率和分组指标；`common/trainer.py` 负责训练、VALID 选模和缺失验证曲线；`utils/quick_start.py` 在内存最佳权重上执行最终诊断并保存报告。模型文件不再承担报告导出、绘图或独立 checkpoint 命令入口。

当前只保留既定的 `group` 超图传播。历史 `identity` 传播消融入口已移除；仍设置 `LGMRECOPT_HYPER_PROPAGATION=identity` 会明确报错，应取消该环境变量。历史实验材料保留。

2026-10-20 的历史清理仅删除权重为 0 的物品对比损失计算、对应的原始损失日志及辅助返回值，收紧已退役的构图选项，并共用稀疏张量转换函数。当时版本为 `lgmrec-opt-cleanup-v1-group` 或 `lgmrec-opt-cleanup-v1-identity`。实验摘要明确记录 `item_contrast.enabled=false, computed=false, weight=0`；损失日志报告生效的 `loss_bpr` / `loss_reg`。

该次清理保留主体结构、共现筛选、图归一化、超图注意力、融合顺序和参数初始化顺序。`init_reference_width` 对应的两次初始化随机数消耗用于延续已有种子轨迹，应保留；它不产生可训练参数。数值验证与未验证事项见 [2026-10-20 清理记录](docs/20261020-144353-保留主体的冗余代码清理.md)。

2026-10-21 曾回退到第一项研究：当时模型与配置恢复为负采样实验接入前的 `lgmrec-opt-cleanup-v1`，默认版本为 `lgmrec-opt-cleanup-v1-group`。训练至今仍使用单个 TRAIN 均匀负样本。Hard / SR-SNS 的风险缓存、候选选择、采样统计、配置键以及数据加载器中的多候选分支均已删除；旧 `negative_*` 启动参数会明确报错。本次模态缺失模块不恢复这些采样机制。

当时停止依据是 seed999 的 Hard 正式运行退化，未据此宣称 SR-SNS 风险项已被正式实验证伪。历史日志和审计材料继续保留。回退范围、数值验证及服务器同步方法见 [回退记录](docs/20261021-085701-回退第一项研究并移除实验二采样模块.md)。

2026-10-23 的 v3 保留原行为残差，统一 TRAIN 二值构图口径，并加入同轮评估表示缓存。2026-10-26 回传的三种子 200 轮日志确认其 R20/N20 与旧同种子选定结果一致。这些是模块一完整输入训练的历史证据；当前遮蔽训练与邻居补全会改变训练输入，尚未完成同预算正式实验，不能沿用旧结果声称新模块有效。

## 历史实验依据

完整输入历史参考以已选定版本的最佳种子为主对照：v3、Baby、seed999、200轮，VALID Recall@20 为 **0.10349196145416252**，同一选中权重的 NDCG@20 为 **0.04465375366027156**，最佳 epoch=173（0起始）。来源为 [10月26日 v3 日志](log/LGMRecOpt-baby-seed999-Oct-26-2026-14-33-22-0266fa26.log)。该次未保存磁盘检查点，未评估 TEST；不能作为当前模块二的新结果。

以下保留早期实验记录，预算与版本各自独立：

按已有 Baby 同轮三种子 VALID Recall@20 主指标选择：

| 方案 | R20 | N20 | R50 |
|---|---:|---:|---:|
| 同轮 B | 0.101294596 | 0.043961313 | 0.164467459 |
| 锚点超图、无 HCL | 0.101834254 | 0.043907213 | 0.165960722 |

它是这组已完成实验中 R20 均值最好的方案，但 N20 略降，且 R20 一个种子下降。sports/clothing 尚无同等效果证据，不宣称稳定提升或 SOTA。

2026-10-15 新回传 B_history 的 R20/N20 在三个种子上均低于同轮 B，R20 均值 0.101120 vs 0.101296，因此不纳入实现。不将该轮 B 与锚点版跨轮比较。结果包和分析保留原目录。

## 安装与运行

Python >=3.10，使用合适的 PyTorch/CUDA 环境：

```bash
pip install -r requirement.txt
python main.py -m LGMRecOpt -d baby -g 0 --seed 999 --no-checkpoint
# 原始基线
python main.py -m LGMRec -d baby -g 0 --seed 999 --no-checkpoint
```

从 cf_Rec 根目录运行。数据放入 data/<dataset>/：<dataset>.inter、image_feat.npy、text_feat.npy。也可读取原数据：

```powershell
cd D:\project_sum\result_project\cf_Rec
python main.py -m LGMRecOpt -d baby --seed 999 --data-dir "D:\project_sum\result_project\lgmrec-opt-v1-complete\data"
```

每条 `python main.py` 命令只运行指定数据集和种子。现有 `train.sh` 会依次运行三个数据集各三个种子，不解析 `--seed`，不能作为上述单次命令的等价入口。

两个模型共享公共训练默认值：lr=0.002、batch=512、weight_decay=1e-5、最多1000轮、连续20次 VALID 无改善早停。本轮重构未改变这些默认值。固定200轮对照需要追加 `--set epochs=200 --set stopping_step=200`；分别运行种子999/2026/2027，保持数据指纹和训练设置一致。更多参数见 `python main.py --help`；旧 `--variant` 和算法切换键已删除。

### 模块二的最小对照

默认配置为 `modality_mask_rate=0.3`、`neighbor_imputation=true`、`missing_diagnostics=true`、`missing_eval_interval=5`、`missing_eval_seed=20261021`。所有组使用现有 LGMRecOpt，不新增模型变体文件。A/C 共用一次完整输入训练，由最终同一份内存最佳权重分别做 zero / neighbor 推断；B 训练时只遮蔽，D 训练时遮蔽并补全。

| 对照 | 训练输入 | 用于主对照的缺失推断 |
|---|---|---|
| A | 完整输入 | zero，缺失行置零 |
| B | 遮蔽，不补全 | zero |
| C | 与 A 同一次训练、同一权重 | neighbor，可观测 TRAIN 邻居均值 |
| D（默认） | 遮蔽并补全 | neighbor |

以下每条只运行一个种子，模型和数据集沿用默认的 LGMRecOpt/Baby。日志自动使用唯一名称，无需手填输出目录；正式对照将 seed 替换为999、2026、2027，并保持200轮预算：

```bash
# A/C：训练一次，最终同权重比较两种缺失推断
python main.py --seed 999 --set modality_mask_rate=0 --set epochs=200 --set stopping_step=200 --no-checkpoint
# B：仅遮蔽训练
python main.py --seed 999 --set neighbor_imputation=false --set epochs=200 --set stopping_step=200 --no-checkpoint
# D：遮蔽训练与邻居补全
python main.py --seed 999 --set epochs=200 --set stopping_step=200 --no-checkpoint
```

完整 VALID Recall@20 负责选模。每5轮用当轮权重做固定缺失诊断；训练结束恢复内存中按完整 VALID 选出的最佳状态，在同一权重上分别执行 zero / neighbor 两种推断。最终报告共13行：1个完整输入结果，以及图像缺失、文本缺失、图文均缺各20%/50%的6个场景乘2种推断方式。固定评估掩码与训练随机掩码分离；缺失曲线不替代完整 VALID 选模。

同时报告按模态的可用邻居覆盖率及分组推荐指标；分组统计仍在完整物品候选集上排序，不改成组内小候选集评估。最终诊断只保存一份 `results.json`，其中包含固定掩码的物品排列、13行结果、覆盖率、分组指标和各行同权重 zero/neighbor 差值；不再重复导出 CSV、NPY 或图片。最终诊断直接使用内存最佳状态，`--no-checkpoint` 不妨碍诊断；本流程不评估 TEST。当前尚未用三种子200轮证实模块二收益，完整输入与各缺失场景必须分别报告；本地实现不表示已经上传远程服务器。

默认每次运行生成一份 `log/LGMRecOpt-baby-seed999-时间-唯一标识.log`（名称随模型、数据集和种子变化），记录配置、数据指纹、运行环境、每轮损失与 VALID 指标、最佳结果及完成状态。每次缺失验证曲线只记一次；最终实验摘要只记录诊断报告位置，不重复写入整份结果。

模块二最终报告为唯一的 `log/missing-…/results.json`，不包含模型权重。默认不保存 checkpoint；`--no-checkpoint` 保留兼容，只有显式 `--save-checkpoint` 才保存模型。可选 `--run-dir` 指向尚不存在的目录，额外保存 `status.json`，训练日志仍在根 `log/` 下。

## 兼容与验证

当前模型使用 architecture revision 6，历史 v3 及更早检查点不能直接作为新实现权重 strict 加载，不手工补版本标记绕过兼容检查。优化模型不再保存旧 v_hyper/t_hyper 参数；历史审计和检查点继续保留。init_reference_width 仅消耗历史原型初始化对应的随机数，使保留权重的初始化与选定版本一致，不产生参数或超边，不是新调参项。

2026-10-26 按参考模型组织完成等价重构：模型文件由871行减为579行，相关四个代码文件合计净减少256行。CPU三种子×三组训练设置、GPU单组及完整入口检查通过；同条件初始化、梯度、更新及缺失评估结果与重构前一致。验证范围和已移除接口见[本次重构记录](docs/20261026-163428-改动说明.md)，不代表推荐效果提高。

历史独立实现的三种子微型数据对照曾通过：初始化、RNG、损失、梯度、两步带 weight decay 的 Adam 更新和推理结果与旧锚点候选精确一致。两模型当时均通过独立 CLI 单轮合成训练和检查点保存；验证文件在原目录 test_runtime/standalone-anchor-2e6569a8。这些历史检查不替代当前模块二的掩码、补全、缓存与完整入口验证；本轮实际验证范围见 [模块二改动记录](docs/20261026-154916-改动说明.md)。

仅 TRAIN 构图、负采样、学习；VALID 选模；训练 CLI 不开放 TEST。TEST 只在最终方案冻结后评估一次。当前模块二尚无正式200轮效果结论，未进行真实 TEST。

原始材料无可确认的根 LICENSE；整理不新增或改变上游授权，算法出处保留在代码。新目录运行不依赖原仓库 Python 包。

### 方法参考与采用范围

本次独立实现只参考问题定义和计算思路，没有复制以下仓库代码，也没有移植其训练器、额外损失或测试流程：

- [I³-MRec，ACM MM 2025](https://arxiv.org/abs/2508.04247)：参考训练与推断缺失状态的区分；核查版本为 [commit 82a71c2 的 model.py](https://github.com/HuilinChenJN/I3-MRec/blob/82a71c2d6b5aceba628d8c437a91402c65500a46/model.py#L292)。其 TEST 掩码生成及训练中使用 TEST 的流程不适用于本项目。
- [DGMRec，SIGIR 2025](https://arxiv.org/abs/2504.16352)：参考观测集合和缺失集合的显式区分、按可观测目标筛选监督；核查版本为 [sigir25 commit 992196e](https://github.com/ptkjw1997/DGMRec/blob/992196e68eb2614f3198868d19113568bc51f45b/src/models/dgmrec.py)。本项目没有采用其生成器、动态图或多项损失；其按 TEST 选择超参的流程不采用。
- [Training-Free Graph-Based Imputation，TKDE 2026](https://arxiv.org/abs/2602.17354)：参考 TRAIN 图上的特征补全，核查版本为 [commit 1d76070 的 impute.py](https://github.com/danielemalitesta/Graph-Missing-Modalities-TKDE/blob/1d7607027f6b52ca48281aec67fb542f96a6978e/impute.py)。本实现进一步明确仅使用当前可观测非自身邻居、只填缺失行、不递归传播，不直接使用其稠密构图和完整训练评估管线。

公开代码不等于已有本项目的复制授权或效果证明。DGMRec 与 Graph-Missing-Modalities-TKDE 的核查版本未发现明确根许可证；本次按上述方法说明在现有 LGMRecOpt 内自行实现，保留来源和差异。

## 部署审查与变更记录

服务器准备、数据指纹、环境及复现命令见 [部署审查](docs/20261015-163500-服务器部署审查.md)。后续每次改动必须在 docs/ 新建“时间-改动说明.md”，规则见 AGENTS.md。

日志布局以当前说明为准：早期部署文档中的 log/runs/ 与默认 checkpoint 描述已由平铺日志改动取代。
