# cf_Rec — LGMRec / LGMRecOpt

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

models/ 严格只有上述两个 Python 文件。当前发布为 `lgmrec-opt-train-canonical-v3-group`。历史实验日志、数据与检查点保留在本地，不随源码发布。

## 当前实现与阅读顺序

LGMRecOpt 固定为 `sparse_anchor_nohcl`：LightGCN 协同编码 + 图文局部编码 + TRAIN 锚点群组超图 + 双侧行为残差。损失仅 BPR + 正则。无原型分配、Gumbel、HCL、history、id_cond 或 C0 切换。

`models/LGMRecopt.py` 按以下顺序阅读：

1. `cge` / `mge`：协同表示和两路图文局部表示。
2. `forward`：局部融合 → 图文加权的锚点超图 → 原始物品 ID 的行为残差。图文特征决定超图权重，超图传播值来自协同物品表示；行为分支不新增可训练参数。
3. `calculate_loss`：单个 TRAIN 均匀负样本对应的 BPR + 原有表示正则。
4. 文件下半部分的构图函数、`MSCABehaviorView` 和 `SparseGroupHypergraph`：上述模型使用的内部算子，不是其他模型。

默认执行 `group`。环境变量 `LGMRECOPT_HYPER_PROPAGATION=identity` 仅用于已有传播诊断，保留同一主体的其他计算和行为残差；不能把这个对照当成关闭行为残差。两个模式使用相同 TRAIN 图和参数集合。

当前版本恢复原有效行为残差，`behavior_eta=0.2`；所有图和用户读出共用去重、二值化的 TRAIN 交互，分别保留各算子的原归一化。超图消息值仍为 CGE 物品表示。没有新增残差、门控或训练目标，物品对比损失关闭且不再计算。

完整输入评估使用单次 eval 内的表示缓存；重新 eval/train、设备或精度迁移、加载权重与训练损失计算会清空缓存。手动修改参数、图或融合系数后应重新调用 eval()。`init_reference_width` 只消耗历史初始化随机数以保留种子轨迹，不产生可训练参数。

第二项研究目前是模型文件内的推理模态缺失诊断：固定检查点，对 VALID 进行缺图、缺文本或图文均缺的输入干预。它不参与训练和选模，不能表述为已实现缺失感知训练。训练继续使用单个 TRAIN 均匀负样本。

发布范围、源码哈希、验证与限制见 [本次发布记录](docs/20261023-180404-改动说明.md)。

## 当前 v3 已完成的 200 轮结果

2026-10-26 的 Baby seed999 完整运行已完成：最佳 epoch=173（从0起），完整 VALID Recall@10=0.064168044、Recall@20=0.103491961、NDCG@20=0.044653754。这是目前按完整 VALID Recall@20 选取的最佳版本；未评估 TEST，未保存 checkpoint。本仓库保留该 v3，不包含后续第二模块训练及 β 可靠性权重实验。

复现命令（已有相同数据和环境）：

```bash
python main.py -m LGMRecOpt -d baby -g 0 --seed 999 --no-checkpoint --set epochs=200 --set stopping_step=200
```

源码核验、输入指纹、运行环境及限制见 [发布核验记录](docs/20261028-122421-改动说明.md)。

## 历史最好结果：Baby，固定 200 轮

以下为历史 `lgmrec-opt-hyper-contribution-ab-v1-group` 无 CL 实验，种子 999/2026/2027；epochs=200、stopping_step=200、eval_step=1，按 VALID Recall@20 选取同一轮的全部指标。它们是保留当前主体结构的历史依据，**不是本次上传重新跑出的成绩**，也不是 TEST 结果。

| seed | 最佳 epoch（从 0 起） | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 999 | 173 | 0.064168044 | 0.103491961 | 0.166504243 | 0.034639235 | 0.044653754 | 0.057225606 |
| 2026 | 101 | 0.065504292 | 0.103030342 | 0.167027374 | 0.035010302 | 0.044577901 | 0.057380594 |
| 2027 | 154 | 0.065006367 | 0.102121797 | 0.167468890 | 0.035014974 | 0.044452126 | 0.057492412 |
| 均值 | — | 0.064892901 | **0.102881367** | 0.167000169 | 0.034888170 | **0.044561260** | 0.057366204 |

本次上传的是恢复该主体结构并保留构图修复的 v3 源码，没有附带历史最佳 200 轮权重；该组原日志未保存检查点。当前 v3 的真实 Baby seed999 20 轮验证为 Recall@20=0.098277621、NDCG@20=0.042661066，最佳 epoch=18；严格重载检查点后 VALID 指标一致。后续 v3 seed999 已完成200轮，见上节；v3 尚未完成200轮三种子，Sports/Clothing 也没有同等完整效果证据。

## 更早的独立实验记录

下表保留较早一轮 Baby 三种子结果；其训练预算与上面的固定 200 轮不同，不能跨表直接比较提升幅度。该轮按 VALID Recall@20 选择：

| 方案 | R20 | N20 | R50 |
|---|---:|---:|---:|
| 同轮 B | 0.101294596 | 0.043961313 | 0.164467459 |
| 锚点超图、无 HCL | 0.101834254 | 0.043907213 | 0.165960722 |

它是这组已完成实验中 R20 均值最好的方案，但 N20 略降，且 R20 一个种子下降。sports/clothing 尚无同等效果证据，不宣称稳定提升或 SOTA。

2026-10-15 新回传 B_history 的 R20/N20 在三个种子上均低于同轮 B，R20 均值 0.101120 vs 0.101296，因此不纳入实现。不将该轮 B 与锚点版跨轮比较。结果包和分析保留原目录。

## 安装与运行

当前发布的已验证环境为 Python 3.12.7、PyTorch 2.6.0+cu124、NumPy 1.26.4、SciPy 1.13.1。请使用对应的 PyTorch 2.6.0 环境及适配硬件的构建。

已知限制：`requirement.txt` 的历史宽范围仍允许 torch 2.2，但当前缓存调用的 `torch.is_autocast_enabled(device_type)` 不兼容 2.2 的无参数接口；因此不能仅满足该依赖下限就视为可运行。本次按原样发布模型，未修改依赖或该接口。

```bash
pip install -r requirement.txt
python main.py -m LGMRecOpt -d baby --seed 999
# Bash 等价入口
bash train.sh --seed 999
# 原始基线
python main.py -m LGMRec -d baby --seed 999
```

从 cf_Rec 根目录运行。数据放入 data/<dataset>/：<dataset>.inter、image_feat.npy、text_feat.npy。也可读取原数据：

```powershell
cd D:\project_sum\result_project\cf_Rec
python main.py -m LGMRecOpt -d baby --seed 999 --data-dir "D:\project_sum\result_project\lgmrec-opt-v1-complete\data"
```

两个模型共享公共训练默认值：lr=0.002、batch=512、weight_decay=1e-5。正式对照分别运行种子 999/2026/2027，固定数据指纹和训练设置。更多参数见 python main.py --help；旧 --variant 和算法切换键已删除。

每次运行直接生成 log/LGMRecOpt-baby-seed999-时间-唯一标识.log（名称随模型、数据集和种子变化），采用时间 + INFO 的文本格式，记录配置、数据指纹、运行环境、每轮损失与 VALID 指标、最佳结果及完成状态。默认不保存 checkpoint、不创建实验子目录或 status.json；--no-checkpoint 保留兼容。仅显式 --save-checkpoint 时保存模型。可选 --run-dir 指向尚不存在的目录，额外保存 status.json，但训练日志仍在根 log/ 下。

## 兼容与验证

优化模型不再保存旧 v_hyper/t_hyper 参数，旧检查点不能直接 strict 加载；历史审计留在原仓库。init_reference_width 仅消耗历史原型初始化对应的随机数，使保留权重和后续 RNG 与选定版本一致，不产生参数或超边，不是新调参项。

历史独立实现审计中，三种子微型数据对照通过：初始化、RNG、损失、梯度、两步带 weight decay 的 Adam 更新和推理结果与旧锚点候选精确一致。两模型均通过独立 CLI 单轮合成训练和检查点保存；验证文件在原目录 test_runtime/standalone-anchor-2e6569a8。

仅 TRAIN 构图、负采样、学习；VALID 选模；训练 CLI 不开放 TEST。TEST 只在最终冻结后评估一次。本次上传只做只读核验，没有重新训练或评估 TEST。当前 v3 的验证范围及既有 20 轮结果见上述发布记录。

原始材料无可确认的根 LICENSE；整理不新增或改变上游授权，算法出处保留在代码。新目录运行不依赖原仓库 Python 包。

## 部署审查与变更记录

服务器准备、数据指纹、环境及复现命令见 [部署审查](docs/20261015-163500-服务器部署审查.md)。后续每次改动必须在 docs/ 新建“时间-改动说明.md”，规则见 AGENTS.md。

日志布局以当前说明为准：早期部署文档中的 log/runs/ 与默认 checkpoint 描述已由平铺日志改动取代。
