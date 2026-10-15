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
  img/                     官方布局占位
  log/                     新运行产物
  models/LGMRec.py          原始基线
  models/LGMRecopt.py       独立优化模型及内部算子
  utils/                   数据、配置、评估、产物身份
  main.py
  requirement.txt
  train.sh
  README.md
```

models/ 严格只有上述两个 Python 文件。历史实验、备份、测试、虚拟环境均留在原仓库。

## 唯一优化方案

LGMRecOpt 固定为 `sparse_anchor_nohcl`：LightGCN 协同编码 + 图文局部编码 + TRAIN 锚点群组超图 + 双侧行为残差。损失仅 BPR + 正则。无原型分配、Gumbel、HCL、history、id_cond 或 C0 切换。

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

三种子微型数据对照通过：初始化、RNG、损失、梯度、两步带 weight decay 的 Adam 更新和推理结果与旧锚点候选精确一致。两模型均通过独立 CLI 单轮合成训练和检查点保存；验证文件在原目录 test_runtime/standalone-anchor-2e6569a8。

仅 TRAIN 构图、负采样、学习；VALID 选模；训练 CLI 不开放 TEST。TEST 只在最终冻结后评估一次。本次未运行新的完整实验或真实 TEST。

原始材料无可确认的根 LICENSE；整理不新增或改变上游授权，算法出处保留在代码。新目录运行不依赖原仓库 Python 包。

## 部署审查与变更记录

服务器准备、数据指纹、环境及复现命令见 [部署审查](docs/20261015-163500-服务器部署审查.md)。后续每次改动必须在 docs/ 新建“时间-改动说明.md”，规则见 AGENTS.md。

日志布局以当前说明为准：早期部署文档中的 log/runs/ 与默认 checkpoint 描述已由平铺日志改动取代。
