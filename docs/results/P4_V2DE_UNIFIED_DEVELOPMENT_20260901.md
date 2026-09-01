# P4 v2d/v2e 统一种子重训与完整实验（已落实，2026-09-01）

## 结论

已按统一开发协议重训 v2d、v2e，并在与 v2c 相同的
`556000..556004` 场景上完成 clean、Random、FGSM、PGD、MAD、legacy v2c、
重训 v2d、重训 v2e 八条件比较。输出已通过文件哈希、数据集绑定、critic
绑定、summary 重算和表格重算验证。

重训 v2e 在 5/5 seeds 上降低 discounted return，mean ΔG 为 `0.2560`、median
ΔG 为 `0.1848`，且最大正质量占比为 `0.4859`，通过冻结的
signed-return effect gate。它证明统一重训后的 v2e 能稳定运行并产生回报攻击效果，
但仍未通过 strong-baseline envelope：mean/median 均弱于 MAD，mean 也弱于 FGSM，
因此不能声称 v2e 优于强基线。

重训 v2d 没有改善：mean ΔG 为 `-1.1110`，并在 seed `556003` 上产生
`-5.7498` 的强反向效果。当前正部 critic 的尺度和动作排序诊断较差，不适合继续作为
主要攻击方向。

## 实验身份与范围

- 环境：`RL_Attack_Core_Py310`（项目 `.venv`，CPU，BLAS/Torch 单线程）；
- 配置：`configs/experiments/p4_mergelite9_v2de_unified_development.yaml`；
- 输出：`outputs/p4_v2de_unified_development_20260901`；
- manifest SHA-256：
  `a883275fedf4920cc48e91d6a45ae37033ecf19f7bbbca3e54247214098c67aa`；
- victim：同一冻结 MergeLite9 PPO；
- 威胁：PPO observation-only、ratio `6.0`、有效逐维 ε 上限 `0.3`；
- solver：FGSM 原生单步；PGD、MAD、v2c、v2d、v2e 使用 `20×5`；
- schedule：由重训 v2e signed-return critic 在 clean trajectory 上统一选取 top-2，
  所有攻击条件使用相同 timing 和 STFA restart plan；
- v2c：保留 legacy 算法，不重训，但在本轮公共 schedule 上重新执行；
- 公共 counterfactual collection：129 rows；
- train episodes：`556000..556003`；validation episode：`556004`；
- comparison episodes：`556000..556004`。

这 5 个场景同时参与 critic 数据采集、开发拆分和效果比较，属于有意的数据复用。
因此本结果只可作为 development/in-sample 工程筛选，不是独立 hold-out、统计显著性
或 formal 泛化证据。配置和 manifest 中所有 effectiveness、superiority、formal、SUMO
claims 均保持 false。

## 完整聚合表

ΔG 定义为同 seed 的 `G_clean - G_attack`；正值表示攻击降低 PPO discounted return。

| 方法 | mean G | mean ΔG | median ΔG | 正下降 seeds | 最大正质量占比 | flip rate | merge fail | safety cost | native grad queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 8.2492 | 0.0000 | 0.0000 | 0/5 | — | — | 0.20 | 1.6801 | 0 |
| Random | 8.0318 | 0.2174 | 0.0779 | 3/5 | 0.8261 | 0.70 | 0.20 | 2.0447 | 0 |
| FGSM | 7.9449 | 0.3043 | 0.1139 | 5/5 | 0.7007 | 1.00 | 0.20 | 2.1891 | 10 |
| PGD-20×5 | 9.2808 | -1.0316 | 0.1128 | 4/5 | 0.3780 | 1.00 | 0.00 | 4.4019 | 1000 |
| MAD-20×5 | 7.9375 | 0.3117 | 0.2236 | 5/5 | 0.5726 | 1.00 | 0.20 | 2.2061 | 1000 |
| v2c legacy | 9.7547 | -1.5055 | 0.1784 | 4/5 | 0.3278 | 0.90 | 0.00 | 3.6672 | 1000 |
| v2d unified retrain | 9.3602 | -1.1110 | 0.0027 | 3/5 | 0.8972 | 0.90 | 0.00 | 4.2519 | 1000 |
| **v2e unified retrain** | **7.9931** | **0.2560** | **0.1848** | **5/5** | **0.4859** | **1.00** | **0.20** | **2.0398** | **1000** |

## 逐 seed ΔG

| seed | Random | FGSM | PGD | MAD | v2c legacy | v2d retrain | v2e retrain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 556000 | -0.0034 | 0.0859 | 0.0767 | 0.0904 | 0.0859 | 0.2044 | **0.0904** |
| 556001 | 0.0779 | 0.1784 | 0.1784 | 0.2394 | 0.1784 | 0.0027 | **0.1594** |
| 556002 | 0.0000 | 0.0771 | 0.2236 | 0.2236 | 0.2236 | -0.0329 | **0.2236** |
| 556003 | 0.9007 | 1.0662 | -5.7498 | 0.8924 | -8.2096 | -5.7498 | **0.6220** |
| 556004 | 0.1117 | 0.1139 | 0.1128 | 0.1128 | 0.1941 | 0.0207 | **0.1848** |

seed `556003` 对 PGD、v2c 和 v2d 的均值造成显著反向影响。这不是统计噪声应被隐藏，
而是说明在该动力学状态上，单步动作翻转可能改善后续轨迹；仅有 action flip 并不等于
return attack 成功。v2e 在该 seed 仍保持正 ΔG，但不及 FGSM、MAD 和 Random。

## v2e 相对各方法

advantage 为 `ΔG_v2e - ΔG_comparator`，正值表示 v2e 更强。

| comparator | mean advantage | median advantage | v2e 胜-平-负 |
|---|---:|---:|---:|
| Random | 0.0387 | 0.0815 | 4-0-1 |
| FGSM | -0.0483 | 0.0044 | 3-0-2 |
| PGD | 1.2877 | 0.0137 | 3-1-1 |
| MAD | -0.0557 | 0.0000 | 1-2-2 |
| v2c legacy | 1.7616 | 0.0000 | 2-1-2 |
| v2d retrain | 1.3670 | 0.1641 | 4-0-1 |

v2e 对 FGSM 在 3/5 seeds 配对占优且 median advantage 略正，但 seed `556003` 使 mean
advantage 为负。对 MAD 则没有优势。预注册的 oracle comparator envelope 检查未通过，
所以本轮不支持 superiority 结论。

## critic 重训诊断

### v2d

- train/validation rows：102/27；
- final train/validation MAE：`0.5548 / 0.5634`；
- validation action argmax accuracy：`0.0000`；
- validation mean true opportunity：`0.00515`；
- validation mean predicted opportunity：`0.62024`；
- validation opportunity MAE：`0.61509`。

v2d critic 将机会量级高估约两个数量级，且动作排序失败。v2d 的反向行为与 critic
诊断一致，不应通过扩大 seeds 掩盖。

### v2e

- train/validation rows：102/27；
- final train/validation MAE：`0.01251 / 0.00422`；
- validation near-optimal top-1：`1.0000`；
- validation pairwise concordance：`0.9363`；
- validation opportunity NMAE：`2.2971`，高于门限 `0.75`；
- heldout/runtime-eligible rows：`27/27`，低于正式门限 `300/200`；
- validation top-1 baseline advantage：`0.0000`，低于门限 `0.05`；
- critic adequacy：false。

v2e 的动作排序和输入梯度路径可用，但机会幅度校准仍不合格，且单 validation episode
样本量不足。因此行为 effect gate 通过不等于 critic preparation gate 通过。

## 执行完整性与代价

- 10/10 scheduled timing steps reachable、selected、nonzero；
- 10/10 runtime targets 为合法、非 clean 且 predicted signed loss 严格为正；
- clean probe target 与 live runtime target 匹配 8/10；
- v2e action flip rate：1.0；
- structural integrity、runtime target contract、query ledger closure 全部通过；
- shared schedule physical queries：387；
- native execution queries：15750；
- 实验 physical queries：16137；
- 逐条件 schedule 的 2709 queries 仅作逻辑归因，没有重复计入 physical total。

## 对下一步的约束

1. 保留 v2e，停止继续优化当前 v2d 正部 critic；
2. v2e 下一轮优先增加公共训练场景并修复 opportunity calibration，而不是增加 solver
   迭代；
3. 继续用固定 development seeds 做调参，但任何正式结论必须换到未参与训练和选择的
   hold-out seeds；
4. FGSM 只使用 10 次 native gradient queries，mean ΔG 仍高于 v2e；后续 v2e 必须同时
   报告攻击效果和查询效率；
5. P5 仍只保留接口、预算、manifest 和 adaptive smoke，不应基于本轮结果启动正式防御
   有效性结论。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2de_unified_development run `
  configs/experiments/p4_mergelite9_v2de_unified_development.yaml `
  --output-dir outputs/p4_v2de_unified_development_20260901

.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2de_unified_development verify `
  configs/experiments/p4_mergelite9_v2de_unified_development.yaml `
  --run outputs/p4_v2de_unified_development_20260901 `
  --expected-manifest-sha256 a883275fedf4920cc48e91d6a45ae37033ecf19f7bbbca3e54247214098c67aa
```
