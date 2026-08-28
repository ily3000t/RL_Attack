# P4 v2d 纯回报短反事实攻击：5-seed 工程筛选已完成

## 结论

P4 v2d 已完成 formal preparation、完整 collection replay、5-seed engineering run 和
完整矩阵 replay。攻击执行与回报目标闭合通过，但相对旧 v2c composite comparator 的
配对中位优势未通过，因此 **不得扩大到 matched/final seeds，也不得声称优于基线**。

- 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`）；
- victim：同一 MergeLite9 9-action Vanilla SB3 PPO；
- 威胁：PPO observation-only，MergeLite9 sensor-v2 projector，ratio 6；
- 时机：每 seed 两个由 clean episode 的 return-opportunity selector 冻结的时机；
- engineering seeds：`559000..559004`；
- preparation manifest：`6ba2f1202140c0681d598506769e77dc6c37d6b893c3be50a5e1432fa8fe4eaa`；
- run manifest：`1cb6babac0ed0f7ff008613d5785266d01083b47bd30682dc92381a93ee2d03e`。

## 方法对比

主指标是与同 seed clean episode 配对的 signed discounted-return drop；正值表示攻击降低
victim 回报。表中 query 为 5 seeds 合计，包含同一冻结 selector 的逻辑查询。

| 条件 | 均值回报下降 | 中位回报下降 | 正值 seeds | episode-return 下降 | safety cost 增量 | 动作翻转 | gradient queries | total queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FGSM fixed | 0.097685 | 0.011825 | 4/5 | 0.097925 | 0.165382 | 10/10 | 10 | 416 |
| PGD 20×5 | 0.070339 | 0.013258 | 4/5 | 0.089417 | 0.132845 | 10/10 | 1000 | 3496 |
| MAD 20×5 | -0.021023 | 0.011825 | 3/5 | -0.022396 | 0.072845 | 10/10 | 1000 | 3496 |
| legacy v2c composite | 0.071100 | 0.035812 | 4/5 | 0.062148 | 0.152214 | 8/10 | 1000 | 3506 |
| **v2d return-only STFA** | **0.034075** | **0.017632** | **4/5** | **0.042173** | **0.093142** | **10/10** | **1000** | **3506** |

clean 的 mean discounted return 为 8.842194、mean episode return 为 10.614559、mean
safety cost 为 0.539179。所有条件的 merge-failure rate 均为 0.2、collision rate 均为
0，因此本轮没有观测到攻击改变这两个二值端点。

## v2d 与 legacy 的逐 seed 配对

| seed | v2d drop | legacy drop | v2d − legacy |
|---:|---:|---:|---:|
| 559000 | 0.300661 | 0.035812 | 0.264849 |
| 559001 | 0.011825 | 0.010115 | 0.001710 |
| 559002 | 0.028765 | 0.131502 | -0.102737 |
| 559003 | -0.188505 | -0.072768 | -0.115737 |
| 559004 | 0.017632 | 0.250838 | -0.233206 |

配对优势中位数为 -0.102737，仅 2/5 seeds 为正。v2d 的正回报下降有约 83.8% 来自
seed 559000；leave-one-out mean 最低为 -0.032571，说明当前均值闭合对单 seed 较敏感。

## Gate 与证据

| Gate | 结果 |
|---|---|
| 每个 condition × seed 至少一个可达攻击 | 通过 |
| 所有可达攻击执行且扰动非零 | 通过 |
| 回报下降均值 > 0、中位数 > 0、至少 3/5 为正 | 通过 |
| v2d − legacy 配对中位数 > 0 | **未通过** |
| scale-up gate | **未通过** |

artifact integrity、victim binding、shared restart plan 和 deterministic full-matrix
replay 均已验证。所有 formal/effectiveness/superiority/statistical/SUMO/Vanilla/causal
director claims 保持 `false`。

## 为什么当前 v2d 未形成优势

1. return critic 的 validation action-argmax accuracy 仅 28.2%；validation opportunity
   MAE 为 0.025764，而 mean target opportunity 仅 0.012833。它能提供梯度，但对“哪个
   动作最伤害回报”的排序仍弱。
2. 当前标签是 `E[(G_clean-G_a)_+/25]`。positive-part 会把提高回报的动作截为零，不能
   学到 signed action ordering；训练集中 positive target opportunity rate 约 99.4%，
   对时机区分也偏弱。
3. inner objective 优化 categorical expected loss，而 victim 执行 deterministic
   argmax。10/10 动作均翻转不等于翻到预测的最坏动作，surrogate/execution mismatch
   仍然存在。
4. FGSM 仅用 10 次梯度查询就得到本轮最高均值下降；PGD/MAD/STFA 使用 1000 次梯度
   查询，当前额外计算没有转化为稳定收益。

## 冻结决策与下一步

本轮 `559000..559004` 已消费，只能用于报告和机制诊断，不能用于调参。`559300..559349`
和 `559400..559449` 保持未消费。当前 v2d 不扩种子；下一版本应先冻结为独立 v2e：

1. 用 paired signed return difference `(G_clean-G_a)/25` 训练 action-wise critic，并加入
   pairwise action-ranking loss；failure/safety 仍保持 report-only；
2. 将 inner solver 与 deterministic argmax 对齐：选择预测 signed loss 最大且为正的
   action，使用 target-logit margin + signed return auxiliary objective，而不是仅优化
   categorical expectation；
3. selector 使用“预测回报损失 × 可攻击到目标动作的 margin feasibility”，并保留完整
   非因果工程声明；
4. 预先增加 critic action-ranking、leave-one-out mean 和 positive-mass concentration
   gate；目标冻结后使用全新 development seeds（候选 `559010..559014`）做一次性筛选；
5. 只有同时优于 matched FGSM、PGD 和 legacy comparator 后，才进入保留 seeds。
