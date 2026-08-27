# P4 v2c top-2 matched engineering：已落实运行结果（2026-08-26）

本记录对应 MergeLite9 单一冻结 PPO victim 的 P4 v2c 小规模工程筛选。v2c 保留
v2b 的多步风险目标与内层攻击器，只修复未触发攻击的时机选择契约：B3 从绝对阈值门
改为 episode 内排序器，删除 `p >= 0.5` 和 `opportunity >= 0.05` 两个硬门，并在
`K=8 / min_gap=2 / window=16 / window_k=2` 预算下选择 clean trajectory 的 top-2
可行时刻。

本结果用于确认攻击真正运行并进行同 seed、同 schedule 的小规模比较；不是正式统计
结论、不是 SUMO 证据，也不能证明 STFA 优于基线。

## 最终有效运行与独立校验

- 源码提交：`cf194e36dc78f1d99d9708b639fa1bb52d9f347a`
- 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`），CPU，Torch / interop
  threads 为 1 / 1
- 配置：`configs/experiments/p4_mergelite9_v2c_matched_engineering.yaml`
- calibration seeds：20 个，`554000..554019`
- engineering seeds：5 个，`556000..556004`
- bundle：`outputs/p4_v2c_engineering_cf194e3_20260826`
- run manifest SHA-256：
  `999ecadcb6229cee22cab9f9f0d4fba0e8caedfcbc89d50bb72841d42852b637`
- preparation manifest SHA-256：
  `f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0`
- 独立 verify：`status=verified`

Verifier 重新核验了 artifact 哈希、preparation、victim 绑定、top-2 schedule 和完整
攻击矩阵的确定性重放。victim policy state 在运行前后保持一致。

## 时机门修复结果

20 个 calibration episodes 共 540 个 clean 时刻，其中 477 个具有正的 predicted
opportunity。每个 episode 都生成了恰好 2 个满足时序预算的攻击时刻，总计 40 个。
calibration 只使用 clean prediction 与 schedule 信息，没有记录或使用 return、safety、
collision、merge outcome，也没有打开离线数据集。

5 个 engineering seeds 中，四个攻击条件均执行 10 个选定时刻并产生 10 个非零扰动。
FGSM、PGD 和 MAD 各造成 10/10 次动作翻转；STFA 造成 8/10 次动作翻转。因此，v2b
“solver 从未运行”的直接故障已在 v2c 中修复。

## 五种条件的小规模结果

| 条件 | mean return | return drop vs clean | mean safety cost | safety delta | merge failure | collision | action flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 10.0222 | 0.0000 | 1.6801 | 0.0000 | 0.20 | 0.00 | 0 |
| FGSM | 9.5551 | 0.4671 | 2.2786 | +0.5985 | 0.20 | 0.00 | 10/10 |
| PGD 20x5 | 6.0940 | 3.9283 | 5.1222 | +3.4421 | 0.20 | 0.20 | 10/10 |
| MAD 20x5 | 9.7704 | 0.2518 | 2.0819 | +0.4018 | 0.20 | 0.00 | 10/10 |
| STFA v2c 20x5 | 12.0676 | -2.0453 | 3.6072 | +1.9271 | 0.00 | 0.00 | 8/10 |

所有攻击共享同一冻结 victim、5 个 episode seeds、clean-derived top-2 schedule 和
ratio=6 projector。该比较是 **schedule-matched**，不是 target-matched，也不是
query-matched。FGSM 每个选定时刻只做一次梯度；PGD、MAD 与 STFA 使用 20x5，因此
不能把数值差异解释为同等计算成本下的优越性。

### STFA 逐 seed 结果

| seed | clean return | STFA return | return drop | clean safety | STFA safety | safety delta | flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 556000 | 14.9508 | 15.0203 | -0.0694 | 0.1500 | 0.3000 | +0.1500 | 2/2 |
| 556001 | 14.4892 | 14.2267 | +0.2625 | 0.0000 | 0.3000 | +0.3000 | 2/2 |
| 556002 | 15.2904 | 15.1479 | +0.1425 | 0.0000 | 0.1500 | +0.1500 | 2/2 |
| 556003 | -9.4276 | 1.2652 | -10.6928 | 8.2506 | 17.1362 | +8.8856 | 1/2 |
| 556004 | 14.8083 | 14.6777 | +0.1306 | 0.0000 | 0.1500 | +0.1500 | 1/2 |

STFA 在 3/5 seeds 上产生小幅正 return drop，但 seed `556003` 上 clean 本身已经 merge
failure，STFA 提高了 episode return、同时显著增加 safety cost。这个单 seed 使 STFA
平均 return 高于 clean，但平均 safety cost 仍明显恶化，说明当前 composite-risk 目标
与“稳定降低 PPO 奖励”并未闭合，奖励与安全风险两个目标存在冲突。

PGD 在 seed `556003` 触发一次 collision，造成约 19.29 的 return drop；它主导了 PGD
的均值。只有 5 个工程 seeds，不能据此声称 PGD 的总体效果或统计优势。

## 阶段结论

1. 不回档 P4 v2a。v2a 保留为历史机制参照，v2b 保留为“绝对机会门导致零攻击”的失败
   证据；v2c 在不改写历史 artifact 的前提下修复 selector。
2. v2c 已达到本轮工程目标：top-2 schedule 饱和、四个攻击器都实际执行、扰动非零且
   预算合规，并在至少一个 seed 上观察到 outcome 变化。
3. v2c 尚未解决“观测攻击稳定降低 Vanilla/episode return”的科学目标，也没有证明
   STFA 优于 FGSM、PGD 或 MAD。当前 STFA 更像 safety-risk attack，而不是稳定的
   reward-degradation attack。
4. 下一版攻击目标应把预测的多步 return loss / failure probability 设为主目标，将
   safety cost 作为独立报告维度或约束；director 应按短反事实 rollout 的 expected
   return drop 排序，而不是只按 raw composite risk 排序。新目标冻结后再使用新的
   seeds 做工程筛选。
5. matched seeds `557000..557049` 与 future-final seeds `558000..558049` 均未消费；
   所有 formal、effectiveness、superiority、statistical、SUMO 与 Vanilla claim flags
   保持 `false`。

P5 当前也只完成 adaptive-attack 工程 smoke。它证明接口、BPDA、Guard runtime、账本和
manifest/replay 闭环，不证明防御提高了奖励或安全性；正式 P5 效果比较应等待攻击目标与
outcome 指标闭合后再启动。
