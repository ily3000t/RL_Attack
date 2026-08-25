# P4 v2b preparation/verify：已落实运行结果（2026-08-25）

本记录对应 MergeLite9 单一冻结 PPO victim 的 P4 v2b 开发准备阶段。
它不是攻击效果结论、不是 SUMO 证据，也不消费 validation、matched 或
future-final cohort。

## 最终有效运行

- 源码提交：`7d0b72fdeded926cc7e25e3cedc13bb7bba2743e`
- 环境：`RL_Attack_Core_Py310`，Python 3.10.16，Torch 2.0.0+cpu
- Torch / interop threads：1 / 1
- bundle：`outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825`
- preparation manifest SHA-256：
  `f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0`
- preparation contract SHA-256：
  `2d47740862f41523ec2d9f4f86472cecc7fe990ae028a473fca739789c7b93a1`
- verified bundle SHA-256：
  `5dde9d8adc32d1a8a499fdcb245e6cf0d1ab0720110ff6e8e5ec748ba53b7701`
- verify 状态：`verified`

## 数据与训练

| 项目 | 结果 |
| --- | ---: |
| critic collection | 200 episodes / 5,252 rows |
| director collection | 200 episodes / 5,248 rows |
| 每个状态的 oracle 标签 | 全 9 个首动作 |
| critic train / validation loss | 0.00404789 / 0.00200231 |
| director train / validation weighted BCE | 0.84199339 / 0.92544776 |
| director train positives / negatives | 122 / 4,077 |
| director validation positives / negatives | 31 / 1,018 |
| victim policy unchanged | true |

风险契约固定为 horizon 64、discount 0.99、CRN replicate 1、return/safety
scale 25/10，以及 return drop、merge failure、cumulative safety cost 三项等权。
观测攻击预算固定为 ratio 6，对应 normalized mutable features 的有效上限 0.3。

## 可重复性检查

修复 verifier 前后的两次完整 preparation 使用相同采集和训练科学配置：

- critic dataset SHA-256 均为
  `65f0c294a580259cc144bd1d53189aaf80c788778501659d372bd54fb159963d`；
- critic state SHA-256 均为
  `28cf867b89f607b9b12a2d4fa413e49d9440ba5991b1451bcda4ecd664be4698`；
- critic train/validation loss 逐值一致。

Torch checkpoint 容器字节因 bundle 内绝对 victim 路径不同而不同；模型 state
和训练指标一致，因此没有把容器 SHA 差异误判为训练不确定性。

## 运行中发现并修复的问题

提交 `b6304b3` 的第一次完整 prepare 成功，但真实 verify 检出 dataset binding
拥有 `schema_version`、critic artifact binding 不拥有该字段，而旧比较错误地把它
当作共同字段索引，触发 `KeyError`。该 bundle 未被接受为有效证据。

提交 `7d0b72f` 将 schema authority 单独验证，其余 dataset/critic 科学摘要逐项
交叉绑定，并增加正确绑定、错误 schema、错误 digest 回归测试。随后重新执行
200+200 preparation，独立 verify 全部通过。

## 种子与证据边界

- 已消费：critic `548000..548199`、director `549000..549199`；
- 未消费：validation `550000..550049`；
- 未消费：matched `551000..551049`；
- 未消费且无配置：future-final `552000..552049`。

B5 verified handoff 仅导出 victim、critic/director checkpoint 与 sidecar、runtime
contract/evidence 以及 validation/matched 配置。critic/director 离线数据集路径不导出，
其中的 exact-oracle 标签禁止进入 B5 schedule 或在线攻击运行时。

因此，本阶段只证明 P4 v2b 的真实数据、训练、持久化与严格 reload 链路已经贯通；
攻击是否优于 matched FGSM/PGD/MAD 仍须由后续 50-seed B5 development matched
实验判断。
