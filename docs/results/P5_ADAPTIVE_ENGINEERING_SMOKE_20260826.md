# P5 adaptive-attack engineering smoke 记录（2026-08-26）

## 结论

P5 adaptive-attack engineering smoke 已在干净提交上完成运行并通过独立
`verify`。本结果只证明冻结 PPO、BPDA 净化代理、真实 `RapidGuard.step`、
MergeLite9 transition、双账本与 manifest/replay 链路贯通；不构成防御有效性、
攻击强度或 SUMO 效果结论。

## 可复现实验上下文

- 源码提交：`dd4c4cd4`
- 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`），CPU
- 配置：`configs/experiments/p5_mergelite9_adaptive_engineering_smoke.yaml`
- episode seeds：`554100`、`554101`
- 输出：`outputs/p5_mergelite9_adaptive_smoke_dd4c4cd_20260826`
- manifest SHA-256：
  `5736cfd5dbc52e6de15c982d2878303c86154031bda03fba121fd5a7d0fb26fa`
- 独立验证：`status=verified`

## 工程结果

| 检查项 | 结果 |
| --- | ---: |
| 梯度有限且非零 | 2 / 2 |
| 非零扰动且满足预算 | 2 / 2 |
| 原始攻击造成动作翻转 | 2 / 2 |
| Guard 进入 purified 路径 | 2 / 2 |
| Guard 改变受攻击动作 | 1 / 2 |
| 真实 RapidGuard runtime step | 2 / 2 |
| 真实 MergeLite9 transition | 2 / 2 |

攻击侧账本记录 victim forward/backward `12/8`、defense forward/backward
`8/8`、BPDA surrogate `8`、预算投影 `10`、语义投影 `8`。防御侧账本记录
policy query `6`、detector query `4`、projection `2`、purification attempt
`2` 和 purified step `2`；两类账本不可兑换，也没有合并成统一 query 总额。

独立 verifier 通过了 artifact 完整性、victim 绑定、攻击账本、防御账本和
确定性运行时重放。P4 v2b development gate 仍固定为失败，matched 与
future-final seeds 均未消费。

## 证据边界

- detector 是确定性的 test-scope fixture，不是训练后的 RAPID detector；
- certificate 与 safety shield 被关闭；
- 未使用训练后的完整 RAPID-Guard bundle；
- BPDA 只覆盖 fixed-anchor purifier surrogate，不覆盖 Guard 的硬门；
- 没有 defended/undefended episode return 或 safety outcome 对比；
- 所有 formal、effectiveness、full-adaptive 与 SUMO claim flags 均为 `false`。

因此，`Guard changed action = 1/2` 只能证明工程链路按预期运行，不能解释为
防御提升了奖励或安全性。正式 P5 效果比较仍须等待一个能在相同 victim/seeds
上实际触发并产生轨迹伤害的 P4 攻击版本。
