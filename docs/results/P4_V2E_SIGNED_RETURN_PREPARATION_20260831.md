# P4 v2e 有符号短回报攻击：preparation 已验证、engineering 未解锁

## 结论

P4 v2e 已完成 formal preparation 和完整 deterministic verification，但 critic adequacy
的 `opportunity_nmae` 未达到预注册阈值，因此 **不得运行 5-seed engineering，也不能
形成攻击效果或优于基线的结论**。这不是执行故障：artifact、victim/critic binding、
counterfactual collection replay、deterministic training replay 和真实 FLAT solver
gradient probe 均已验证。

- 实现提交：`610601e`（`feat(p4): 已落实 v2e 有符号回报攻击协议`）；
- 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`）；
- victim：同一 MergeLite9 9-action Vanilla SB3 PPO；
- 威胁：PPO observation-only，MergeLite9 sensor-v2 projector，ratio 6；
- 标签：`E_r[(G_clean-G_a)/25]`，H=12、gamma=0.99、R=4 paired CRN，无 clipping；
- critic train seeds：`559200..559247`；held-out adequacy seeds：`559248..559263`；
- preparation：`outputs/p4_v2e_signed_prepared_610601e_20260830`；
- preparation manifest：`8fbf3dec0e461ff02c06dace869f954ed14f49371af2d22b6532c657ece7c83a`；
- dataset SHA-256：`73f20c8d33885d6d20e35f7d120f198e2d98d1f164dbbc62316cd402a3b5b492`；
- critic state SHA-256：`1921a9f493d37f986eae8f064cfe434db28dc7251e758b5c738fcf268df44c57`。

## Critic adequacy

| held-out 指标 | 实际值 | 冻结阈值 | 结果 |
|---|---:|---:|---|
| rows | 415 | >= 300 | 通过 |
| runtime-eligible rows | 411 | >= 200 | 通过 |
| positive non-clean label fraction | 0.529819 | >= 0.05 | 通过 |
| negative non-clean label fraction | 0.448795 | >= 0.05 | 通过 |
| near-optimal top-1 | 0.720195 | >= 0.35 | 通过 |
| top-1 相对 majority baseline 优势 | 0.058394 | >= 0.05 | 通过 |
| pairwise concordance | 0.813773 | >= 0.65 | 通过 |
| pairwise 相对 action-mean baseline 优势 | 0.090552 | >= 0.05 | 通过 |
| selected oracle-positive fraction | 0.851582 | >= 0.75 | 通过 |
| **opportunity NMAE** | **0.893272** | **<= 0.75** | **未通过** |

validation loss 为 0.010614，validation non-clean MAE 为 0.009532。预测的 mean maximum
opportunity 为 0.009381，而 oracle mean maximum opportunity 为 0.015725；opportunity
MAE 为 0.014079。当前 critic 的动作排序证据较强，但对 selector 直接使用的最大回报损失
幅值校准不足，因此不能安全地消费 engineering seeds。

## Solver 与完整验证

真实 detached-q `FLAT` solver-objective probe 在 411 个 predicted-positive held-out rows
上得到 finite/nonzero mutable-observation gradient fraction = 1.0，超过冻结阈值 0.95。
这说明当前阻断点不是内层 solver 无梯度，而是 offline critic 的 opportunity calibration。

完整 verifier 返回：

| 验证项 | 结果 |
|---|---|
| artifact integrity | 通过 |
| victim binding | 通过 |
| critic binding | 通过 |
| counterfactual collection replay | 通过 |
| deterministic critic training replay | 通过 |
| solver-gradient gate | 通过 |
| offline critic adequacy | **未通过** |
| engineering unlocked | **否** |

## 种子与声明边界

`559010..559014` engineering seeds、`559300..559349` matched seeds 和
`559400..559449` future-final seeds 均未消费。未创建绑定该 preparation 的正式
engineering 配置，也未运行 Random/FGSM/PGD/MAD/v2c/v2d/v2e 效果矩阵。

因此本轮只能表述“P4 v2e 实现与 preparation 可复现，但 critic readiness gate 未通过”。
不能据此声称 v2e 攻击有效或无效，所有 effectiveness、superiority、statistical、SUMO、
Vanilla 已解决和 causal-online claims 继续为 `false`。

## 冻结决策与下一步

1. 保留 v2e bundle 和失败结果，不放宽 `opportunity_nmae <= 0.75`，不在相同
   adequacy seeds 上反复调参或重训。
2. 若继续优化，应另命名为 v2f，并分配全新、互斥的 train/adequacy/engineering seeds。
3. v2f 优先针对幅值校准，而不是继续改变在线攻击目标：可预注册 training-only 的
   maximum-opportunity auxiliary loss、仅用 training split 拟合的正比例 calibration，
   或更低方差的 paired-return supervision；正式 adequacy split 仍只作一次性 gate。
4. 只有新的 preparation 双门通过后，才运行同 victim、同 ratio-6 projector 下的
   STFA/FGSM/PGD/MAD/v2c/v2d 小规模比较。
