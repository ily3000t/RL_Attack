# P4 v2b development validation：已落实运行结果（2026-08-25）

本记录对应 MergeLite9 单一冻结 PPO victim 的 P4 v2b B5 development
validation。它只用于判断是否应进入 matched baseline；不是 SUMO 证据、不是正式攻击
有效性结论，也不能外推为一般鲁棒性结论。

## 最终有效运行与独立校验

- 源码提交：`c9a42facd412ab551f61a12642a8b5f0920f7fa2`
- 环境：`RL_Attack_Core_Py310`，CPU，Torch / interop threads 为 1 / 1
- bundle：`outputs/p4_v2b_development_B5_c9a42fa_20260825`
- development seeds：50 个，`550000..550049`
- run manifest SHA-256：
  `3b6c86cb610ccdc8a321d018b692318f1b14057f94d36ffb1afa4e5d2271646b`
- preparation manifest SHA-256：
  `f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0`
- statistics contract SHA-256：
  `2a6f6d5d72ae9708847a907953a31e5da8e6de731b8a00fe02bd7c50ad317ef8`
- 独立 verify：`status=verified`；victim policy state 前后 SHA-256 一致

独立 verifier 重新核验了 preparation/run manifest、结果文件摘要、统计契约、种子、
query accounting 与 victim 不变性。校验通过表示运行及证据链完整，不表示攻击效果门通过。

## 50-seed 结果

| 条件 | mean return | mean safety cost | merge failure | collision | selected steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 9.10511305 | 4.57421142 | 0.12 | 0.00 | 0 |
| STFA v2b fixed schedule | 9.10511305 | 4.57421142 | 0.12 | 0.00 | 0 |
| STFA v2b online secondary | 9.10511305 | 4.57421142 | 0.12 | 0.00 | 0 |

三组 outcome 完全相同。fixed 与 online 均没有选中攻击时刻，也没有产生非零扰动或
动作翻转。

| 条件 | observation | gradient | projection | critic | director | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0 | 0 | 0 | 0 | 0 | 0 |
| STFA v2b fixed schedule | 1,321 | 0 | 0 | 1,321 | 0 | 2,642 |
| STFA v2b online secondary | 1,321 | 0 | 0 | 1,321 | 1,321 | 3,963 |

这些非零 query 是 clean-derived schedule 的 victim/B2 评估，以及 online-secondary
逐时刻的 B3 director 评估；gradient 与 projection 均为 0，说明 STFA 内层 solver
在本次运行中没有被调用。

## 未进入攻击的直接原因

50 个 clean trajectories 共包含 1,321 个候选时刻。按冻结契约计算
`max(predicted_target_risk - predicted_clean_action_risk, 0)` 后：

| opportunity 统计量 | 数值 |
| --- | ---: |
| maximum | 0.0397727191 |
| p95（NumPy linear） | 0.0307092592 |
| mean | 0.0150724258 |
| 达到固定 minimum opportunity 0.05 | 0 / 1,321 |

B3 director 的 `model_gate` 在 1,321 个在线决策中有 342 次为 true，但
`opportunity_gate` 为 true 的次数是 0。因此，fixed schedule 为空；online-secondary
即使模型门通过，也会被 opportunity gate 拒绝，最终两者 selected steps 都为 0。
这表明本阶段失败发生在时机筛选门，而不是内层扰动优化器。

## Development gate 结论

STFA fixed 相对 clean 的 discounted return drop、merge-failure increase、safety-cost
increase、collision increase 与 realized composite risk 均为 0。10,000 次 paired
bootstrap 得到 composite-risk effect 的 95% CI 为 `[0, 0]`；预注册判据要求
CI lower bound `> 0`，因此：

- integrity gate：通过；
- effectiveness gate：失败；
- overall development gate：失败；
- `effectiveness_claim_eligible=false`。

按照阶段契约，本结果不支持继续消费 matched baseline。`551000..551049` 的 matched
seeds 未消费；`552000..552049` 的 future-final seeds 也未消费。不得用本结果声称
STFA solver 无效，因为 solver 实际没有运行；当前只能得出“v2b 时机/机会门没有放行
任何攻击”的结论。

## 建议的 P4 v2c 方向（本阶段不实现）

下一步应先修正 selection contract，再重新验证 solver，而不是直接运行 matched：

1. 在 B2 训练/校准数据上检查 predicted risk 与 action-risk difference 的尺度和校准，
   将绝对 `0.05` 门改为预注册的相对优势、分位数或经校准阈值；不得依据本次 outcome
   反向选择能制造显著性的阈值。
2. 将 `has_target`、`model_gate`、`opportunity_gate`、`budget_gate` 和最终 selected 的
   逐级计数写入下一版结果契约，并先用 smoke test 确认确实触发 solver、产生合规非零
   扰动和完整 20×5 query accounting。
3. v2c 方法和阈值冻结后使用新的 development cohort。保留当前 matched/future-final
   cohort，不把已经观察过的 `550000..550049` 再包装成独立确认性证据。

只有 v2c development 的 realized composite-risk CI 下界通过预注册 gate，才进入同版本
的 Random/FGSM/PGD/MAD matched baseline；P5 仍保持接口、日志、预算与 adaptive smoke
范围，不据此宣称防御有效。
