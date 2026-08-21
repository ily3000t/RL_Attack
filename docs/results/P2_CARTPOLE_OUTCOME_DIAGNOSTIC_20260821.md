# P2 CartPole outcome diagnostic 记录（2026-08-21）

## 结论

本轮 P2 outcome diagnostic 已完成并通过独立 `verify`，但其声明等级永久为
**post-hoc / diagnostic-only / nonformal**。四项机器门禁中两项通过、两项失败：

| Diagnostic gate | 结果 | 关键证据 |
| --- | --- | --- |
| `environment_outcome_sensitive` | PASS | Vanilla 的 `opposite_all` 使平均回报下降 `490.9`，平均最小归一化联合安全裕度下降 `0.71560` |
| `observation_attack_outcome_aligned` | FAIL | Vanilla 在 FGSM / PGD / MAD 下的平均动作翻转率均为 `0.5`，但平均配对回报下降均为 `0.0` |
| `pgd_incremental_value` | PASS | Vanilla 的最终迭代相对各 restart 第 1 次迭代最优值，目标均值增加 `11.18645`，动作翻转率增加 `0.125` |
| `defense_comparison_interpretable` | FAIL | Adv / SA / CAR 的训练与评估 epsilon、步数、restart 数和 step size 均不匹配 |

因此，CartPole 本身并非“回报对动作完全不敏感”：持续强制相反动作会迅速终止
episode。当前失败点更具体地位于**观测攻击目标与多步 outcome 不对齐**。同时，
已有 Adv-PPO、SA-PPO、CAR-PPO 数字处于 out-of-training threat，不能用于宣称防御
有效或无效。后续按既定路线进入 P4 v2b 的多步回报损失 / safety cost / merge
failure 风险目标；P5 只继续接口、日志、预算、manifest 与 adaptive-attack smoke，
不开展正式防御效果结论。

## 可复现实验上下文

- 固定源码提交：`8e91042a7a773ccf5c5f8159c3a48aeda0847d5b`
- Python 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`）
- Python / Torch：CPython `3.10.16` / Torch `2.0.0+cpu`
- 执行设备：CPU
- 配置：`configs/experiments/p2_cartpole_outcome_diagnostic_eps600.yaml`
- 配置 SHA-256：`074a13abedd389b3cb0e38181992c82243223543b99ade17fd82946a04e49587`
- episode seeds：`25000..25009`，共 `10` 个诊断 seeds
- victim：Vanilla-PPO、Adv-PPO、SA-PPO、CAR-PPO，各一个 training seed
- 干预：clean、全程相反动作、前 `1 / 5 / 20` 步相反动作
- 数据规模：`200` 条 episode、`320` 条 state-bank、`320` 条 PGD state trace
- PGD trace：`20` steps × `5` restarts，每个 clean episode 最多取 `8` 个状态
- epsilon ratio：`6.0`；评估有效 epsilon 为
  `[0.30000001, 0.30000001, 0.06000000, 0.06000000]`
- 输出目录：
  `outputs/p2_cartpole_outcome_diagnostic_eps600_8e91042_20260821`
- run fingerprint：
  `86c0e16674a9206484e59d90165b161ab0b945bb0fd4701f9daedfb5d2170a74`
- 外部计算的 `manifest.json` SHA-256：
  `f129f8eb98212bb72a8d6c3433e7227f6744fe89df7ce20660803030603b368b`
- 来源 P12 manifest SHA-256：
  `e41f9613f000a32407011f8f3cec388f63887c3bfc14ffd476736546eba31467`
- run 状态：`complete`；独立 `verify`：通过

`manifest.json` 按契约有意不记录自身哈希；上面的 manifest 摘要是运行结束后
从 bundle 外部计算的发布校验值。`verify` 重新加载冻结模型，并重放 PGD trace
及 production solver parity，而不是只比较文件哈希。

## 强制动作干预

下表的 margin 是每个 episode 的“最小归一化联合安全裕度”再取均值；负值表示
至少一个 CartPole 安全边界已被越过。return drop 均为相对同 seed clean 的配对
下降。

| 方法 | 条件 | Mean return | Mean paired drop | Mean min margin | Termination rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Vanilla | clean | 500.0 | 0.0 | 0.6203 | 0.00 |
| Vanilla | opposite first 1 | 500.0 | 0.0 | 0.6183 | 0.00 |
| Vanilla | opposite first 5 | 304.1 | 195.9 | 0.1627 | 0.40 |
| Vanilla | opposite first 20 | 9.1 | 490.9 | -0.0953 | 1.00 |
| Vanilla | opposite all | 9.1 | 490.9 | -0.0953 | 1.00 |
| Adv | clean | 500.0 | 0.0 | 0.8188 | 0.00 |
| Adv | opposite first 1 | 500.0 | 0.0 | 0.7966 | 0.00 |
| Adv | opposite first 5 | 255.0 | 245.0 | 0.0671 | 0.50 |
| Adv | opposite first 20 | 8.6 | 491.4 | -0.1077 | 1.00 |
| Adv | opposite all | 8.6 | 491.4 | -0.1077 | 1.00 |
| SA | clean | 500.0 | 0.0 | 0.4668 | 0.00 |
| SA | opposite first 1 | 500.0 | 0.0 | 0.3884 | 0.00 |
| SA | opposite first 5 | 352.9 | 147.1 | 0.1608 | 0.30 |
| SA | opposite first 20 | 9.0 | 491.0 | -0.1070 | 1.00 |
| SA | opposite all | 9.0 | 491.0 | -0.1070 | 1.00 |
| CAR | clean | 429.3 | 0.0 | 0.4222 | 0.50 |
| CAR | opposite first 1 | 403.2 | 26.1 | 0.2476 | 0.70 |
| CAR | opposite first 5 | 118.3 | 311.0 | -0.0159 | 1.00 |
| CAR | opposite first 20 | 8.5 | 420.8 | -0.1036 | 1.00 |
| CAR | opposite all | 8.5 | 420.8 | -0.1036 | 1.00 |

Vanilla `opposite_all` 的配对回报下降 95% bootstrap CI 为
`[490.2, 491.5]`；前 5 步干预的 CI 为 `[49.0, 342.9]`。这说明动作干预
达到足够持续时间后，CartPole outcome 会发生大幅变化。CAR 的 clean mean
return 只有 `429.3` 且 termination rate 为 `0.5`，也提示它在该 10-seed
诊断 cohort 上自身并未稳定处于天花板。

## 观测攻击与 outcome 对齐

该门禁读取已冻结并验证的 P12 eps600 source episodes。每个单元格为
`mean paired return drop / mean action flip rate`：

| 方法 | FGSM | PGD | MAD |
| --- | ---: | ---: | ---: |
| Vanilla | 0.0 / 0.5000 | 0.0 / 0.5000 | 0.0 / 0.5000 |
| Adv | 201.1 / 0.4974 | 201.0 / 0.4973 | 300.3 / 0.4836 |
| SA | 233.9 / 0.4946 | 241.4 / 0.4937 | 364.8 / 0.4601 |
| CAR | 334.8 / 0.4547 | 334.8 / 0.4547 | -3.8 / 0.3568 |

正式门禁只以 Vanilla 作为决策方法。其三个攻击均达到约 50% 动作翻转，却未
造成任何平均回报下降，低于冻结阈值 `1.0`，所以 gate 失败。防御模型上的较大
下降只作为上下文，不能反过来支持 gate，也不能作防御排名，因为训练—评估威胁
不闭合。

## PGD 逐迭代诊断

| 方法 | States | Final flip | Mean final objective | Best − final objective | Mean gradients to first flip | Production parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 80 | 1.00 | 19.336587 | 0.000000 | 3.7500 | 1.00 |
| Adv | 80 | 1.00 | 12.662610 | 0.000000 | 4.6500 | 1.00 |
| SA | 80 | 1.00 | 1.948081 | 0.000002 | 4.2250 | 1.00 |
| CAR | 80 | 1.00 | 0.971602 | 0.000003 | 7.8625 | 1.00 |

四个 victim 的 `any_flip_rate`、`best_seen_flip_rate` 与
`final_only_flip_rate` 均为 `1.0`，production parity 也均为 `1.0`。用于
机器决策的 Vanilla 指标显示，完整迭代相对第 1 次迭代的目标均值增加
`11.18645384`，动作翻转率增加 `0.125`，超过 `0.001` 或 `0.05` 的
“任一阈值”规则，因此 gate 通过。

这一结果只证明迭代对**单状态代理目标及动作翻转**有增量价值；它没有证明
`20 × 5` PGD 在 episode return 或 safety outcome 上优于单步攻击。所有状态
平均在约 `3.75..7.86` 次累计梯度评估内已首次翻转，也为后续考虑早停与分离
“动作翻转成本”和“继续提高多步风险目标成本”提供了依据。

## 防御训练—评估威胁闭合

Adv / SA / CAR 三种训练 recipe 均通过当前配置一致性检查，最后一次训练指标
有限，训练实际 `L∞` 约为 `0.02`。但本轮评估威胁与训练威胁存在同一组系统性
差异：

| 项目 | 防御训练 | eps600 评估 |
| --- | ---: | ---: |
| epsilon | scalar `0.02` | `[0.30, 0.30, 0.06, 0.06]` |
| evaluation / training epsilon ratio | — | `[15, 15, 3, 3]` |
| PGD steps | 10 | 20 |
| restarts | 1 | 5 |
| step size | scalar `0.005` | `[0.03, 0.03, 0.006, 0.006]` |

所以 Adv、SA、CAR 均标记为 `mismatched` 和 `out_of_training_threat=true`，
`defense_comparison_interpretable` 必须失败。Vanilla 是 clean reference，状态为
`not_applicable_reference`，不能错误地归类为 out-of-training defense。

## 决策路由与证据边界

1. **继续 P4 v2b。** 强制动作结果排除了“环境完全无 outcome 敏感性”，而
   Vanilla 观测攻击 gate 的失败支持把攻击目标从单步动作 / CE 改为多步回报
   损失、merge failure risk 或累计 safety cost，并用 risk-to-go 或短反事实
   rollout 选择攻击时机。
2. **在同一 MergeLite9 victim 上做 matched baselines。** FGSM、PGD、MAD 与
   P4 v2b 必须共享 victim、seed、epsilon/projector、时间窗口和预算账本；梯度、
   rollout 与 critic query 分币种报告，不能只用动作翻转率代替 outcome。
3. **P5 暂不作效果结论。** 当前只允许接口贯通、日志、预算、manifest 和
   adaptive attack smoke；正式防御比较必须先建立训练—评估闭合的 threat
   contract。
4. **不把 P2 当作正式统计证据。** 本轮只有 10 个诊断 seeds、每种方法一个
   victim-training seed，复用了先前 P12 结果并作 post-hoc 分析；它不是 final
   cohort、不是 SUMO 证据，也不证明任一防御优于 Vanilla 或任一攻击优于强基线。
