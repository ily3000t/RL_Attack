# P4-v2f 双视图开发实验：已落实、已运行、未通过回报效果门

日期：2026-09-02  
环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`）  
实现提交：`9591e70`  
结果目录：`outputs/p4_v2f_development_9591e70_20260902`  
manifest SHA-256：`6d3fc2f3996d2fcfdd224aa70d4addeaa38081007291eaeb17b20e4f9cd93853`

## 结论

P4-v2f 的实现、正式 Dev-5 运行、普通验证和确定性 full replay 已全部完成。
运行链路有效：20/20 个被选攻击步产生非零扰动，20/20 个攻击步改变 PPO 动作，
20/20 个攻击步都提高了求解器内部的直接预期回报损失目标。然而，fixed timing 与
own timing 的回报效果门均失败，因此当前不能声称 v2f 能稳定降低 PPO 回报，也不能
声称其优于 FGSM、MAD 或 v2e。

失败不是攻击没有执行，而是短期 critic 目标与真实 episode 终局发生错位。seed
`556003` 中，clean PPO 原本以 `missed_merge` 失败；v2f 把轨迹改成了安全代价更高的
`merge_success`。攻击使安全性明显恶化，却因为成功合流奖励而提高了任务回报。这证明
回报与安全代价必须继续独立报告，不能合成一个会互相抵消的标量结论。

## 冻结协议

- victim、projector、动作本体、epsilon ratio=6、Dev-5 seeds 均保持不变。
- v2f 使用直接目标 `sum_a pi(a|o+delta) * q_hat(o_clean,a)`，固定 8 步、1 restart。
- 每个 episode 固定攻击 2 次；每次攻击严格计 30 个 native queries：11 observation、
  8 gradient、9 projection、1 critic、1 director。
- fixed timing 使用统一 v2e 已冻结时机；旧 Random/FGSM/PGD/MAD/v2c/v2d/v2e 只从
  byte-pinned golden bundle 导入，不重新运行。
- own timing 在完整 clean episode 上选择 v2f opportunity 最大且满足 K8/gap2/W16/KW2
  的两个时机。它是 offline/noncausal 开发诊断，不是在线 director。
- fixed timing 只匹配 victim、seed、schedule、projector 和 epsilon；攻击目标、solver
  与查询预算并不匹配。
- 这些 seeds 已用于开发和调参，不是独立 hold-out。

## 主要结果

这里 `DeltaG = G_clean - G_attacked`，正值表示攻击降低折扣回报；`DeltaC` 正值表示
安全代价上升。

| 方法 | mean DeltaG | median DeltaG | 正值 seeds | worst DeltaG | mean DeltaC | failure rate delta | native gradients |
|---|---:|---:|---:|---:|---:|---:|---:|
| FGSM | 0.3043 | 0.1139 | 5/5 | 0.0771 | 0.5090 | 0.0000 | 10 |
| PGD-20x5 | -1.0316 | 0.1128 | 4/5 | -5.7498 | 2.7217 | -0.2000 | 1000 |
| MAD-20x5 | 0.3117 | 0.2236 | 5/5 | 0.0904 | 0.5259 | 0.0000 | 1000 |
| v2c legacy | -1.5055 | 0.1784 | 4/5 | -8.2096 | 1.9871 | -0.2000 | 1000 |
| v2d retrain | -1.1110 | 0.0027 | 3/5 | -5.7498 | 2.5717 | -0.2000 | 1000 |
| v2e retrain | 0.2560 | 0.1848 | 5/5 | 0.0904 | 0.3597 | 0.0000 | 1000 |
| **v2f fixed** | **-1.0136** | **0.1784** | **4/5** | **-5.7498** | **2.7217** | **-0.2000** | **80** |
| **v2f own** | **-1.2671** | **0.2592** | **4/5** | **-7.3019** | **2.2041** | **-0.2000** | **80** |

fixed 和 own 均通过“median 为正”“至少 4/5 seeds 为正”“正效果不过度集中”三项，
但均未通过 mean、最小 leave-one-out mean 和 worst seed 三项。因此两个总 gate 都是
`false`。

v2f 只使用 PGD/MAD/v2c/v2d/v2e 的 8% gradient queries，但当前 mean DeltaG 为负，
所以不能把查询少解释为有效的效率优势。v2f 相对 FGSM、PGD、MAD、v2c、v2d、v2e
的逐 seed paired advantage 和查询效率均已写入 `summary.json` 与比较表。

## v2f 逐 seed 结果

| seed | fixed steps | fixed DeltaG | fixed DeltaC | own steps | own DeltaG | own DeltaC |
|---:|---:|---:|---:|---:|---:|---:|
| 556000 | 10,26 | 0.0859 | 0.1500 | 19,24 | 0.0832 | 0.1500 |
| 556001 | 21,25 | 0.1784 | 0.3000 | 0,4 | 0.2661 | 0.3000 |
| 556002 | 22,25 | 0.2236 | 0.3000 | 0,4 | 0.3581 | 0.3000 |
| 556003 | 9,12 | -5.7498 | 12.5587 | 0,7 | -7.3019 | 9.9705 |
| 556004 | 11,26 | 0.1941 | 0.3000 | 0,3 | 0.2592 | 0.3000 |

own timing 在 556001、556002、556004 上比 fixed timing 得到更大的回报下降，也把
median DeltaG 从 0.1784 提高到 0.2592；但它在 556003 上把负向异常进一步放大，导致
总体 mean 和 worst 更差。这说明 opportunity 排序对正常轨迹有信号，但没有识别终局
类别切换风险。

## seed 556003 诊断

| 条件 | return | discounted return | safety cost | length | 终止原因 | near miss | min gap | min TTC |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| clean | -9.4276 | -7.6079 | 8.2506 | 21 | missed_merge | 3 | 4.4787 | 1.8011 |
| v2f fixed | -1.5472 | -1.8581 | 20.8093 | 30 | merge_success | 9 | 0.4921 | 0.1213 |
| v2f own | 0.2017 | -0.3060 | 18.2210 | 30 | merge_success | 6 | 0.2550 | 0.0798 |

该 seed 同时给出三个证据：

1. 动作攻击确实工作；两个视图各有 2/2 action flips。
2. 安全攻击效果很强；安全代价增加、gap/TTC 显著降低、near miss 增加。
3. 直接预期回报损失目标没有稳定传递到真实 episode 回报；成功合流奖励压过了安全
   恶化，使任务回报反而改善。

因此，当前问题已经从“梯度或时机门没有运行”收敛为“surrogate 与终局目标不一致”。

## 求解器与查询闭合

- fixed：10/10 selected attacks 的内部 objective improvement 为正，均值 0.005115；
  10/10 action flips。
- own：10/10 selected attacks 的内部 objective improvement 为正，均值 0.007631；
  10/10 action flips。
- fixed native execution：300 queries，其中 80 gradients；golden schedule 的 387 queries
  只作为逻辑归因导入，没有重新执行。
- own native execution：300 queries；offline selector 实际执行 387 queries；own 总物理
  查询为 687。
- 本次实验实际物理查询总计 987；所有 step-to-episode、查询、outcome、objective、表格
  和 schedule SHA 均通过 verifier 重算。

## 下一版建议：v2g terminal-aware constrained return attack

不建议回档到 v2a，也不建议在当前 v2f 上直接扩大 seeds 或进入 P5 正式有效性实验。
v2f 已经证明“直接可微目标 + 低查询 solver”可以稳定改变动作；下一步应修复终局一致性：

1. critic 改为多头并保持指标分离：预测 paired return loss、terminal event delta、safety
   cost delta，不把它们简单加权成单个结果指标。
2. 将 H12 改成多时域/终局感知标签，例如 H12 + terminal-aware H32，或在短 rollout
   尾部加入 absorbing terminal/bootstrap 校正，覆盖 `missed_merge <-> merge_success` 切换。
3. director 使用 return-loss conservative lower bound，而不是点估计 opportunity；对预测
   会提高 merge-success 概率或 return loss 下界不为正的候选执行 return-branch veto。
4. own timing 允许 abstain 或 fallback：没有满足 return-harm 条件的两个时机时，不强迫
   return attack；若转用 safety branch，必须另行命名和报告，不能计为 return 成功。
5. 保留 seeds 556000..556004 作为开发/回归集，特别将 556003 固定为 hard-case gate：
   `worst DeltaG >= -0.25`、mean/median/LOO 均为正，且不得再通过“将 clean failure 变成
   attacked success”获得负 DeltaG。通过后再冻结新版本并使用新 hold-out seeds。

## 复现命令

```powershell
Set-Location E:\RL_Attack

.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2f_development run `
  configs\experiments\p4_mergelite9_v2f_development.yaml `
  --output-dir outputs\p4_v2f_development_9591e70_20260902

.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2f_development verify `
  configs\experiments\p4_mergelite9_v2f_development.yaml `
  --run outputs\p4_v2f_development_9591e70_20260902 `
  --expected-manifest-sha256 6d3fc2f3996d2fcfdd224aa70d4addeaa38081007291eaeb17b20e4f9cd93853 `
  --full-replay
```

## 产物

- `outputs/p4_v2f_development_9591e70_20260902/manifest.json`
- `outputs/p4_v2f_development_9591e70_20260902/summary.json`
- `outputs/p4_v2f_development_9591e70_20260902/comparison_table.md`
- `outputs/p4_v2f_development_9591e70_20260902/comparison_table.csv`
- `outputs/p4_v2f_development_9591e70_20260902/episodes.json`
- `outputs/p4_v2f_development_9591e70_20260902/steps.json`
- `outputs/p4_v2f_development_9591e70_20260902/schedules.json`

所有 formal evaluation、formal summary、effectiveness、superiority、statistical
significance、causal online director 和 SUMO effectiveness claims 继续固定为 `false`。
