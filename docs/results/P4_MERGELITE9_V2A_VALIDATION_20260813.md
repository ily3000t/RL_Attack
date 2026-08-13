# P4 MergeLite9 STFA v2a validation 记录（2026-08-13）

## 结论

本轮 v2a validation 的结论是：**可达性机制验证通过，可继续 P4
development；正式效果门禁、final audit 与 P5 授权仍为 NO-GO**。

v2a 将动作翻转率从 v1 的 `17.09%` 提升到 `41.18%`，将目标命中率从
`0.855%` 提升到 `37.25%`，平均配对回报下降也从 `-0.03198` 改善为
`+0.39370`。这支持“PPO softmax 语义对齐与 top-3 可达动作约束修复了
不可达目标”的机制假设。

但是，平均回报下降只占 clean mean return 的约 `3.39%`，没有达到预注册
的 `10%` 实用伤害条件；collision 未增加，merge success 只下降 `2` 个
百分点，也没有达到对应事件门槛。伤害还高度集中于少数 episode。因此不
执行 final audit，也不调用 final-only `analyze`。

## 可复现实验上下文

- 固定源码提交：`219f6abd73602f3151fedcc19a319e743a18923c`
- Python 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`）
- Python / Torch：CPython `3.10.16` / Torch `2.0.0`
- 执行设备：CPU
- Torch intra-op / inter-op threads：`1 / 1`
- 协议：`rl_attack.p4_mergelite9_effect_protocol.v2`
- 方法：`p4_mergelite9_stfa_effect_screening_v2a`
- preparation：约 `189.6 s`，15 个 artifacts 全部通过校验
- verify：通过，victim admission、seed 分离、artifact、softmax/top-3
  binding 均被独立重算
- validation：约 `29.5 s`，seeds `544000..544049`，50 个配对 episode
- final seeds：`545000..545049`；配置元数据在 verify 中完成预注册校验，
  但没有执行 rollout，也没有消费 final cohort

本轮 preparation 输出位于
`outputs/p4_mergelite9_effect_v2a_prepared_219f6abd_20260813`，validation
输出位于
`outputs/p4_mergelite9_effect_v2a_validation_219f6abd_20260813`。这些本地
实验制品继续由 Git 忽略，不作为源码提交内容。

## Validation 结果

### 回报与探索性统计

- clean mean return：`11.6245721090`
- attacked mean return：`11.2308684409`
- paired return drop（clean − attacked）：`0.3937036681`
- paired return drop median：`0`
- 相对 clean mean return：约 `3.39%`
- 受损 / 改善 / 完全不变：`18 / 12 / 20` 个 episode
- 探索性 paired bootstrap：`B=100000`、seed `20260813`
- 探索性 95% CI：`[0.00537, 1.02820]`

该置信区间只用于 validation 诊断，不是 formal statistical evidence。最大
单例（seed `544043`）的 return drop 为 `13.19725`，占净回报下降约
`67.04%`；前两个 episode 占全部正伤害约 `81.25%`。删除最大单例后平均
下降为 `0.13241`，删除前两个后仅为 `0.04725`，说明效果尚不稳健。

### 环境与安全指标

- collision：`0/50 → 0/50`
- merge success：`46/50 → 45/50`（`-2` 个百分点）
- near miss：`7/50 → 6/50`（没有恶化）
- safety cost：`108.21850 → 112.03331`
- attacked − clean safety cost：总计 `+3.81481`，平均每 episode `+0.07630`

最大回报伤害 episode 的 safety cost 反而下降，说明当前回报伤害与 safety
critic 的长期风险目标尚未稳定对齐。

### 攻击行为与预算

- selected：`102 / 1323 = 7.71%`
- nonzero perturbation：`102 / 102 = 100%`
- action flip：`42 / 102 = 41.18%`
- target hit：`38 / 102 = 37.25%`
- 目标动作分布以 action `5` 为主：`42 / 102`
- joint target 命中：`38`；仅命中一个动作 factor：`30`
- 总 query：`34812`
- gradient / projection / observation query：`10200 / 10812 / 11808`
- critic / director query：均为 `996`
- 连续扰动基本达到 `L∞ = 0.025`，没有离散编辑

机制门槛（nonzero `100%`、flip 至少 `30%`、target hit 至少 `25%`）均已
达到；主要剩余问题不再是目标不可达，而是伤害的长期一致性、跨 episode
稳健性及 joint target 与 factor 辅助目标之间的竞争。

## 决策与下一步

本轮不运行 final。下一阶段仍属于 P4 development，应在不使用 final seeds
的前提下完成：

1. 使用多个 victim-training seeds 和独立 development evaluation seeds，
   检查 median、trimmed mean、leave-one-out、harm rate 与尾部指标；
2. 将 director 标签进一步对齐增量 safety cost 或 merge-failure 风险；
3. 将 joint action target 设为词典序主目标，factor 项只作辅助或 tie-break，
   降低“只命中一个 factor”的情况；
4. 在冻结方法后完成 matched-budget Random / FGSM / PGD / MAD 对比；
5. 只有稳健伤害和匹配基线门禁均通过，才考虑一次性 final audit。

## 证据边界

该结果仅来自仓库自有 MergeLite9 合成环境和一个 PPO training seed：

- 不是 SUMO 实验证据；
- 不是多 victim seed 的正式统计结论；
- 尚未完成 matched-budget baseline 对比；
- 不证明 STFA 已优于强攻击基线；
- 不授权直接进入 P5，也不构成防御有效性结论。
