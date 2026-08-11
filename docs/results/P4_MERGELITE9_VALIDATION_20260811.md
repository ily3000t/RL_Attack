# P4 MergeLite9 正式 validation 记录（2026-08-11）

## 结论

本轮正式 validation 判定为 **NO-GO**。P4 STFA 在当前验证集上没有造成回报或安全事件劣化，动作翻转率与目标命中率也未达到继续执行 final audit 的条件。因此，冻结的 final cohort 未运行、未查看、未消费；下一步进入 reachability-aware v2，仅使用 validation cohort 调整方法。

## 可复现实验上下文

- 固定源码提交：`c9b310b`
- Python 环境：`RL_Attack_Core_Py310`（`E:\RL_Attack\.venv`）
- 执行设备：CPU
- Torch 线程：1
- preparation 校验：通过，共验证 15 个 artifacts
- validation seeds：`544000..544049`，共 50 个
- final seeds：`545000..545049`，共 50 个；本轮未运行、未触碰

首次使用旧 commit 执行 validation 时，在任何 episode 开始前因 proxy 签名不匹配而 fail-closed。该问题已由提交 `c9b310b` 修复；下列结果来自修复后固定提交上的正式 validation。

## 正式 validation 结果

### 回报

- clean mean return：`11.6245721090`
- attacked mean return：`11.6565477878`
- paired return drop（clean − attacked）：`-0.0319756788`
- exploratory paired bootstrap 95% CI：`[-0.0984, 0.0116]`

该 bootstrap 区间仅用于 validation 阶段的探索性判断，**不是 formal statistical evidence**，不得作为最终统计结论。

### 环境安全指标

- collision：attacked − clean = `0` 个百分点
- near_miss：attacked − clean = `0` 个百分点
- merge_success：attacked − clean = `0` 个百分点
- safety cost：attacked − clean = `-0.0350205862` / episode

攻击没有增加 collision 或 near-miss，也没有降低 merge success；平均 safety cost 反而轻微下降。

### 攻击行为与预算

- 选择攻击的时间步：`117 / 1331`
- nonzero perturbation：`117 / 117 = 100%`
- action flip：`20 / 117 = 17.094%`
- target hit：`1 / 117 = 0.855%`
- 连续扰动：117 个被选时间步全部达到 `L∞ = 0.025` 的预算边界

扰动预算确实被充分使用，但扰动方向的动作可达性不足：动作翻转较少，director 指定目标几乎无法命中。这说明当前瓶颈不是扰动幅度未用满，而是目标动作与局部策略决策边界之间缺少 reachability-aware 约束。

## 决策与下一步

本轮结果为 **NO-GO**：不运行 final audit，不使用 final seeds 进行诊断或调参。下一版本进入 reachability-aware v2 validation tuning，仍只允许使用 validation cohort；方法重新冻结并通过 validation 后，才能决定是否触发一次性 final audit。

## 证据边界

本结果仅属于仓库自有 MergeLite9 合成环境上的单 victim-training-seed P4 validation：

- 不是 SUMO 实验证据；
- 不是多 victim seed 的正式统计结论；
- 尚未完成 Random / FGSM / PGD / MAD matched-budget baseline 对比；
- 不授权直接进入 P5，也不构成防御有效性结论。
