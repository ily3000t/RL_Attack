# P4-v2f：校准排序 critic 与直接预期回报攻击契约

P4-v2f 是 MergeLite9 PPO 观测攻击的 development 版本。它保留 v2e 的
H12/R4 成对短反事实回报标签，但不继承 v2e 的目标动作 logit margin。
本版本只攻击 PPO 的 8 维策略观测；环境状态、奖励、动作执行器和 PPO 参数均不修改。

## 数据边界

- `Train-A`：`559200..559263`，复用已经冻结的 v2e signed-return 数据集；
- fit：`559200..559247`；
- held-out adequacy：`559248..559263`；
- `Dev-5`：`556000..556004`，只允许用于反复调参与回归比较；
- `Train-A` 与 `Dev-5` 必须完全不相交；
- `559010..559014`、`559300..559349` 和 `559400..559449` 不在 v2f development
  阶段消费。

Dev-5 不得参与 critic 梯度、输出尺度拟合、early stopping、阈值拟合或模型选择。
所有 Dev-5 结果均属于 reusable development evidence，不能作为独立验证或最终结论。

## Critic

Critic 输入为 clean policy observation 与 clean action，输出九个有符号短期回报损失：

\[
\hat q(o,a;a_c) = \hat z(o,a)-\hat z(o,a_c),
\qquad \hat q(o,a_c;a_c)=+0.
\]

训练目标由三部分组成：

1. signed magnitude SmoothL1，拟合每个非 clean 动作的回报损失幅度；
2. RankNet pairwise loss，拟合同一状态内非 tie 动作对的危险排序；
3. opportunity magnitude loss，拟合最佳非 clean 动作的最大回报下降机会。

归一化与任何输出尺度校准只能根据 fit episodes 计算。held-out episodes 只计算已冻结的
adequacy 指标，不参与优化。

进入 Dev-5 前还必须通过 preparation 工程门。held-out 门至少检查 300 个完整状态、300 个
direct-opportunity 可运行状态、正负标签覆盖、top-1 与 pairwise 排序相对 fit-only baseline
的优势、direct expected-opportunity NMAE（仅要求不超过 `1.25`）以及所选动作的 oracle-positive
比例。这个较宽的幅值阈值只用于排除退化 critic；它不构成效果或优越性结论。另一个独立门在
真实冻结 PPO 上求 `J` 对可攻击观测维度 `1..6` 的梯度，至少 `95%` 的 eligible held-out rows
必须同时 finite 且 non-zero。两个门都通过才允许消费 Dev-5。

## 攻击目标

在一次攻击求解中，critic 只在 clean observation 上查询一次，输出随后 detach。
候选扰动仅通过 PPO policy logits 接收梯度：

\[
J(\delta)=
\sum_a \pi(a\mid o+\delta,\mathcal A_{\mathrm{available}})
\hat q(o,a;a_c).
\]

v2f 最大化 `J`。接口仍需携带一个 `target_action`，但该动作只是诊断字段，不进入目标函数；
joint target margin、factor margin、CE/MAD、安全代价和 merge-failure primitive 的权重均为零。
通用 STFA 内部字段 `safety_costs` 以及 objective variant `SAFETY` 在本契约中只是数值槽别名，
不表示使用了安全 critic。

运行时只接受经过 v2f checkpoint/sidecar 严格 loader 证明的 critic；builder 必须逐字段匹配完整
artifact binding，不能用仅匹配 state hash 的人工记录替代。时机分数使用本次 clean PPO query
产生的概率并按 available-action mask 重新归一化，不从含义未冻结的 context score 猜测概率。

## 时机分数

clean trajectory 上的 development director 使用：

\[
s(o)=\max_{a\ne a_c}\hat q(o,a;a_c)
-\sum_a\pi(a\mid o)\hat q(o,a;a_c).
\]

完整 clean episode 的 top-2 排序属于 offline/noncausal development selector，不能声称为在线
因果 director。正式在线 director 必须在后续版本单独设计和验证。

## 低查询求解器

- epsilon ratio：`6.0`，仍由 MergeLite9 sensor-v2 projector 给出逐特征有效预算；
- 最大步数：`8`；
- restarts：`1`；
- 从 clean observation 启动；
- 第一次梯度给出 FGSM 方向，后续执行投影迭代；
- 不启用 early stop，使用最终投影 iterate；
- 每个被选中的攻击时刻固定 `8` 次原生梯度查询。

运行证据必须同时记录配置上限、实际梯度次数、预期回报目标改善、动作翻转和
真实 episode `Delta G`。Safety cost、collision 与 merge failure 独立报告，不参与攻击效果门。

该求解器通过 v2f 专用 runtime 配置使用既有 STFA 接口，但不修改通用 `attack.py`；这是为了
保证已经发布并绑定旧源码字节的 v2b/v2c/v2d/v2e artifact 仍能按原 verifier 复验。

## Development 双视图

1. `fixed_timing`：固定 unified v2e 的 Dev-5 timing 与 restart authority，直接复用哈希冻结的
   Random/FGSM/PGD/MAD/v2c/v2d/v2e 结果，只新执行 v2f；
2. `end_to_end`：v2f 使用自己的 opportunity top-2 timing，单独评估完整 v2f 系统。

两张表不得混称。fixed-timing 用于隔离 critic/objective/solver，end-to-end 用于评估完整方法。
没有原生 director 的基线若出现在 end-to-end 表中，必须标为 timing-unmatched descriptive only。

## Development gates

完整性门要求 artifact、dataset、victim、projector、seed、schedule、query ledger 和 summary replay
全部闭合。效果门预注册为：正 `Delta G` 至少 4/5、mean/median/leave-one-out mean 为正、最大正
质量占比不超过 0.60、最差 seed `Delta G >= -0.25`。强基线比较分别针对 FGSM 与 MAD，不能以
post-hoc oracle envelope 代替可执行基线。

本契约不授权 effectiveness、superiority、SUMO、统计显著性或正式 scale-up 结论。
