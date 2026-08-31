# P4 v2e 有符号短反事实回报攻击：冻结工程契约

状态：**目标、数据、critic、selector、比较矩阵与 gate 已预注册；preparation 已完整
验证但 offline critic adequacy 未通过，engineering 未解锁，尚未产生工程效果结论。**

运行记录见 `docs/results/P4_V2E_SIGNED_RETURN_PREPARATION_20260831.md`。

P4 v2e 是 v2d 的独立后继版本，不覆盖或修改 v2d 的已签名源码和结果。它修复 v2d
把负动作效应裁剪为零、动作排序弱，以及 categorical expected-loss 与 deterministic
argmax 执行不完全对齐的问题。

## 冻结主目标

对 observation `o`、clean action `c`、候选动作 `a` 和共同随机数 replicate `r`：

```text
d(o,a,r) = (G_clean(o,c,r) - G_a(o,a,r)) / 25
y(o,a)   = mean_r d(o,a,r)
```

正值表示候选动作降低短反事实折扣回报，负值表示它提高回报。标签不做 positive-part、
clipping 或 safety/failure 混合；clean action 在每行精确锚定为 `+0.0`。短反事实固定
为 H=12、gamma=0.99、R=4 paired common random numbers。merge failure、collision 和
safety cost 只作为独立结果端点报告，不能进入标签、critic loss、selector 破同分或
效果 gate。

critic 固定为 `8 -> 128 -> 128 -> 9` SiLU MLP 和线性输出。预测按 clean action
结构性居中：`q_a = z_a - z_c`。训练损失固定为非 clean action 的 SmoothL1 value loss
与全部非平凡动作对的 SmoothL1 gap-regression loss 之和，权重 1:1，beta=0.04。训练和
held-out 按 episode 分组，held-out 不用于 early stopping 或调参。

## 冻结攻击与时机选择

每个 clean trajectory step 只从当前 available、non-clean 动作中选取预测 `q_a` 最大的
动作；最大值必须严格为正。时机分数为：

```text
predicted_signed_loss * exp(min(target_logit - max_other_logit, 0))
```

该 attackability factor 只影响时机。clean trajectory 上的动作只是时机评分 probe；
实际 selected step 使用当前未扰动 observation 和同一次已计费 critic 向量，重新选择
available non-clean 动作中的全局最大正损失目标，不增加 policy/critic query。selector 枚举满足
`K=8 / min_gap=2 / window=16 / window_k=2` 的所有二时机组合，先最大化总分，再最大化
较弱时机分数，最后按 step/row index 确定性破同分。selector 是完整 clean episode 上
的非因果工程筛选工具，不产生 causal-online director claim。

内层固定为 ratio=6 的 MergeLite9 sensor projector、20 steps x 5 restarts。使用现有
STFA `FLAT` 路径：目标动作 C&W-style joint margin 对齐 deterministic PPO argmax，
softmax 下的 signed-return expectation 作为辅助项；factor、CE/MAD、failure 和 safety
项权重均为零。当前 step 的 clean action 和在线重选目标在一次 solver 调用中固定。

## 种子边界与一次性规则

- critic train episodes：`559200..559247`；
- critic held-out adequacy episodes：`559248..559263`；
- critic model seed：`547004`；
- one-shot engineering：`559010..559014`；
- matched reserved：`559300..559349`；
- future-final reserved：`559400..559449`。

先提交冻结实现，再运行 preparation。critic adequacy 与真实 FLAT solver-gradient probe
组成的双门全部通过后才允许运行五个 engineering seeds。engineering 只运行一次并完整
deterministic replay；若科学 gate 失败，保留结果、
不调参、不复用这些 seeds，下一算法版本必须另命名并分配新开发 seeds。

## Critic adequacy gate

held-out episode 聚合证据必须同时满足：held-out rows >= 300、runtime-eligible rows >= 200、
非 clean signed labels 正负样本各 >= 5%、tie tolerance=0.002、near-optimal top-1 >= 0.35
且超过 training-only majority-action baseline 至少 0.05、non-tied pairwise concordance
>= 0.65 且超过 training-only action-mean baseline 至少 0.05、opportunity NMAE <= 0.75、
critic 选中动作的 oracle signed loss 为正比例 >= 0.75、eligible rows 上真实 FLAT
solver objective（critic 向量已 detach）的 mutable observation 梯度 finite/nonzero 比例
>= 0.95。critic 自身输入梯度仅作诊断，不参与该门。所有 dataset/checkpoint/split/
supervision/victim/source SHA
还必须闭合。任一项失败时 engineering seeds 保持未消费。

## 五 seed 工程 gate

矩阵固定为 clean、Random、FGSM、PGD-20x5、MAD-20x5、legacy v2c composite、legacy
v2d positive-part return 和 v2e。所有条件共享 victim、episode seeds、ratio-6 projector、
quota 和 v2e-derived schedule；三种 STFA 使用同一个 restart plan。该矩阵 schedule-matched，
但不是 target/objective/query matched，且存在 v2e schedule bias，只用于工程筛选。

令 `d_s=G_clean-G_v2e`、`tau=1e-6`。主效果必须同时满足 mean/median > tau、至少 4/5
为正、最小 leave-one-out mean > tau，以及最大单 seed 正效果质量占比 <= 0.50。

令强基线 envelope `b_s=max(FGSM,PGD,MAD,v2c,v2d)`，`a_s=d_s-b_s`。相对门必须同时
满足 advantage mean/median > tau、至少 3/5 为正、最小 leave-one-out mean > tau，
以及最大单 seed 正 advantage 质量占比 <= 0.50。还要求每 condition x seed 至少一个
计划攻击在终止前可达，所有可达攻击均执行并产生非零扰动，预算与 query ledger 闭合。
v2e 每个实际 selected step 的原生查询向量固定为 observation=107、gradient=100、
projection=106、critic=1、director=1，总计 315；clean-trajectory schedule 查询只物理执行
一次，各 condition 仅保存逻辑归因账本。

即便通过，也只能表述“在五个预注册 MergeLite9 engineering seeds 上通过工程扩展门”。
formal effectiveness、superiority、statistical、SUMO、Vanilla 已解决和 causal-online 等
claims 继续固定为 false。
