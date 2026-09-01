# P4 v2e post-gate 八条件综合诊断（已完成，2026-09-01）

## 结论状态

本次完整的 8-condition 工程诊断矩阵已经运行并通过确定性重放验证。它比较了
clean、Random、FGSM、PGD、MAD、v2c、v2d 和 v2e，但由于冻结的 v2e critic
在 preparation 阶段未通过 opportunity 幅值校准门，本结果永久为
post-gate、diagnostic-only、claim-ineligible，不能形成正式有效性、优越性、统计显著性、
SUMO 或因果在线 director 结论。

- 环境：`RL_Attack_Core_Py310`（仓库 `.venv`）
- 固定源码提交：`4fbef3a84777a66a92b6d689ff49f1475f2ed963`
- victim：同一 MergeLite9 PPO
- threat：PPO observation-only，`epsilon_ratio=6.0`，有效逐维上限 0.3
- 固定诊断 seeds：`559500..559504`
- formal engineering seeds `559010..559014`：未消费
- matched `559300..559349` / final `559400..559449`：未消费
- 输出：`outputs/p4_v2e_postgate_diagnostic_4fbef3a_20260901`
- run manifest SHA-256：
  `d6816034b185ea93ca471b74d0905ad6ca647f85daef1418fb857aeb6de4f7fa`
- artifact、failed-preparation gate、victim、shared restart plan 和完整 8×5
  deterministic replay：全部验证通过

准备门并非全面失效：near-optimal top-1 为 0.7202、pairwise concordance 为
0.8138、selected-oracle-positive 为 0.8516，均通过；失败项仅为 opportunity NMAE
`0.8933 > 0.75`。因此本矩阵可诊断“排序信号尚可但幅值失准的 critic 是否仍能产生攻击”，
但不能证明完整 v2e 设计有效。

## 同条件结果

主指标为同 seed 的折扣回报下降 `ΔG = G_clean - G_attack`；正值表示攻击降低回报。
查询数只列各方法 native execution，不重复加入共享 schedule 的 logical attribution。

| 条件 | mean ΔG | median ΔG | 正下降 seeds | 最大正贡献占比 | 动作翻转 | mean episode drop | mean safety Δ | native grad | native total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.0641 | 0.0000 | 2/5 | 0.5755 | 3/10 | 0.0745 | 0.0900 | 0 | 10 |
| FGSM | 0.9651 | 0.0915 | 3/5 | 0.9595 | 10/10 | 1.2186 | 1.7981 | 10 | 50 |
| PGD20×5 | 0.9699 | 0.0474 | 4/5 | 0.9592 | 10/10 | 1.2246 | 1.8571 | 1000 | 3130 |
| MAD20×5 | 0.1847 | 0.1069 | 4/5 | 0.7313 | 10/10 | 0.2135 | 0.4531 | 1000 | 3130 |
| v2c composite | 0.0062 | 0.0804 | 3/5 | 0.5202 | 9/10 | -0.0041 | 0.1609 | 1000 | 3140 |
| v2d positive-part | 0.6879 | 0.0000 | 1/5 | 1.0000 | 5/10 | 0.8956 | 1.3643 | 1000 | 3140 |
| **v2e signed-return** | **0.1465** | **0.0883** | **4/5** | **0.4382** | **10/10** | **0.1705** | **0.3002** | **1000** | **3150** |

clean 的 mean discounted return 为 8.1349、mean episode return 为 9.8753、mean safety
cost 为 1.3305；v2e 对应的 mean discounted return 为 7.9884、mean episode return 为
9.7048、mean safety cost 为 1.6307。

v2e 自身满足预先定义的诊断性 return-effect 公式：mean/median 为正、4/5 seeds
为正、leave-one-out 最小均值为 0.1016、最大正贡献占比 0.4382；它在该固定
5-seed cohort 上呈现“小且较分散”的描述性回报下降。该子指标不覆盖失败的 preparation gate，也不等于
正式效果门通过。

FGSM 与 PGD 的 mean ΔG 分别为 0.9651 和 0.9699，差异仅 0.0047；PGD 使用
1000 次 native gradient query，FGSM 仅 10 次。两者约 96% 的正下降质量都来自单个
seed `559501`，因此高均值不能解释为跨 seed 稳定优势。

## 每 seed 配对结果

| seed | Random | FGSM | PGD | MAD | v2c | v2d | v2e | strong envelope | v2e − envelope |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 559500 | 0.0000 | 0.0915 | 0.0474 | 0.0474 | -0.1931 | -0.1931 | **0.3260** | 0.0915 | **0.2345** |
| 559501 | 0.0000 | **4.8307** | 4.7179 | 0.7258 | 0.0896 | 3.7537 | 0.0883 | 4.8307 | **-4.7424** |
| 559502 | 0.1360 | -0.1302 | 0.0410 | 0.1069 | -0.1302 | -0.1212 | **0.2429** | 0.1069 | **0.1360** |
| 559503 | **0.1843** | 0.1124 | 0.1124 | 0.1124 | **0.1843** | 0.0000 | 0.0867 | 0.1843 | **-0.0976** |
| 559504 | 0.0000 | -0.0787 | -0.0692 | -0.0692 | **0.0804** | 0.0000 | -0.0114 | 0.0804 | **-0.0918** |

strong envelope 是按 seed 事后取 FGSM、PGD、MAD、v2c、v2d 中最大 ΔG 的 oracle
包络，不含 Random，也不是一个可直接运行的单一攻击方法。
v2e 对包络为 2 胜、0 平、3 负；mean advantage `-0.9123`，median advantage
`-0.0918`，因此 strong-baseline-envelope 子门失败。

## v2e 对每个比较方法

advantage 定义为同 seed 的 `ΔG_v2e - ΔG_comparator`；正值代表 v2e 更强。

| comparator | mean advantage | median advantage | v2e 胜-平-负 |
|---|---:|---:|---:|
| Random | 0.0824 | 0.0883 | 3-0-2 |
| FGSM | -0.8186 | 0.0672 | 3-0-2 |
| PGD20×5 | -0.8234 | 0.0578 | 3-0-2 |
| MAD20×5 | -0.0382 | 0.0578 | 3-0-2 |
| v2c composite | 0.1403 | -0.0013 | 2-0-3 |
| v2d positive-part | -0.5414 | 0.0867 | 3-0-2 |

v2e 对 FGSM、PGD、MAD、v2d 都是 3/5 seeds 配对占优且 median advantage 为正，
但均值被少数大幅失败反转；对 v2c 的均值更高，却只在 2/5 seeds 占优。`n=5`
不足以形成显著性结论，即使出现 5/5 单侧 sign pattern 也不能在双侧 0.05 水平显著。

## 攻击是否真正执行

- v2e schedule steps：10/10 reachable、10/10 selected、10/10 nonzero。
- v2e actions：10/10 相对 clean action 翻转。
- live target：10/10 non-clean 且 critic signed loss 严格为正。
- probe/runtime target：9/10 匹配；唯一不匹配仍满足 live-target contract。
- query ledger：闭合；victim policy state 在 run 前后完全一致。

因此此前“攻击时机门未触发、攻击未真正运行”的问题已经解决。本次 v2e 效果有限不能归因于
攻击没有执行。

## 查询账本

- physical shared schedule：390 queries，只对每个 seed 的 clean schedule 计算一次。
- native execution（七个攻击条件合计）：15,750 queries。
- 实验实际 physical total：16,140 queries，即 `390 + 15,750`。
- logical per-condition schedule attribution：2,730 queries，仅用于公平归因；它不是新增物理
  开销，不能再加到 16,140 上。
- v2e 每个被选 step 的 native query vector 为 obs 107 / grad 100 / projection 106 /
  critic 1 / director 1，共 315；10 steps 合计 3,150。

按总 signed drop / native gradient query 的描述性比值，FGSM、PGD、MAD、v2c、v2d、v2e
依次约为 0.4826、0.00485、0.000923、0.000031、0.00344、0.000733。由于各方法的
objective 与 target 不匹配，这只能作为 query/效果 Pareto 描述，不能单独证明算法效率。

## 结果解释

1. **v2e 在该固定 5-seed cohort 内呈现小且较分散的描述性效应。** 4/5 正下降和较低的
   最大正贡献占比说明 signed-return 方向不是完全无效；但它的 mean 远低于 FGSM/PGD，
   与 MAD 尚无明确胜负，对 v2c 仅 2/5 胜且 median advantage 为负，对 v2d 虽 3/5 胜且
   更分散但 mean 更低。oracle 包络门明确失败，不能进入正式 scale-up，也未显示总体优势。
2. **FGSM/PGD 的高均值主要是 seed 559501 的两步联合轨迹分岔。** 在该 seed 上
   FGSM/PGD 分别产生 4.8307/4.7179 的 ΔG，而 v2e 仅为 0.0883。FGSM 在 step 14
   执行动作 8，v2e 按 critic target 执行动作 3；累计 safety cost 数值上分别为
   12.8831 和 4.6425，clean 为 4.4925。该现象与 critic target/幅值校准不足的假设一致，
   但单个 seed 和两次联合干预不能建立因果关系，也不能排除 objective、target 与轨迹分岔
   等共同因素。
3. **PGD 在本 cohort 上没有体现与 100×梯度开销匹配的收益。** 它相对 FGSM 的 mean
   ΔG 只增加约 0.5%，并共享同一个单 seed 集中问题；FGSM 应继续作为必要的廉价强基线。
4. **MAD 与 v2e 最接近。** MAD mean 略高（0.1847 对 0.1465），v2e 在 3/5 seeds
   更强且下降更分散；当前样本不足以判定谁更优。
5. **v2c/v2d 呈现明显的 seed 变异与贡献集中。** v2c mean 接近零；v2d mean 较高却仅
   1/5 为正且全部正贡献集中在 seed 559501。相比之下，v2e 在本 cohort 的下降更分散，
   但这不是跨 cohort 稳定性结论。
6. **collision 与 merge-failure 未变，但 near-miss 有变化。** 所有条件 collision rate
   均为 0，merge-failure rate 均为 0.2；v2e 没有新增 collision、merge failure 或 near-miss。
   seed 559501 中 FGSM、PGD、v2d 分别出现 6、6、5 个 near-miss，而 clean、MAD、v2c、
   v2e 为 0。v2e 的 ΔG 主要反映 shaped reward / safety-cost 和轨迹细节变化，不能写成碰撞
   或合流失败攻击成功。

## 后续决策

不应回滚到 v2a，也不应直接扩大 v2e 规模。下一步应定义 v2f preparation，并在不查看
formal/matched/final seeds 的前提下完成：

1. 修复 critic opportunity 幅值校准门，而不是放宽 `NMAE <= 0.75`；增加对 destructive
   action tail 的幅值监督和校准诊断。
2. 先在开发 seeds 上对两个攻击时刻做单步消融、9 动作短反事实枚举，并报告 critic
   rank/regret 校准；再根据证据选择对可达 action 的保守 return-loss 排序，或把 untargeted
   expected-return objective 作为候选，而不是现在直接冻结新 objective。
3. 继续保留 FGSM、PGD、MAD、v2c、v2d 同 schedule 对照；重点报告 median、正 seed 数、
   leave-one-out 与 concentration，不能只优化 mean。
4. 新 critic 先重新通过 preparation 双门，再使用全新 engineering seeds；本次
   `559500..559504` 已消费，不得用于 v2f 调参或筛选。

本矩阵使用 v2e 派生的非因果 clean-episode schedule，可能偏向 v2e；同时 comparator 的
target、objective 和 query 均不完全匹配。结果仅适用于单个 MergeLite9 PPO victim，不能
外推到 SUMO 或未来 `PPO + Safety Shield + ACCVP` 分层验证。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2e_postgate_diagnostic run `
  configs/experiments/p4_mergelite9_v2e_postgate_diagnostic.yaml `
  --output-dir outputs/p4_v2e_postgate_diagnostic_4fbef3a_20260901

.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2e_postgate_diagnostic verify `
  configs/experiments/p4_mergelite9_v2e_postgate_diagnostic.yaml `
  --run outputs/p4_v2e_postgate_diagnostic_4fbef3a_20260901 `
  --expected-manifest-sha256 `
  d6816034b185ea93ca471b74d0905ad6ca647f85daef1418fb857aeb6de4f7fa
```
