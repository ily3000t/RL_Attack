# P4-v2f preparation：已落实并通过 full replay

## 结论

P4-v2f expected-return critic preparation 已在 clean commit
`87783df64cd9f4e41d6ca9be757dd8612d6c52a3` 上完成。它只复用字节固定的 v2e
signed-return 数据集，没有重新采集反事实轨迹，也没有消费 Dev-5。artifact 完整验证、80 epochs
确定性重训、held-out adequacy 和真实冻结 PPO 输入梯度门全部通过，因此允许进入 Dev-5
development 实验。

这只是工程解锁，不是 effectiveness、superiority、统计显著性或 SUMO 结论。

## 固定边界

| 项目 | 值 |
|---|---:|
| fit episode seeds | `559200..559247` |
| held-out episode seeds | `559248..559263` |
| Dev-5（未用于训练） | `556000..556004` |
| fit rows | 1292 |
| held-out rows | 415 |
| Dev-5 training rows | 0 |
| dataset SHA-256 | `73f20c8d33885d6d20e35f7d120f198e2d98d1f164dbbc62316cd402a3b5b492` |
| training batch SHA-256 | `164fdf896519f6e1aadf4ae501c0ad3654e1e4c42e731fb6120e925359de00e0` |
| victim policy state | `9b29eb2b873851daa4aade33957d6d811f47c722d4616e48dfc83836391bb881` |

## Critic 训练

| 指标 | fit | held-out |
|---|---:|---:|
| total loss | 0.485610 | 0.492118 |
| magnitude loss | 0.006041 | 0.006049 |
| RankNet loss | 0.471615 | 0.477230 |
| opportunity loss | 0.007955 | 0.008840 |

critic state SHA-256 为
`994a656e5d8644082604211f28b45e86041c0940e5b28370545cc20d84d0f925`。

## Held-out adequacy

| 指标 | 观测值 | 门槛 | 结果 |
|---|---:|---:|---:|
| complete held-out rows | 415 | >= 300 | 通过 |
| runtime-eligible rows | 415 | >= 300 | 通过 |
| near-optimal top-1 | 0.802410 | >= 0.60 | 通过 |
| top-1 baseline advantage | 0.146988 | >= 0.05 | 通过 |
| pairwise concordance | 0.864343 | >= 0.75 | 通过 |
| pairwise baseline advantage | 0.145195 | >= 0.05 | 通过 |
| direct opportunity NMAE | 1.045783 | <= 1.25 | 通过 |
| selected oracle-positive fraction | 0.903614 | >= 0.85 | 通过 |

需要注意，opportunity NMAE 仍然偏高。当前 gate 只确认 critic 没有退化、排序和攻击梯度可用；
幅值校准质量必须在结果解释中作为限制保留，不能据此声称 v2f 已经有效。

真实 PPO direct-objective 梯度 probe 覆盖 415 个 eligible rows，415/415 在可攻击观测维度
`1..6` 上 finite 且 non-zero，比例 1.0（门槛 0.95）。probe 前后 victim policy 和 critic state
哈希均保持不变，PPO 参数梯度保持为空。

## Artifact 与验证

- 本地 artifact：`outputs/p4_v2f_prepared_87783df_20260902`
- preparation manifest SHA-256：
  `c597f3e940537b2b5874dbd12990914bdafbcfe2113673a31d7c4d5b36c509fb`
- critic checkpoint SHA-256：
  `d98b57f273624fc5350b0d717f5b13cfa5d53e251ec9613be3dff21073a16850`
- critic sidecar SHA-256：
  `5f266748653cbf2ccbb8f1326c38b9bb045948d29959d8d9d1a8335f64a6d747`
- artifact integrity：通过
- source dataset verification：通过
- critic binding verification：通过
- Train-A / Dev-5 disjoint verification：通过
- deterministic 80-epoch full replay：通过
- counterfactual recollection：未执行

运行命令：

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2f_preparation prepare `
  configs/experiments/p4_mergelite9_v2f_preparation.yaml `
  --output-dir outputs/p4_v2f_prepared_87783df_20260902

.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2f_preparation verify `
  configs/experiments/p4_mergelite9_v2f_preparation.yaml `
  --preparation outputs/p4_v2f_prepared_87783df_20260902 `
  --expected-manifest-sha256 c597f3e940537b2b5874dbd12990914bdafbcfe2113673a31d7c4d5b36c509fb `
  --full-replay
```

