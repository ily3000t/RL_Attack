# P4 v2d 短反事实回报损失攻击：目标与实现已落实

状态：**P4 v2d 算法、preparation、engineering runner 与 verifier 已落实；正式
preparation 已完成并通过完整 replay，5-seed 工程结果待运行。**

已验证 preparation：`outputs/p4_v2d_return_prepared_880836a_20260828`，1704 条 critic
rows，manifest SHA-256 为
`6ba2f1202140c0681d598506769e77dc6c37d6b893c3be50a5e1432fa8fe4eaa`；artifact、
critic、victim 与 counterfactual collection replay 均已验证。

## 本次修改说明

P4 v2d 不覆盖 v2a/v2b/v2c 历史实现，而是增加一条独立、可验证的攻击路线。主目标
固定为 H=12、折扣 0.99、R=4 共同随机数短反事实回报损失：

```text
L(o,a) = E_r[(G_clean^r - G_a^r)_+ / 25]
J(delta) = sum_a pi(a | project(o + delta)) * stopgrad(Lhat(o,a))
```

ratio 固定为 6，内层求解固定为 20 steps × 5 restarts。环境原始 reward 已包含已注册
的 safety penalty，因此 v2d 不再叠加 safety 权重。merge failure、collision 和累计
safety cost 仅作为结果端点报告，不能使回报 gate 通过。

## 已落实的 return-only 训练边界

v2d 使用专用 8→128→128→9 Softplus critic。它只消费通用离线轨迹数据的
`discounted_return_drop` 分量；failure/safety 没有输出头、损失项或共享表示梯度
路径。critic 固定 CPU、seed 547001，并按 episode 分组切分训练/验证集。checkpoint、
sidecar、dataset、victim、risk contract、return supervision 和模型状态均由 SHA-256
闭合。

这一区分很重要：v2d 的 return predictor 不是旧 9×3 多任务 critic 的运行时切片，
因此 auxiliary safety/failure 标签不能间接改变 return head。

## 已落实的时机选择与比较

5 个工程 seeds 固定为 `559000..559004`。每个 seed 先运行完整 clean episode；selector
仅对 victim top-3 non-clean 可达动作计算 predicted return opportunity，枚举所有满足
`K=8 / min_gap=2 / window=16 / window_k=2` 的二元时机组合，并按两时机预测机会总和
选择最优组合。不存在旧 B3 阈值门，也不使用 failure/safety 破同分。

保存的 `target_action` 只是 opportunity probe action。v2d 内层目标是全部 9 个动作的
categorical expected loss，并不使用定向 margin；实际 victim 执行 deterministic
argmax，因此该 surrogate mismatch 会在限制中明确报告。

工程矩阵固定为 clean、FGSM、PGD-20×5、MAD-20×5、旧 v2c composite comparator 和
v2d return-only STFA。所有攻击共享同 victim、seed、ratio-6 projector 与 v2d-derived
schedule；两种 STFA 共享相同随机重启 seed plan。该比较不是 target-matched、
query-matched 或 objective-isolated ablation，也不能产生 superiority claim。

## 已落实的 gate 与失败保留

scale-up 必须同时满足：

1. 每个 condition × seed 至少有一个计划攻击在实际终止前可达，且该 seed 的所有可达
   攻击均执行并产生非零扰动；攻击导致合法提前终止时，后续不可达时机单独计数；
2. v2d 的 paired signed discounted-return drop 均值与中位数严格大于零，且至少 3/5
   seeds 为正；
3. 每 seed 的 `v2d drop - legacy composite drop` 的中位数严格大于零。

若扰动为零或效果 gate 失败，run 仍保留完整证据；只有结构、账本或绑定损坏才拒绝
发布。所有 formal/effectiveness/superiority/statistical/SUMO/Vanilla claims 固定为
`false`。

prepare 与 engineering verifier 都使用唯一 JSON key 和类型敏感比较：重复键、
`false/0`、`1/1.0` 等类型混淆会失败关闭；源码、配置、文件账本、critic、victim 与
运行时记录在 replay 前后再次核验。

## 种子隔离

- critic：`559100..559163`；
- engineering：`559000..559004`；
- matched reserved：`559300..559349`；
- future-final reserved：`559400..559449`。

四组严格不相交。只有 critic seeds 可用于 preparation；本阶段不得根据 engineering
结果重训 critic 或修改冻结目标。通过 5-seed gate 后，才允许扩大到新的保留 seeds。
