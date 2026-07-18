# Dependency isolation

Each upstream repository must use a dedicated environment named
`rlattack-upstream-<repository>`. Start from the upstream environment file at
its locked commit; do not merge those dependencies with the core project.

Legacy SA-PPO-derived repositories commonly require Python 3.7 and MuJoCo 1.5.
They are used first for paper-fidelity checks. Their algorithms are then ported
and tested independently against the modern SB3 interface.

Recommended locations (all ignored by the parent Git repository):

```text
third_party/envs/SA_PPO/
third_party/envs/CAR-RL/
third_party/envs/WocaR-RL/
third_party/envs/paad_adv_rl-mujoco/
third_party/envs/paad_adv_rl-atari/
third_party/envs/ATLA_robust_RL/
third_party/envs/Robust-Gymnasium/
```

Do not make a single shared `paper-code` environment. Exact environment exports
belong under `third_party/environments/records/<name>/` only after that pinned
checkout has been reproduced successfully.
