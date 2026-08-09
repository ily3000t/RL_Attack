# P12 CartPole development victim cohort — 2026-08-09

## Status

The development victim-cohort stage is implemented and completed（已落实）.
Twenty fresh CPU victims were independently trained on clean commit
`1ee193f383bc0382049978df5b8456ee46645522`:

```text
Vanilla PPO / Adv-PPO / SA-PPO / CAR-PPO
  × training seeds 0, 1, 2, 3, 4
  × requested 100,000 PPO steps
```

All runs completed at 100,352 model steps because SB3 finishes complete
1,024-step rollouts. Every bundle contains 100 deterministic clean validation
episodes using seeds 10,000–10,099.

This document records the frozen victim cohort and validated development plan.
The 84,000-row attack matrix is not yet an empirical result and remains
pending; no development robustness ranking is claimed here.

## Environment

- Saved name: `RL_Attack_Core_Py310`
- Entry point: `E:\RL_Attack\.venv\Scripts\python.exe`
- Python 3.10.16, PyTorch 2.0.0 CPU, Stable-Baselines3 2.3.2,
  Gymnasium 0.29.1, NumPy 1.23.0
- `include-system-site-packages=false`; no packages were installed into or
  changed in the user's active Conda `pytorch` environment.
- Training used at most four simultaneous processes and one CPU thread per
  process. CUDA was not used.

## Clean validation results

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean across seeds | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla PPO | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00–500.00 |
| Adv-PPO | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00–500.00 |
| SA-PPO | 499.87 | 500.00 | 500.00 | 477.68 | 495.01 | 494.51 | 477.68–500.00 |
| CAR-PPO | 477.29 | 495.10 | 441.26 | 475.07 | 500.00 | 477.74 | 441.26–500.00 |

The weaker SA seed 3 and variable CAR seeds are retained. No model or seed was
discarded after observing clean performance. CAR-PPO has a substantial clean
quality/variance issue at the frozen hyperparameters; this must be included in
the clean/robust Pareto interpretation.

## Identity and provenance audit

- 20 model files and 20 `rl_attack.defense_run.v2` manifests exist.
- Every model file SHA-256 matches its training manifest.
- Every training manifest records commit `1ee193f`, `git_dirty=false`, CPU, the
  exact core and third-party locks, fresh training, and no input checkpoint.
- Loaded policy class, full robust configuration, PPO hyperparameters,
  optimizer learning rate, clip range, and `num_timesteps` match the manifest.
- All observation/action spaces match the CartPole policy-input contract.
- All 20 complete in-memory policy-state SHA-256 values are unique.

The ignored local bundles are under:

```text
E:\RL_Attack\outputs\p12_cartpole_development_20260809\<method>_seed<0..4>\
```

## Validated development design

The strict P12 design gate accepted:

- claim tier `development`, cohort role `test`;
- five independent training seeds per method;
- held-out paired episode seeds 20,000–20,199;
- Random Uniform, FGSM-CE, PGD-CE, and categorical MAD-PGD;
- per-feature L∞ base epsilon `[0.05, 0.05, 0.01, 0.01]`;
- epsilon ratios `[0, 0.25, 0.5, 1, 2]`;
- attack probability 1.0;
- PGD/MAD 20 steps × 5 restarts;
- maximum 128 attack-internal policy queries and 100 gradients per attacked
  step;
- 10,000 crossed hierarchical bootstrap replicates.

The accepted plan contains 20 victims, 420 strict shards, 4,000 clean rows,
80,000 attack rows, and 84,000 total episode rows. Its pre-publication plan
fingerprint on commit `1ee193f` was
`3af476d3ce8a57abc3ce1082a12a225cb00365dd11e0f6436863385d216e1ab4`.
A documentation commit changes the repository component of the final run
fingerprint by design, so the plan must be regenerated immediately before the
attack run.

The ignored resolved configuration is:

```text
E:\RL_Attack\outputs\p12_cartpole_development_20260809\benchmark_development.yaml
```

## Remaining development gate

The attack matrix must still be executed to completion and independently
verified from its shards. Until then there is no formal development comparison
of attack strength or defense effectiveness. The full registered grid performs
up to two billion attack-internal gradient evaluations on long CartPole
trajectories, so it must be treated as a long-running resumable workload rather
than an interactive smoke test.
