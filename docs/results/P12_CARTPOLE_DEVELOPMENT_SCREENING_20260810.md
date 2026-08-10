# P12 CartPole development screening — 2026-08-10

## Status

The `development_screening` experiment is implemented, complete, and
independently verified（已落实并验证）.

This is a fast, non-formal screening experiment. It is intentionally declared
as `claim_tier: smoke` with cohort role `validation`: one training seed, 50
episodes, and a single epsilon ratio do not satisfy the registered development
or test-cohort gates. Its results may decide whether P4/P5 ideas merit a larger
run, but must not be reported as a formal robustness ranking.

## Frozen screening matrix

- Four frozen seed-0 victims: Vanilla PPO, Adv-PPO, SA-PPO, and CAR-PPO.
- Four attacks: Random Uniform, FGSM-CE, PGD-CE, and categorical MAD-PGD.
- Epsilon ratio fixed to `0.5` against the CartPole per-feature base profile
  `[0.05, 0.05, 0.01, 0.01]`.
- PGD-CE and categorical MAD-PGD retain 20 steps × 5 restarts.
- Maximum 128 policy queries and 100 gradient evaluations per attacked step.
- Fifty paired validation episodes use seeds 25,000–25,049.
- Statistics use 1,000 bootstrap replicates.

Each victim owns one clean shard plus four attack shards. The complete matrix
therefore contains 20 shards and 1,000 episode rows:

```text
4 victims × (1 clean + 4 attacks × 1 epsilon ratio) × 50 episodes
  = 20 shards
  = 1,000 rows
```

The tracked configuration is
`configs/experiments/p12_cartpole_development_screening_seed0.yaml`. Its four
checkpoint and training-manifest inputs are pinned by SHA-256 to the existing
ignored cohort under `outputs/p12_cartpole_development_20260809`.

## Verified execution

- Scientific commit: `dcbbc1b0ee24655332f36499022911795879c7f9`.
- Fixed worktree: `outputs/worktrees/p12-screening-dcbbc1b0ee24`.
- Environment: `RL_Attack_Core_Py310`, CPU-only Torch `2.0.0+cpu` from
  `E:\RL_Attack\.venv`.
- Core run time: 33 minutes; end-to-end process time: about 33 minutes 7
  seconds.
- Exit evidence: 20/20 shards, 1,000 rows, `resume_count: 0`, and empty
  stderr.
- Run fingerprint:
  `de7d4745c6648c681cc7a1114169f04e63baf725aa43e597540f564b7374f70d`.
- Manifest SHA-256:
  `8A1A9FE377EF72B9CF78B459902425897A521791E57C71A0033562AF4285B03E`.
- Independent `verify`: passed with 20 shards and 1,000 rows.

The manifest correctly records `formal_result_eligible: false`, with reasons
`claim_tier_is_smoke`, `claim_tier_is_not_formal`, and `cohort_is_not_test`.

## Execution isolation

- Saved environment name: `RL_Attack_Core_Py310`.
- Interpreter: `E:\RL_Attack\.venv\Scripts\python.exe`.
- Device: CPU only.
- Parallelism: four victim worker processes, one Torch thread per worker.
- `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` prevent nested BLAS/OpenMP
  oversubscription.

This parallel path is restricted to CPU `gymnasium_standard` runs with
`claim_tier: smoke`, cohort role `validation`, and the built-in environment
factory and victim loader. It cannot be combined with `--max-new-shards`.
Workers only evaluate victims; the single parent coordinator owns every output
write, including shard publication and `run_state.json`. Do not start multiple
coordinators against this output directory. Worker/thread counts are execution
controls and do not enter the scientific fingerprint. If the process is hard
interrupted, a partially computed victim batch is recomputed; only complete
shards already published by the parent can be reused by `--resume`.

The four-worker limit was selected for a 16-core/32-thread CPU while a separate
Conda `pytorch` workload is already using one SUMO environment, about four CPU
cores, one worker Torch thread, and four main Torch threads. The screening run
does not activate, modify, or terminate that Conda environment or process tree.
Only the `E:\RL_Attack\.venv` environment is used for this experiment.

## Screening results

Mean episode return at epsilon ratio `0.5`:

| Method | Clean | Random | FGSM | PGD | MAD | Episode-wise worst |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla PPO | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 |
| Adv-PPO | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 |
| SA-PPO | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 |
| CAR-PPO | 462.86 | 462.70 | 470.00 | 470.00 | 473.04 | 462.70 |

CAR-PPO's clean 95% episode-bootstrap interval is `[442.34, 480.98]`; its
episode-wise-worst interval is `[442.56, 480.72]`. In the matched-Vanilla
worst-over-attacks comparison, Adv-PPO and SA-PPO are both exactly `0.00`,
while CAR-PPO is `-37.30` with interval `[-57.89, -18.99]`.

Action-flip rates show that the attacks did affect the policy even when return
stayed at the CartPole ceiling:

| Method | Random | FGSM | PGD | MAD |
|---|---:|---:|---:|---:|
| Vanilla PPO | 9.124% | 24.824% | 24.824% | 23.384% |
| Adv-PPO | 9.404% | 20.416% | 20.416% | 20.348% |
| SA-PPO | 8.352% | 20.900% | 20.900% | 19.184% |
| CAR-PPO | 8.805% | 19.455% | 19.455% | 18.201% |

For Vanilla PPO, Adv-PPO, and SA-PPO, all 50 paired episodes under every
attack retained return 500. Each CAR-PPO gradient attack improved 13/50
episodes, left 37/50 unchanged, and harmed none; Random harmed 6, left 38
unchanged, and improved 6. Consequently, the screening does not establish a
defense advantage:
Adv-PPO and SA-PPO tie a saturated Vanilla baseline, while CAR-PPO has lower
absolute clean and attacked return.

The principal finding is a return-level sensitivity limitation in this
CartPole/epsilon setting. Action-level attack success is nonzero, but the task
metric is saturated. P4 should therefore first demonstrate a reliable
return-level attack effect (or use a less saturated victim/task or a stronger
predeclared epsilon profile) before P5 is judged against it. These one-seed
intervals quantify episode uncertainty only, not generalization across training
seeds. CartPole also supplies no collision metric.

## Run command

From PowerShell in `E:\RL_Attack`:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
E:\RL_Attack\.venv\Scripts\python.exe -m rl_attack.cli.p12_benchmark run `
  E:\RL_Attack\configs\experiments\p12_cartpole_development_screening_seed0.yaml `
  --output-dir E:\RL_Attack\outputs\p12_cartpole_development_screening_20260810 `
  --device cpu `
  --workers 4 `
  --worker-torch-threads 1
```

After completion, verify the bundle independently:

```powershell
E:\RL_Attack\.venv\Scripts\python.exe -m rl_attack.cli.p12_benchmark verify `
  E:\RL_Attack\outputs\p12_cartpole_development_screening_20260810
```

The commands above reproduce the completed bundle only when the same pinned
inputs and fixed scientific commit remain available. This report makes no
formal or cross-seed robustness claim.
