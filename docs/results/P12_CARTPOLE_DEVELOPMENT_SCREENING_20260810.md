# P12 CartPole development screening — 2026-08-10

## Status

The `development_screening` experiment definition is implemented（已落实）;
execution and empirical results are pending.

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

The result section must be updated only after both execution and verification
complete. Until then, no attack-strength or defense-effectiveness conclusion is
claimed.
