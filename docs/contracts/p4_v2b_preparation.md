# P4 v2b preparation contract

P4 v2b is a MergeLite9 development preparation, not a SUMO or formal
effectiveness result. It imports the already-admitted PPO checkpoint from the
v2a bundle without retraining or changing it. The imported checkpoint,
manifest, and policy state are pinned independently; its historical training
commit (`3a1b114`) is retained as victim provenance and is not required to be
the current preparation commit.

## Offline labels and trainable artifacts

The critic cohort is exactly seeds `548000..548199`; the director cohort is
exactly `549000..549199`. At every clean pre-action state until termination or
time limit, the counterfactual oracle evaluates all nine possible first
actions followed by frozen deterministic PPO continuation. The risk contract
is fixed to horizon 64, discount 0.99, one common-random-number replicate,
return/safety scales 25/10, and unit weights for return drop, merge failure,
and cumulative safety cost.

`rows_per_episode: 64` in the B2 sidecar is the protocol maximum, not an
assertion that every episode has 64 rows. The full collector contract records
the actual row count of every episode and stops immediately after a terminal
or truncated transition.

Simulator latent state and RNG state never cross the dataset boundary. The B2
critic learns the three primitive all-action risks. The B3 offline training
dataset necessarily persists exact oracle risks, opportunities, diagnostic
targets, and selection labels; these are privileged labels, not online-safe
runtime inputs. The B3 model learns only whether to select a step and has no
learned target-action head. Verification may inspect those offline files, but
the B5 executable hand-off forbids them and does not export their paths.

## Frozen development cohorts

Preparation writes, but does not execute, two configurations:

- development validation: seeds `550000..550049`;
- matched baselines: seeds `551000..551049`.

The matched configuration preregisters clean, Random, FGSM, PGD20x5, MAD20x5,
STFA-v2b fixed-schedule, and STFA-v2b online-secondary conditions. Fixed
conditions share one schedule derived from the clean trajectory using only B2
predicted composite risk; neither exact counterfactual risk nor B3 offline
labels are available to that schedule. Random and FGSM retain native
efficiency accounting; no artificial query matching is claimed.

The fixed schedule is fully deterministic. At each clean row it ranks the
victim-softmax top-three reachable non-clean actions, chooses the greatest B2
predicted composite risk (lower action index breaks a tie), and defines
opportunity as the positive target-minus-clean predicted-risk difference. Rows
below 0.05 are ineligible. Remaining rows are greedily considered by decreasing
opportunity, then lower step and row index, with the exact K8/gap2/W16/KW2
ledger replayed over steps 0..63. The selected steps and targets are reused
unchanged by every fixed-schedule condition and never respond to an attacked
trajectory.

Random is independent per-feature uniform noise followed by the ratio-6
projector. FGSM and PGD maximize clean-greedy-action cross entropy; MAD
maximizes `KL(pi_clean || pi_candidate)`. PGD and MAD use random uniform starts,
20 steps, five restarts, automatic per-feature step size `2*epsilon/20`, and a
deterministic lower-restart tie break. The generated configuration contains the
complete byte-hashed method, RNG, schedule, and query contracts.

Query currencies remain separate: observation, input-gradient, projection,
B2-critic, B3-director, and their unweighted exact sum `total_queries`. Batching
does not reduce counts: a batch of N observations counts N. Fixed-schedule
construction is physically performed once per episode but its full logical
observation/critic cost is charged to each fixed condition; the shared physical
cost is also reported separately. Dummy Random/FGSM queries are forbidden.
Excluding that shared schedule charge, the frozen per-applied-attack totals are
Random 1, FGSM 5, PGD/MAD 313, STFA fixed 314, and STFA online-secondary 315;
the generated contract retains every component rather than relying on totals.

Seeds `552000..552049` are future-final reservations. Preparation emits no
final configuration and contains guards that prevent collection or execution
from consuming them. Bootstrap seed is `553001`; attack RNG base is
`55100000`.

## Runtime and verification

Preparation requires the repository `.venv`, recorded as
`RL_Attack_Core_Py310`, CPU, and a clean source tree. The CLI sets OMP, MKL,
OpenBLAS, and NumExpr thread variables to one before importing NumPy, Torch, or
SB3; an in-process call that imported the scientific stack first fails closed.
Torch and interop threads are also checked at runtime. Exact Python, NumPy,
Gymnasium, Stable-Baselines3, Torch, and PyYAML versions plus the core dependency
lock hash are recorded and must match during verification. All files are
no-overwrite and SHA-256 bound. Verification requires the external
preparation-manifest SHA-256, reloads the frozen PPO, B2 dataset/critic and B3
dataset/director, then reconstructs the new trajectory-risk runtime directly.
It does not use the legacy learned-director audit entry point.

Run after the implementation commit is clean:

```powershell
E:\RL_Attack\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2b prepare `
  E:\RL_Attack\configs\experiments\p4_mergelite9_v2b_preparation.yaml `
  --output-dir E:\RL_Attack\outputs\p4_mergelite9_v2b_prepared_<commit>_<date>
```

Record the returned manifest hash, then verify:

```powershell
E:\RL_Attack\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2b verify `
  E:\RL_Attack\outputs\p4_mergelite9_v2b_prepared_<commit>_<date> `
  --expected-manifest-sha256 <SHA256_FROM_PREPARE>
```

The verifier returns a `verified_bundle` hand-off record for B5. Its executable
allowlist contains only victim, critic, director, runtime, and stage-config
artifacts. Critic/director training datasets are named as forbidden offline
artifacts but their paths are not exported. The verifier performs a final
manifest/artifact/source/dependency rehash immediately before returning. This
is a point-in-time check, not a filesystem lock, so B5 must rehash each
allowlisted executable immediately before opening it and reject every unlisted
bundle file. A later clean integration commit is allowed: verification compares
every relevant B1/B2/B3/B4 source-file hash instead of requiring the current Git
HEAD to equal the preparation HEAD.
