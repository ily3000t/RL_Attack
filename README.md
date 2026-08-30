# RL Attack

Independent PPO attack-and-defense research benchmark built around Gymnasium,
Stable-Baselines3, public driving environments, and a frozen SUMO highway-merge
scenario.

This repository is intentionally independent from `WCDT_ACCVP_Attack`.
The WCDT project is treated as a read-only source of versioned scenarios and
victim-policy bundles. No runtime import from that repository is allowed.

## Current milestone

Phases P0/P1 are implemented:

- versioned threat-model and perturbation contracts;
- categorical SB3 policy adapter;
- Random-Uniform and Random-Sign observation attacks;
- FGSM-CE and PGD-CE observation attacks;
- Categorical-MAD with random-start PGD;
- single-environment evaluation wrapper and runner;
- frozen `sumo_merge_core_v1` scenario provenance;
- locked, isolated upstream research repositories.

Release record: [`docs/releases/P0_P1.md`](docs/releases/P0_P1.md).

Phase P2 defense implementations are also complete:

- native SB3 Vanilla PPO;
- an explicitly specified Adv-PPO engineering baseline;
- clean-room SA-PPO and empirical CAR-PPO-style objectives;
- one-step IBP certification of greedy actions for supported MLP actors;
- auditable model bundles with resolved configs, hashes, runtime versions,
  clean evaluation seeds, and fidelity metadata.

The implementation gate is complete; multi-seed benchmark sweeps remain a
separate compute run. Release record: [`docs/releases/P2.md`](docs/releases/P2.md).

Phase P3 strong-attack reproduction machinery is implemented:

- a clean-room categorical Robust-Sarsa critic and observation attack;
- a clean-room PA-AD stochastic-PAMDP director, actor, and training loop;
- victim/checkpoint hash binding and fail-closed action-mode contracts;
- a learned-attacker training CLI;
- a paired, budget-matched audit spanning PGD-CE, categorical MAD-PGD,
  Robust-Sarsa, and PA-AD with strict artifacts and worst-over-attacks.

PA-AD formally supports only stochastic categorical victims; the distinct
deterministic D-PAMDP is not approximated. Its learned director is bound to one
victim and one exact non-zero perturbation contract; epsilon sweeps train and
pin a separate director per non-zero epsilon. The implementation/test gate is
separate from the pending fixed-checkpoint, multi-seed statistical runs.
Release record: [`docs/releases/P3.md`](docs/releases/P3.md).

Phase P4 STFA implementation is complete:

- hard per-episode temporal ledgers and factorized categorical targets;
- semantic policy-input projection for SUMO and Highway contracts;
- continuous projected optimization plus separately charged discrete search;
- victim-bound safety-critic/director artifacts and strict NPZ training paths;
- a paired, deterministic-argmax audit with fail-closed evidence scopes.

This is an implementation and contract milestone. A stable SUMO PPO victim is
not yet available, so no SUMO effectiveness or strongest-attack result is
claimed. Contract: [`docs/contracts/p4_stfa.md`](docs/contracts/p4_stfa.md).
Release record: [`docs/releases/P4.md`](docs/releases/P4.md).

Phase P5 RAPID-Guard implementation and audit contracts are complete:

- a detector fusing temporal innovation, categorical policy divergence, and
  scoped one-step IBP margin evidence;
- clean episode-level split-conformal calibration with disjoint fit,
  validation, attacker-training, and test cohorts;
- semantic-temporal policy-input purification with a frozen learned proposal;
- a strict caller-attested, consecutive trusted-prefix bootstrap with
  fail-closed uncalibrated warm-up;
- transactional trusted anchors and an explicit legal-action fallback that
  invalidates temporal continuity until explicit rebootstrap;
- a complete P1/P3/P4 non-adaptive and defense-aware audit matrix with
  independent episode-wise worst safety and utility endpoints.

This is an implementation and contract milestone, not an empirical robustness
result. The H1 detector, H2 purifier, and H3 complete-Guard hypotheses still
require frozen public-driving cohorts, converged adaptive attacks, and paired
confidence intervals. A stable SUMO PPO victim is not yet available, so SUMO
empirical effectiveness remains explicitly false. Contract:
[`docs/contracts/p5_rapid_guard.md`](docs/contracts/p5_rapid_guard.md).
Release record: [`docs/releases/P5.md`](docs/releases/P5.md).

The current attack track is **test-time observation evasion**. Action-channel
attacks, reward/rollout poisoning, parameter attacks, and backdoors will be
implemented as separate tracks so their threat models are not mixed.

## Repository rules

1. Core code under `src/rl_attack` must never import from
   `third_party/upstream`.
2. Upstream repositories are checked out at detached, locked commits by
   `scripts/sync_upstream.ps1`.
3. Every upstream repository uses its own environment. Its legacy dependencies
   must not be installed into the core environment.
4. SUMO scenarios and victim policies are immutable versioned artifacts.
5. Evaluation-time normalization statistics are frozen.

## Core setup

```powershell
.\scripts\create_core_env.ps1 -UseLauncher
.\.venv\Scripts\Activate.ps1
```

The setup script creates only `RL_Attack\.venv` and installs the fully resolved
Windows/Python 3.10 lock in `requirements/core-py310-windows.lock.txt`.
`requirements/wcdt-compat.txt` records the source-compatible top-level stack.
The script never changes a WCDT environment. Paper repositories use separate
environments described under `third_party/environments`.

## Smoke experiment

```powershell
rl-attack-baseline --env-id CartPole-v1 --timesteps 50000 --episodes 20
rl-attack-baseline --env-id CartPole-v1 --load-model artifacts/cartpole/model.zip `
  --attack pgd-ce --epsilon 0.02 --steps 20 --restarts 5
rl-attack-defense --method sa_ppo --env-id CartPole-v1 --timesteps 50000 `
  --epsilon 0.02 --attack pgd --attack-steps 10
rl-attack-p4-audit <resolved-p4-config.yaml> --output-dir outputs/p4_stfa_audit
rl-attack-p4-v2c-engineering run `
  configs/experiments/p4_mergelite9_v2c_matched_engineering.yaml `
  --output-dir outputs/p4_v2c_engineering_<commit>_<date>
rl-attack-p4-v2c-engineering verify `
  configs/experiments/p4_mergelite9_v2c_matched_engineering.yaml `
  --run outputs/p4_v2c_engineering_<commit>_<date> `
  --expected-run-manifest-sha256 <printed_sha256>
rl-attack-p4-v2d-prepare prepare `
  configs/experiments/p4_mergelite9_v2d_return_loss_preparation.yaml `
  --output-dir outputs/p4_v2d_return_prepared_<commit>_<date>
rl-attack-p4-v2d-prepare verify `
  configs/experiments/p4_mergelite9_v2d_return_loss_preparation.yaml `
  --preparation outputs/p4_v2d_return_prepared_<commit>_<date> `
  --expected-manifest-sha256 <printed_sha256>
rl-attack-p4-v2d-engineering run `
  configs/experiments/p4_mergelite9_v2d_return_loss_engineering.yaml `
  --output-dir outputs/p4_v2d_return_engineering_<commit>_<date>
rl-attack-p4-v2d-engineering verify `
  configs/experiments/p4_mergelite9_v2d_return_loss_engineering.yaml `
  --run outputs/p4_v2d_return_engineering_<commit>_<date> `
  --expected-manifest-sha256 <printed_sha256>
rl-attack-p4-v2e-prepare prepare `
  configs/experiments/p4_mergelite9_v2e_signed_return_preparation.yaml `
  --output-dir outputs/p4_v2e_signed_prepared_<commit>_<date>
rl-attack-p4-v2e-prepare verify `
  configs/experiments/p4_mergelite9_v2e_signed_return_preparation.yaml `
  --preparation outputs/p4_v2e_signed_prepared_<commit>_<date> `
  --expected-manifest-sha256 <printed_sha256>
# Run only when the full preparation verification reports engineering_unlocked=true.
rl-attack-p4-v2e-engineering run `
  configs/experiments/p4_mergelite9_v2e_signed_return_engineering.yaml `
  --output-dir outputs/p4_v2e_signed_engineering_<commit>_<date>
rl-attack-p4-v2e-engineering verify `
  configs/experiments/p4_mergelite9_v2e_signed_return_engineering.yaml `
  --run outputs/p4_v2e_signed_engineering_<commit>_<date> `
  --expected-manifest-sha256 <printed_sha256>
rl-attack-train-rapid-guard train --help
rl-attack-train-rapid-guard verify --help
rl-attack-p5-audit <resolved-p5-config.yaml> `
  --output-dir outputs/p5_rapid_guard_audit
rl-attack-p5-adaptive-smoke run `
  configs/experiments/p5_mergelite9_adaptive_engineering_smoke.yaml `
  --output-dir outputs/p5_mergelite9_adaptive_smoke_<commit>_<date>
rl-attack-p5-adaptive-smoke verify <run-directory> `
  --expected-manifest-sha256 <printed_sha256>
```

The RAPID-Guard training help lists every mandatory pinned input and expected
digest; there is intentionally no shorthand that bypasses those bindings. The
CLI is a correctness smoke path, not the final statistical experiment runner.
Final results must use frozen victim sets, paired test seeds, and the protocol
in `configs/experiments`.
The adaptive-smoke command is intentionally test-scoped: it executes a real
BPDA/purifier/Guard/environment chain but cannot establish attack strength or
defense effectiveness.
The P4-v2c command is likewise engineering-only. It replaces the failed v2b
absolute timing gates with a clean-derived B3 top-2 schedule and compares
FGSM/PGD/MAD/STFA on the same five seeds and schedule; it cannot establish
statistical significance or attack superiority.
P4-v2d is also claim-ineligible. It trains a dedicated nine-output H=12/R=4
critic only on `E_r[(G_clean-G_a)_+/25]`; failure and safety labels have no
head, loss, or shared-backbone gradient path. A noncausal clean-episode
engineering selector uses predicted return opportunity to choose two feasible
times, while the 20x5 inner solver maximizes categorical expected return loss.
The victim still executes deterministic argmax, so this surrogate mismatch is
reported rather than hidden. Merge failure and safety cost remain report-only
endpoints, and the five-seed scale-up gate cannot be passed by safety
degradation or action flips alone.
P4-v2e is a separate claim-ineligible successor and does not rewrite v2d. Its
offline labels are signed paired short-return differences
`E_r[(G_clean-G_a)/25]` with no clipping or safety/failure mixture. The
clean-trajectory probe selects two fixed times; at each reached time the formal
runtime director reselects a strictly positive non-clean target on the current
local clean observation using the already-paid critic vector. A selected 20x5
solver step has the exact native query vector `107/100/106/1/1=315`
(observation/gradient/projection/critic/director). The preparation adequacy and
real detached-q FLAT solver-gradient gates must both pass before the five
one-shot engineering seeds may be consumed.

## Verification

```powershell
python scripts/check_isolation.py
python scripts/verify_core_lock.py
python -m pytest tests -q
.\scripts\sync_upstream.ps1 -VerifyOnly
```

The final command is expected to report missing checkouts before the optional
paper repositories have been downloaded.
