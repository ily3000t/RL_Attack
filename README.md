# RL Attack

Independent PPO attack-and-defense research benchmark built around Gymnasium,
Stable-Baselines3, public driving environments, and a frozen SUMO highway-merge
scenario.

This repository is intentionally independent from `WCDT_ACCVP_Attack`.
The WCDT project is treated as a read-only source of versioned scenarios and
victim-policy bundles. No runtime import from that repository is allowed.

## Current milestone

Phase P0/P1:

- versioned threat-model and perturbation contracts;
- categorical SB3 policy adapter;
- Random-Uniform and Random-Sign observation attacks;
- FGSM-CE and PGD-CE observation attacks;
- Categorical-MAD with random-start PGD;
- single-environment evaluation wrapper and runner;
- frozen `sumo_merge_core_v1` scenario provenance;
- locked, isolated upstream research repositories.

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
```

The CLI is a correctness smoke path, not the final statistical experiment
runner. Final results must use frozen victim sets, paired test seeds, and the
protocol in `configs/experiments`.

## Verification

```powershell
python scripts/check_isolation.py
python scripts/verify_core_lock.py
python -m pytest tests -q
.\scripts\sync_upstream.ps1 -VerifyOnly
```

The final command is expected to report missing checkouts before the optional
paper repositories have been downloaded.
