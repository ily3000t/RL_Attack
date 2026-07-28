# Implementation milestones

## P0 — repository and contracts (implemented)

- independent Git history and one-way source snapshots;
- isolated core and paper-code dependency policy;
- categorical policy protocol and SB3 adapter;
- explicit test-time observation threat model;
- paired-seed evaluation path and attack accounting.

## P1 — attack baseline (implemented)

- Clean, Random-Uniform, Random-Sign;
- FGSM-CE and random-start PGD-CE;
- random-start Categorical-MAD PGD;
- per-feature epsilon, valid bounds, and immutable-feature mask;
- policy-query and gradient-evaluation accounting.

Acceptance gate: attacks pass projection/reproducibility tests, clean and
attacked evaluation use the same victim/seed cohort, and PGD/MAD are not weaker
than their clean objective in deterministic unit cases.

## P2 — public-task defense implementation (implemented)

- Vanilla PPO victim sets on classic control and HighwayEnv;
- Adv-PPO as an explicitly specified adversarial-training baseline;
- SA-PPO and CAR-PPO fidelity boundaries recorded against isolated references;
- maintained clean-room SB3 objective ports with no legacy runtime imports;
- one-step IBP greedy-action certification for supported MLP actors;
- clean-performance/robustness Pareto curves rather than one attack setting.

Implementation gate: unit/integration regression, explicit fidelity metadata,
isolated dependencies, and auditable checkpoint manifests. The statistical
experiment gate remains five or more independently trained victims per method,
fixed validation/test splits, and defense selection without test-seed access.

## P3 — reproduced strong-attack audit (implementation completed)

- Robust-Sarsa critic/attack semantics from the locked SA-PPO reference, plus
  the maintained P1 categorical MAD-PGD baseline;
- maintained clean-room PA-AD actor/director training, with the unresolved-
  license upstream repository retained as reference-only evidence;
- maintained adapters that evaluate frozen SB3 victims without importing
  legacy repositories into the core package;
- executable, budget-matched victim/epsilon/seed sweeps with
  `worst-over-attacks`;
- explicit paper-code versus clean-room fidelity labels.

P1 attacks are correctness baselines. No defense is called robust until it is
audited by the reproduced P3 attacks.

Implementation status: the maintained categorical Robust-Sarsa adaptation,
stochastic-PAMDP PA-AD path, victim-bound checkpoints, learned-attacker
training CLI, and paired executable audit are implemented and tested. The
statistical gate remains pending: fixed multi-seed P2 victims must be used to
train per-victim attack artifacts and run the frozen test matrix before any
robustness ranking is claimed.

## P4 — proposed strong attack

Semantic, Temporally-budgeted, Factorized 9-action attack (working name
`STFA-9`):

- outer director selects attack time and lateral/longitudinal target;
- inner projected optimizer uses physical units and semantic consistency;
- safety-cost critic targets collision/near-miss/TTC/merge failure;
- discrete vehicle-field operations are charged separately from continuous
  perturbations.

Required ablations: semantic vs plain norm ball, random vs learned timing,
flat-nine vs factorized action target, CE/MAD vs safety-Q objective, and
non-adaptive vs defense-aware attacks.

## P5 — proposed defense

Risk-Aware Policy-Invariance Defense (working name `RAPID-Guard`):

- detector fuses temporal innovation, categorical policy divergence, and IBP
  action-margin evidence;
- state purifier projects suspicious observations onto a frozen
  semantics/temporal feasible set;
- uncertainty-calibrated gate selects purified policy input or a minimal
  safety fallback;
- training exposes the detector and purifier to P3 and P4 adaptive attacks;
- thresholds are selected on validation cohorts without test-seed access.

Required ablations: detector channels, purifier, adversarial training,
certificate signal, fallback layer, adaptive white-box attack, clean
performance cost, latency, and worst-over-attacks.

## P6 — SUMO and layered validation

Train/evaluate the same algorithm contracts on `sumo_merge_core_v1`. When WCDT
produces a stable checkpoint, import it as a new immutable victim version.
Only after that compare:

```text
PPO
PPO + Safety Shield
PPO + Safety Shield + ACCVP
```

The layers are victim configurations. They must not fork the core attack code.
