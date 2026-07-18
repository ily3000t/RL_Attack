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

## P2 — public-task defense reproduction (next)

- Vanilla PPO victim sets on classic control and HighwayEnv;
- Adv-PPO as an explicitly specified adversarial-training baseline;
- SA-PPO and CAR-PPO paper-fidelity checks in their isolated legacy repos;
- clean-room SB3 ports after license review;
- clean-performance/robustness Pareto curves rather than one attack setting.

Acceptance gate: five or more independently trained victims per method, fixed
validation/test splits, and defense selection without test-seed access.

## P3 — strong attack audit

- Q/Safety-Critic PGD;
- strategically timed attacks with a fixed temporal budget;
- PA-AD or ATLA learned attacker;
- action-output attacks on a separate leaderboard;
- adaptive attacks that know the detector/purifier.

P1 attacks are correctness baselines. No defense is called robust until it is
audited by P3 and evaluated with `worst-over-attacks`.

## P4 — proposed method

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

## P5 — SUMO and layered validation

Train/evaluate the same algorithm contracts on `sumo_merge_core_v1`. When WCDT
produces a stable checkpoint, import it as a new immutable victim version.
Only after that compare:

```text
PPO
PPO + Safety Shield
PPO + Safety Shield + ACCVP
```

The layers are victim configurations. They must not fork the core attack code.
