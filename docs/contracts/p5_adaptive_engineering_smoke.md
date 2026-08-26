# P5 adaptive-attack engineering smoke contract

Status: implemented engineering gate. This contract does **not** define a
formal defense-effectiveness experiment.

## Purpose

The smoke closes one executable chain:

`MergeLite9 observation -> frozen 9-action PPO -> fixed-anchor purifier BPDA-PGD -> semantic projection -> real RapidGuard.step -> MergeLite9.step`.

It checks that the P5 interfaces, manifests, threat budget, adaptive-gradient
surrogate, runtime state machine, and two non-exchangeable accounting ledgers
work together. It does not compare defended and undefended returns and cannot
support a robustness or safety claim.

The checked-in configuration is
`configs/experiments/p5_mergelite9_adaptive_engineering_smoke.yaml`. It uses
the isolated CPU environment `RL_Attack_Core_Py310`, two dedicated engineering
seeds (`554100`, `554101`), four PGD steps, and the MergeLite9 v2 ratio-6
projector. Ratio 6 means a per-mutable-feature effective epsilon of 0.3; the
two coupled immutable features remain at epsilon 0.

## Adaptive scope

`BPDAIdentityPurifierAdapter` executes a real fixed-anchor
`SemanticTemporalPurifier` forward pass and uses the identity surrogate only
for its backward pass. Every purifier forward includes a real semantic
projection. The adaptive scope is permanently named
`fixed_anchor_purifier_surrogate_only`.

The following hard paths are outside that gradient surrogate:

- detector gate;
- certificate gate;
- legal fallback gate;
- safety-shield gate.

They are subsequently exercised through the real stateful `RapidGuard.step`
runtime. The smoke uses a deterministic mutable-coordinate L-infinity detector
fixture, disables certification, uses the real legal fallback component, and
does not install a shield. It therefore must never be described as a full
trained RAPID-Guard bundle or an exact end-to-end adaptive gradient.

## Execution gate

Each episode must satisfy all of the following or fail closed:

1. The frozen PPO checkpoint and in-memory policy-state hashes match before
   and after execution.
2. Every mutable input gradient is finite and has a non-zero norm.
3. Every PGD update and the final fixed-point check call the exact ratio-6
   `MergeLite9Projector`.
4. The final perturbation is non-zero, preserves immutable coordinates, and
   remains within the exact per-feature epsilon.
5. The test detector marks the adversarial input suspicious and the real Guard
   executes its purification path.
6. The Guard-selected action is applied to a real MergeLite9 transition.
7. The P4 v2b development gate is pinned as failed and explicitly marked
   `required_for_engineering_smoke=false`; matched and future-final seeds remain
   unconsumed.

Passing these checks means only that the engineering chain is executable and
auditable.

## Accounting

The attacker ledger records actual calls at their call sites:

- victim forward and backward calls;
- fixed-anchor defense-forward and BPDA backward calls;
- BPDA surrogate calls;
- attack-budget projections;
- purifier semantic projections;
- EOT samples (zero for this deterministic adapter).

The defense ledger is copied from `GuardStepResult.accounting` and
`GuardEpisodeAccounting`. Policy, detector, certificate, IBP, proposal,
projection, critic, fallback, shield, and purification-attempt fields retain
their native meanings. The two ledgers are not exchangeable, and their sums
are not presented as one fungible query budget.

## Artifacts and verification

A run writes a fresh directory and never overwrites an existing path:

- `resolved_config.json`;
- `steps.json`;
- `episodes.json`;
- `summary.json`;
- `manifest.json` (written last).

The manifest pins the four immutable external inputs, PPO state, projector,
source files, dependency versions, thread settings, output hashes, P4 failed
gate, seed roles, and all claim flags. The manifest cannot contain its own
digest, so the run command returns the required external manifest SHA-256.

Run from `E:\RL_Attack`:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

.\.venv\Scripts\python.exe -m rl_attack.cli.p5_adaptive_smoke run `
  configs\experiments\p5_mergelite9_adaptive_engineering_smoke.yaml `
  --output-dir outputs\p5_mergelite9_adaptive_smoke_<commit>_20260825
```

Then pass the exact digest printed by the run command:

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p5_adaptive_smoke verify `
  outputs\p5_mergelite9_adaptive_smoke_<commit>_20260825 `
  --expected-manifest-sha256 <printed_sha256>
```

Verification rehashes outputs before parsing and again after scientific checks,
rehashes all external inputs, reloads and freezes the PPO, reconstructs the
projector, closes per-step/per-episode ledgers, and rejects an enabled claim
flag. It does not certify simulator safety or defense effectiveness.

## Claim boundary

The following flags are structurally required to remain false in source YAML,
resolved configuration, summary, manifest, and verification result where
applicable:

- formal evaluation/summary eligibility;
- attack-strength and defense-effectiveness claims;
- full adaptive-defense evaluation;
- exact end-to-end gradient;
- empirical defense effectiveness;
- SUMO effectiveness.

Formal P5 comparison remains blocked until a trained, hash-bound RAPID-Guard
bundle exists and a sufficiently strong attack passes its own preregistered
development gate.
