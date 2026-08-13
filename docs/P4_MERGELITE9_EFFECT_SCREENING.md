# P4 MergeLite9 reachability-aware effect screening v2a

This stage is a non-formal, one-victim-seed gate for the proposed STFA attack.
It uses the repository-owned eight-observation, nine-action `MergeLite9Env`;
it is not SUMO evidence and cannot authorize P5 directly.

The original v1 validation was a NO-GO: all 117 selected perturbations reached
the `0.025` L-infinity bound, but only 20 changed the victim action and only one
hit the director target. Mean paired return drop was `-0.03198`, so the attack
did not harm the victim. The frozen v1 final cohort was never run or inspected.
The failure analysis found two coupled contract problems: director training
used argmax one-hot policy features while online inference used softmax, and
targets were ranked for harm without a policy-boundary reachability proxy.

v2a changes only those two factors. Critic Bellman continuation remains
deterministic argmax one-hot. Director training and inference both use the
frozen PPO categorical softmax, and target labels/online decoding are limited
to the three highest-probability available non-clean actions. Within that mask,
labels maximize normalized positive safety-harm advantage multiplied by the
target-to-clean probability ratio. This is a validation hypothesis, not a
positive result claim.

## Ratio-6 post-hoc stress rerun

`configs/experiments/p4_mergelite9_effect_screening_eps600.yaml` registers a
separate, non-formal sensitivity rerun of the v2a validation design. It changes
only the protocol name and `epsilon_ratio` from `0.5` to `6.0`; the victim,
training/data sizes, temporal budget, solver settings, reachability rule and
seed cohorts remain fixed. With the trusted base scale `0.05`, its effective
per-feature epsilon is `[0, .3, .3, .3, .3, .3, .3, 0]`. The contract bounds
the effective epsilon values to `[0, 1]`; the dimensionless ratio itself need
not be at most one.

Because this post-hoc stress run intentionally reuses the prior validation
seeds, it supports only a paired epsilon-sensitivity comparison. It is not an
independent validation result, may not strengthen the evidence tier, and must
not run or consume the registered final cohort. Only `prepare`, `verify` and
the generated validation audit are in scope; the formal final audit and
`analyze` command remain out of scope. This section specifies the experiment
before execution and makes no result claim.

The checked protocol is
`configs/experiments/p4_mergelite9_effect_screening.yaml`. It fixes:

- PPO: 150,000 steps, seed `541001`, CPU/Torch threads `1`, `n_steps=512`,
  `batch_size=128`, `n_epochs=10`, learning rate `3e-4`;
- victim admission: 50 held-out paired seeds, return advantage over uniform
  random at least `2.0`, merge success at least `0.60`, collision at most
  `0.10`, and at least two deterministic victim actions;
- critic/director: 200 disjoint episodes each and 1,500 training steps each;
- observation budget: trusted per-sensor epsilon
  `[0, .025, .025, .025, .025, .025, .025, 0]` at ratio `0.5`; sensor
  indices `0/7` are immutable and the nominal maximum L-infinity is `0.025`;
- temporal budget: `H=64`, `K=8`, `min_gap=2`, `window=16`, `window_k=2`;
- STFA: full objective, learned online director, 20 steps and 5 restarts;
- reachability: exact train/runtime PPO-softmax parity and deterministic top-3
  available non-clean target filtering; ties use the lower action index;
- attack validation: 50 disjoint seeds `544000..544049`; only this split may
  be inspected while tuning;
- final screen: 50 paired held-out seeds `545000..545049` and exactly 10,000
  bootstrap resamples with seed `546001`.

Run from a clean fixed commit with the isolated `RL_Attack_Core_Py310`
environment:

```powershell
Set-Location E:\RL_Attack
$python = ".\.venv\Scripts\python.exe"
$protocol = ".\configs\experiments\p4_mergelite9_effect_screening.yaml"
$prepared = ".\outputs\p4_mergelite9_effect_v2a_prepared"
$validationAudit = ".\outputs\p4_mergelite9_effect_v2a_validation_audit"
$finalAudit = ".\outputs\p4_mergelite9_effect_v2a_final_audit"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

& $python -m rl_attack.cli.p4_effect_screening prepare $protocol `
  --output-dir $prepared
& $python -m rl_attack.cli.p4_effect_screening verify $prepared
& $python -m rl_attack.cli.p4_audit `
  "$prepared\p4_mergelite9_effect_validation_audit.yaml" `
  --output-dir $validationAudit --device cpu --torch-threads 1
# Inspect validation only. If tuning is needed, use a new preparation directory.
& $python -m rl_attack.cli.p4_audit `
  "$prepared\p4_mergelite9_effect_final_audit.yaml" `
  --output-dir $finalAudit --device cpu --torch-threads 1
& $python -m rl_attack.cli.p4_effect_screening analyze `
  $prepared $finalAudit
```

Preparation refuses every non-empty destination and provides no recursive
overwrite operation. The critic dataset covers all nine actions. Director
labels use exact private-latent counterfactual costs only on the disjoint
training cohort; latent state is never added to PPO or audit observations.
Every forced factor-coverage target differs from that row's clean victim
action.

Preparation emits a stable SHA-256 preparation contract over the protocol,
seed registry, source state, runtime contracts, victim state and every
training/data/projector artifact. Both audit configs carry the same
conservative screening claim context and bind that preparation SHA. The
validation and final configs differ only in name/path and their disjoint seed
split. Once the final audit is first run, its configuration and seeds are
single-use: a failed result must not be repaired and rerun on the same final
split.

The v2 director dataset sidecar binds the exact softmax feature source, its
contract SHA-256 and `reachable_top_k=3`. Loading and training recompute the
softmax from the pinned PPO, reconstruct every reachable mask, and compare the
director config to the dataset binding. The critic dataset independently
recomputes its deterministic one-hot continuation probabilities; the two
probability semantics cannot be silently substituted for each other.

Running the final audit consumes that cohort whether the gate passes or fails.
A pass may only feed the already registered matched-baseline comparison and
must not be used to tune STFA. A failure requires a new protocol/version and
entirely new validation and final seed cohorts; the consumed seeds are never
reused for a repaired final run.

`analyze` first performs the full preparation verification, rejects input
aliasing and unbound/fake audit directories, binds all four official output
files to the prepared final config, then reconstructs hard-budget ledgers,
all six query counters, discrete costs and attack rates from step rows. It has
no episode/bootstrap/output override: output is written once to
`<FINAL_AUDIT_DIR>\effect_gate.json`. A passing gate advances only to the
matched Random/FGSM/PGD/MAD P4 comparison. P5 remains blocked until that
comparison also succeeds.

The `analyze` command intentionally exposes no device or thread override. It
forces CPU loading and the protocol-owned one-thread Torch runtime before
reloading the preparation bundle.

Preparation rechecks the Git source immediately before publishing its complete
manifest. The official audit and analyzer must both run from the same clean
commit and pinned Python/platform/Torch runtime; the audit execution contract
is CPU with one Torch intra-op and one inter-op thread. The analyzer repeats
its own clean-source check immediately before writing `effect_gate.json` and
records that analysis provenance in the result.
