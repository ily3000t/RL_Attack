# P4 MergeLite9 effect screening

This stage is a non-formal, one-victim-seed gate for the proposed STFA attack.
It uses the repository-owned eight-observation, nine-action `MergeLite9Env`;
it is not SUMO evidence and cannot authorize P5 directly.

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
$prepared = ".\outputs\p4_mergelite9_effect_prepared"
$validationAudit = ".\outputs\p4_mergelite9_effect_validation_audit"
$finalAudit = ".\outputs\p4_mergelite9_effect_final_audit"
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
