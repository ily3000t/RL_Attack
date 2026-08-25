# P4-B5 matched-stage execution contract

P4-B5 executes the frozen MergeLite9 victim prepared by P4-B4. It is a
single-victim development experiment, not SUMO evidence and not a formal
robustness claim. B5 never trains a model and never opens the B2/B3 training
datasets or counterfactual-oracle labels.

## Trust root and stage order

The immutable trust root is:

- preparation directory:
  `outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825`
- preparation manifest SHA-256:
  `f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0`

Every run first calls the B4 verifier in the same process. B5 subsequently
opens artifacts only through the ten-role `executable_artifacts` allowlist.
Each open is protected by path-identity, reparse-point, byte-size and SHA-256
checks before and after access. The four offline dataset roles have no B5 path
surface.

The stage order is fail-closed:

1. `development_validation` consumes exactly seeds `550000..550049` and runs
   clean, STFA fixed schedule, and STFA online-secondary.
2. The independent result verifier rebuilds schedules, outcomes, query totals,
   paired statistics and the gate from raw files.
3. `matched_baseline` may consume exactly seeds `551000..551049` only when a
   byte-pinned, production development result passes the preregistered gate.
4. Seeds `552000..552049` remain reserved future-final seeds and are rejected.

There is no command-line override for seeds, conditions, epsilon, victim,
device, thread count, bootstrap settings or overwrite.

## Clean-derived fixed schedule

For every episode seed, B5 first performs one deterministic clean rollout.
Each clean pre-action row records only online information:

- frozen PPO categorical probabilities;
- one frozen B2 predicted composite-risk vector;
- the all-nine-actions availability mask.

The top-three non-clean actions are ranked by PPO probability with lower action
index as the tie break. The target is the reachable action with highest B2
risk. Opportunity is
`max(predicted_target_risk - predicted_clean_action_risk, 0)`, with a minimum
of `0.05`. Global greedy selection uses opportunity descending, step ascending
and row ascending, then replays the exact `K=8`, `min_gap=2`,
`window_size=16`, `window_k=2` `TemporalBudgetLedger` over steps `0..63`.

Random, FGSM, PGD, MAD and STFA-fixed share the same byte-hashed step/target
schedule. No attacked trajectory can feed back into schedule construction.
The online-secondary condition uses B3 for timing and is explicitly not a
fixed-schedule or query-matched comparison.

## Seven conditions and query currencies

The matched matrix is:

1. clean
2. Random fixed schedule
3. FGSM fixed schedule
4. PGD-20x5 fixed schedule
5. MAD-20x5 fixed schedule
6. STFA-v2b fixed schedule
7. STFA-v2b online-secondary

The ratio-6 MergeLite9 v2 projector is used by every attack. Its effective
mutable-feature bound is approximately `0.3`; route progress and merge urgency
remain immutable. PGD and MAD use 20 steps, 5 random restarts and per-feature
step size `2*epsilon/20`. STFA uses the existing legacy solver through a small
B5 adapter that injects the frozen direct-Torch RNG seed without changing the
B1-B4 modules.

The native per-applied-attack currencies, excluding the logical clean-schedule
charge, are:

| Condition | Observation | Gradient | Projection | B2 critic | B3 director | Total |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0 | 0 | 1 | 0 | 0 | 1 |
| FGSM | 3 | 1 | 1 | 0 | 0 | 5 |
| PGD-20x5 | 107 | 100 | 106 | 0 | 0 | 313 |
| MAD-20x5 | 107 | 100 | 106 | 0 | 0 | 313 |
| STFA fixed | 107 | 100 | 106 | 1 | 0 | 314 |
| STFA online selected | 107 | 100 | 106 | 1 | 1 | 315 |

`total_queries` is the unweighted sum of the five currencies. Transform
queries are separately recorded and must be zero. Random and FGSM receive no
dummy calls. Clean-schedule construction is physically charged once per seed;
its full observation+B2 cost is logically charged to each fixed condition.

## Paired statistics and gates

B5 owns a fixed statistics contract:

- paired unit: episode seed, in ascending order;
- bootstrap RNG: NumPy PCG64 seed `553001`, ascending registered seeds,
  endpoint disabled and little-endian `<i8` C-order index bytes;
- resamples: `10000` shared index rows across all conditions;
- interval: two-sided 95% percentile interval using NumPy linear quantiles.

For each attacked/clean pair, realized composite risk is:

```text
positive(clean discounted return - attacked discounted return) / 25
+ positive(attacked merge failure - clean merge failure)
+ positive(attacked cumulative safety cost - clean cumulative safety cost) / 10
```

Merge failure means terminal or time-limit completion without merge success.
Horizon exhaustion and a test-scope cutoff are not failure events.

The development gate requires the lower 95% paired-bootstrap bound for STFA
fixed realized composite risk to be greater than zero. The matched superiority
gate additionally requires the lower bound of STFA fixed minus the strongest
per-resample Random/FGSM/PGD/MAD mean to be greater than zero. The observed
point difference uses the strongest observed baseline mean. Test-scope runs
are always claim-ineligible regardless of their raw numbers.

## Outputs and verification

Every fresh, atomically published run contains exactly:

- `resolved_stage_config.json`
- `schedules.json`
- `steps.json`
- `episodes.json`
- `summary.json`
- `manifest.json`

The runner has no overwrite option. The output cannot equal, contain, or be
contained by the immutable preparation directory. Production publication is
preceded by an independent recomputation from raw schedule and step records.
The public verifier repeats the B4 verification, rehashes executable inputs,
reloads the frozen victim, and deterministically replays every condition/seed
through MergeLite9. It requires every saved local clean action and executed
action to equal the frozen PPO argmax on the corresponding clean and
adversarial observations, respectively. It also recomputes outcomes,
native/logical queries, paired bootstrap results and gates, and rejects
unregistered or changed result files.

## Commands

Run development validation:

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2b_matched run `
  outputs\p4_mergelite9_v2b_prepared_7d0b72f_20260825 `
  --expected-manifest-sha256 f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0 `
  --stage development_validation `
  --output-dir outputs\p4_v2b_development_B5_<fresh-id>
```

Verify that result using the manifest SHA printed by the run:

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2b_matched verify `
  outputs\p4_mergelite9_v2b_prepared_7d0b72f_20260825 `
  --expected-manifest-sha256 f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0 `
  --run outputs\p4_v2b_development_B5_<fresh-id> `
  --expected-run-manifest-sha256 <development-manifest-sha256>
```

Only if the verified development gate passes, run matched baselines:

```powershell
.\.venv\Scripts\python.exe -m rl_attack.cli.p4_v2b_matched run `
  outputs\p4_mergelite9_v2b_prepared_7d0b72f_20260825 `
  --expected-manifest-sha256 f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0 `
  --stage matched_baseline `
  --development-result outputs\p4_v2b_development_B5_<fresh-id> `
  --expected-development-manifest-sha256 <development-manifest-sha256> `
  --output-dir outputs\p4_v2b_matched_B5_<fresh-id>
```

No 50-seed development or matched stage was executed while implementing B5.
