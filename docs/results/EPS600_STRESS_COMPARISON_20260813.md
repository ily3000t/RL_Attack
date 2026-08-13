# Ratio-6 Observation-Attack Stress Comparison (Implemented)

## Status

This post-hoc development stress experiment has been completed and verified on
2026-08-13. It compares the already observed `epsilon_ratio=0.5` results with a
new `epsilon_ratio=6.0` run while holding the victims, episode seeds, attack
budgets, and other protocol fields fixed.

The implementation commit is
`3a1b11462e581b144aa022ed245f869442847822` on
`codex/p5-rapid-guard`. The runtime environment was
`RL_Attack_Core_Py310` (`E:\RL_Attack\.venv`), CPU only. Worker Torch threads
were fixed to one.

This evidence is non-formal and post-hoc because the same already inspected
validation seeds were reused for a paired sensitivity comparison. It is not an
independent validation or test cohort.

## Effective-epsilon contract

The ratio is a multiplier, not the direct perturbation radius:

`effective_epsilon[i] = base_epsilon[i] * epsilon_ratio`.

For the new MergeLite9 v2 threat contract, the ratio must be finite and
non-negative and every effective per-feature epsilon must be in `[0, 1]`.
Immutable features retain epsilon zero. The historic v1 contract and hashes
remain unchanged for existing `ratio <= 1` preparations.

| Experiment | Base epsilon | Ratio 0.5 | Ratio 6.0 |
|---|---:|---:|---:|
| CartPole policy input | `[.05,.05,.01,.01]` | `[.025,.025,.005,.005]` | `[.30,.30,.06,.06]` |
| MergeLite9 policy input | `[0,.05,.05,.05,.05,.05,.05,0]` | `[0,.025 x 6,0]` | `[0,.30 x 6,0]` |

The same ratio does not make the two environments physically comparable.
CartPole features use different native units, while the six mutable MergeLite9
features are normalized policy inputs.

## Executed bundles

### CartPole P12 development screening

- Config:
  `configs/experiments/p12_cartpole_development_screening_seed0_eps600.yaml`
- Output:
  `outputs/p12_cartpole_development_screening_eps600_3a1b114_20260813`
- Result: `status=verified`, 20 shards, 1,000 episode rows.
- Design: four seed-0 victims, Random/FGSM/PGD/MAD, 50 paired seeds
  `25000..25049`, four CPU workers with one Torch thread each.
- End-to-end plan/run/verify wall time: approximately 33 minutes 7 seconds.
- Formal eligibility: false (`smoke`, validation cohort, one training seed).

### MergeLite9 P4 v2a validation

- Config:
  `configs/experiments/p4_mergelite9_effect_screening_eps600.yaml`
- Preparation:
  `outputs/p4_mergelite9_effect_v2a_eps600_prepared_3a1b114_20260813`
- Validation audit:
  `outputs/p4_mergelite9_effect_v2a_eps600_validation_3a1b114_20260813`
- Preparation verify: 15 artifacts verified; victim admission passed; all seed
  splits remained disjoint.
- Validation: 50 paired seeds `544000..544049`, `status=complete`.
- Only the validation audit was executed. The final cohort
  `545000..545049` and the final-only analyzer were not run.
- Claim boundary: synthetic MergeLite9, one PPO training seed, no matched
  baseline comparison, no direct P5 authorization.

## CartPole results

Clean mean returns were unchanged from the ratio-0.5 run:
Vanilla/Adv/SA PPO all scored 500.00 and CAR-PPO scored 462.86.

The table reports paired return drop (`clean - attacked`); a positive value
means the attack reduced return. Confidence intervals are the checked bundle's
1,000-replicate paired episode bootstrap intervals.

| Victim | Attack | Drop at ratio 0.5 | Attacked return at ratio 6 | Drop at ratio 6 (95% CI) | Flip rate at ratio 6 |
|---|---|---:|---:|---:|---:|
| Vanilla | Random | 0.00 | 500.00 | 0.00 `[0.00,0.00]` | 26.97% |
| Vanilla | FGSM | 0.00 | 500.00 | 0.00 `[0.00,0.00]` | 50.00% |
| Vanilla | PGD | 0.00 | 500.00 | 0.00 `[0.00,0.00]` | 50.00% |
| Vanilla | MAD | 0.00 | 498.14 | 1.86 `[0.00,5.58]` | 50.00% |
| Adv-PPO | Random | 0.00 | 500.00 | 0.00 `[0.00,0.00]` | 22.62% |
| Adv-PPO | FGSM | 0.00 | 302.12 | 197.88 `[189.24,205.82]` | 49.72% |
| Adv-PPO | PGD | 0.00 | 302.12 | 197.88 `[189.20,205.84]` | 49.72% |
| Adv-PPO | MAD | 0.00 | 201.94 | 298.06 `[277.82,320.82]` | 48.81% |
| SA-PPO | Random | 0.00 | 500.00 | 0.00 `[0.00,0.00]` | 18.70% |
| SA-PPO | FGSM | 0.00 | 268.78 | 231.22 `[221.92,241.92]` | 49.52% |
| SA-PPO | PGD | 0.00 | 263.28 | 236.72 `[227.22,247.60]` | 49.50% |
| SA-PPO | MAD | 0.00 | 171.12 | 328.88 `[299.08,355.58]` | 47.07% |
| CAR-PPO | Random | 0.16 | 447.12 | 15.74 `[-15.23,43.84]` | 15.12% |
| CAR-PPO | FGSM | -7.14 | 95.78 | 367.08 `[347.26,383.34]` | 45.50% |
| CAR-PPO | PGD | -7.14 | 95.76 | 367.10 `[347.64,385.82]` | 45.49% |
| CAR-PPO | MAD | -10.18 | 414.54 | 48.32 `[8.19,89.78]` | 35.97% |

At ratio 0.5, no baseline attack reliably reduced return: Vanilla, Adv-PPO,
and SA-PPO remained at the 500-point ceiling, while gradient attacks improved
CAR-PPO. At ratio 6, FGSM/PGD/MAD strongly reduced Adv-PPO and SA-PPO, and
FGSM/PGD strongly reduced CAR-PPO. Vanilla PPO nevertheless remained almost
entirely at the ceiling even when about half of its actions flipped.

This is evidence that the high-budget baseline attacks can be effective, but it
does not validate the current defenses. In this single-seed CartPole screen,
the nominally defended victims were substantially more vulnerable than the
Vanilla victim. The ceiling and task simplicity prevent interpreting Vanilla as
generally robust.

FGSM used one gradient evaluation and three internal policy queries per
attacked step. PGD/MAD used 100 gradient evaluations and 106 policy queries per
attacked step. FGSM tied PGD on Adv-PPO and nearly tied it on CAR-PPO, so the
100-step-equivalent solver cost did not consistently provide more damage.

## P4 v2a results

The two P4 audits use the same 50 episode seeds, and their clean returns match
exactly seed by seed.

| Metric | Ratio 0.5 | Ratio 6.0 |
|---|---:|---:|
| Maximum effective L-inf | 0.025 | 0.30000004 |
| Mean clean return | 11.62457 | 11.62457 |
| Mean attacked return | 11.23087 | 11.18019 |
| Mean paired return drop | 0.39370 (3.39%) | 0.44439 (3.82%) |
| Exploratory mean bootstrap 95% CI | `[0.0051,1.0246]` | `[0.0841,1.0457]` |
| Median paired drop | 0.00000 | 0.08320 |
| 10% trimmed mean drop | 0.01361 | 0.08787 |
| 10% winsorized mean drop | 0.02654 | 0.10271 |
| Harmed / improved / unchanged episodes | 18 / 12 / 20 | 30 / 19 / 1 |
| Selected attack steps | 102 / 1,323 | 103 / 1,322 |
| Action flip / selected | 41.18% | 98.06% |
| Target hit / declared | 37.25% | 94.17% |
| Total audited queries | 34,812 | 35,136 |
| Merge success, clean to attacked | 46 to 45 | 46 to 45 |
| Collision episodes, clean to attacked | 0 to 0 | 0 to 0 |
| Near-miss episodes, clean to attacked | 7 to 6 | 7 to 7 |
| Mean safety-cost increase per episode | 0.07630 | 0.14055 |

The direct same-seed dose difference was only `+0.05068` mean return drop even
though epsilon increased twelve-fold. Its paired bootstrap interval
`[-0.07208, 0.14478]` includes zero. Thirty seeds were more damaged at ratio 6,
13 were less damaged, and seven were unchanged.

The ratio-6 result is less degenerate than ratio 0.5: its trimmed and
winsorized means are positive and more episodes are harmed. However, the same
episode (`544043`, drop 13.1973) still supplies 59.4% of the net mean effect.
Removing the largest episode reduces the ratio-6 mean drop to 0.1841; removing
the two largest reduces it to 0.1435.

The attack ledger was internally consistent. The ratio-6 audit recorded 1,322
steps, 103 selected/nonzero steps, 101 flips, 97 target hits, and 35,136 total
queries. All hard temporal constraints passed. The observed L-inf excess above
the float32 0.3 record was approximately `3e-8`, below the checked `1e-6`
numerical tolerance and not a budget violation.

## Combined interpretation

The experiments answer two different questions and must not be ranked by raw
return or by the shared ratio number:

1. CartPole uses dense attacks on a two-action task and tests four different
   PPO checkpoints. At ratio 6, the baseline gradient attacks clearly damage
   three nominally defended checkpoints.
2. P4 uses a temporally sparse learned director and a nine-action MergeLite9
   victim. At ratio 6, STFA almost always flips to its intended target, but the
   additional trajectory-level harm over ratio 0.5 is small and statistically
   uncertain.

Both results show why action flip is only a mechanism metric. Vanilla CartPole
can flip on roughly half its attacked steps without losing return; P4 can hit
94% of declared targets without producing a correspondingly large increase in
merge failures, collisions, or near misses.

## Decision for P4 and P5

### P4: continue optimization

P4 passes the observation-attack mechanism gate but not the stable
trajectory-harm gate. Increasing epsilon further is not the priority. The next
P4 version should:

1. optimize predicted multi-step return loss, merge-failure probability, or
   cumulative safety cost rather than primarily an instantaneous target action;
2. train the director on risk-to-go or short counterfactual rollouts so timing
   reflects downstream consequences;
3. retain exact target hit as a constraint, with factor objectives only as
   secondary regularization;
4. compare STFA with Random/FGSM/PGD/MAD on the same MergeLite9 victim, episode
   seeds, selected-step schedule, epsilon profile, and query budget;
5. repeat on multiple independently trained victim seeds before consuming any
   final cohort.

### P5: do not start formal effectiveness validation yet

P5 may run a small engineering smoke test to confirm the defense API, logging,
and adaptive-attack path. It should not yet be presented as evidence that the
proposed defense works. A defense evaluated against an attack whose
trajectory-level objective is still weak can appear robust for the wrong
reason.

Formal P5 evaluation should begin only after P4 demonstrates stable harm across
victim seeds, is compared with matched baselines, and is no longer dominated by
one or two episodes. The CartPole result also requires diagnosing why Adv/SA/CAR
were more vulnerable than Vanilla at high epsilon before those methods are used
as positive defense baselines.
