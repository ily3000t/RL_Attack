# P5 RAPID-Guard contract

Status: implementation and contract machinery implemented（已落实）. This
status is not an empirical robustness result. Public-driving effectiveness,
SUMO effectiveness, and superiority over the P2 defenses remain unestablished
until their frozen statistical gates are run.

## Scope and decision path

Risk-Aware Policy-Invariance Defense (`RAPID-Guard`) is a stateful,
test-time observation defense around a frozen categorical PPO policy. It does
not change simulator state, PPO parameters, rewards, or rollout data. One
episode owns one Guard state:

```text
current policy input + previous trusted history
        |
        v
three-channel detector
        |
        +-- accepted ----------------------> victim action
        |
        +-- suspicious --> purifier --> re-detect --> one-step IBP check
                                                |
                                                +-- accepted --> victim action
                                                |
                                                +-- rejected --> legal fallback
        |
        v
optional, separately bound safety-shield arbitration
```

The three detector channels are:

- temporal innovation from a three-frame, within-episode window containing two
  consecutive trusted frames followed by the current policy input;
- Jensen--Shannon divergence between current and trusted-reference categorical
  policy distributions;
- the non-negative deficit of the clean greedy action's IBP logit margin.

A deterministic logistic head fuses an explicitly declared channel subset.
It is fitted on an attack-exposed `fit` cohort containing clean samples and
declared P3/P4 families. A separate clean `calibration` cohort sets an
episode-maximum split-conformal threshold. This threshold controls a clean
episode-level false-alarm interpretation only under the usual exchangeability
assumption. It does not guarantee attack detection, return, or safety.

## Trusted-anchor transaction

The detector always scores step `t` against state committed before step `t`.
The current observation is never inserted into the anchor or history before
the decision. A complete, validated pass-through or accepted purified path may
commit a new trusted anchor. Suspicious inputs, fallback decisions, component
failures, and failed shield arbitration leave the old anchor unchanged.

The anchor and history reset at every episode boundary. Clean, attacked,
non-adaptive, and defense-aware paired rollouts own independent Guard
instances. They must never share an anchor, cached distribution, history, RNG
state, or mutable environment wrapper. These rules are necessary to prevent
circular detection and slow-drift poisoning from becoming an evaluation
artifact.

Required anchor stress tests include:

- first observation and missing-anchor behavior;
- episode reset and step-order enforcement;
- suspicious-observation rejection;
- repeated attacks and gradual drift;
- purifier, certificate, fallback, and shield exceptions;
- rollback before any externally visible state mutation;
- branch isolation for paired clean and attacked rollouts.

The history bootstrap is part of the calibrated detector, not an unbound
runtime convenience. The maintained runtime accepts only a caller-attested,
attack-free prefix with at least two strictly consecutive trusted frames from
the current episode. It never constructs a repeated-first window such as
`[anchor, anchor, current]`. No-history and single-frame decisions fail closed
before victim-policy or IBP queries, use the legal fallback, and are reported
as an uncalibrated warm-up path outside H1 coverage. A rejected/fallback step
keeps the last trusted anchor but breaks temporal continuity; calibrated
detection remains disabled until an explicit trusted-prefix rebootstrap. The
bootstrap and anchor-transition contracts are hash-bound to the deployed
bundle.

## Purification and fallback boundaries

The purifier intersects a frozen P4-compatible semantic projector with a
coordinate-wise temporal envelope around the previous trusted policy input. It
evaluates a deterministic line-search grid from the minimum envelope repair
toward a trusted or learned proposal. An optional residual denoiser is only a
frozen proposal transform. Its output must still pass the semantic projection,
re-detection, and configured action-invariance check.

The resulting guarantee is exactly:

```text
policy_input_schema_and_temporal_envelope_only
```

It does not prove dynamic consistency, geometric feasibility, or that a
simulator could have produced the purified vector. Projector type, frozen
fields, proposal checkpoint, data, preprocessing, and configuration are
fingerprinted. Mutation or a binding mismatch fails closed.

Fallback always selects a legal discrete action. A fully verified,
action-wise safety-cost critic may choose the minimum proxy cost among legal
actions. A merely present or failed critic is not trusted: the Guard uses an
explicit deterministic static action order instead. Critic scores are proxies,
not ground-truth safety, and the fallback guarantee is limited to:

```text
legal_action_selection_only
```

Fallback count, duration, chosen action, critic availability, critic query,
static-fallback reason, return cost, and safety outcome remain observable
metrics. A fallback or optional shield may not consume future simulator state
or an oracle unavailable to the victim.

## IBP claim boundary

For supported feed-forward SB3 categorical MLP actors, IBP can prove only:

```text
lower(logit_clean_greedy) >
    max(upper(logit_competitor))
```

inside the configured, clipped policy-input box. The epsilon, observation
space, normalization, victim policy state, and preprocessing path must be
frozen and bound. Unsupported actors return `unavailable`; unavailability is
not a stable certificate and may not silently become a zero-risk detector
channel.

This one-step result does not certify:

- stochastic action samples;
- the detector, purifier, hard gate, fallback, or safety shield;
- a trajectory, episode return, collision avoidance, or closed-loop dynamics;
- physical realizability of an input;
- robustness outside the exact certified box.

Certificate coverage uses all eligible decision opportunities as its
denominator, including abstentions and unsupported cases. Reporting coverage
only among successful attempts is forbidden.

## Frozen training and split isolation

The maintained training CLI consumes immutable raw `.npz` files with adjacent
strict-JSON sidecars and `allow_pickle=False`. Raw datasets may contain
observations, clean targets, trusted references, within-episode three-frame
windows, attack-family labels, and episode/scenario identifiers. Cached policy
probabilities, logits, IBP bounds, and detector channels are forbidden. They
are recomputed from the exact frozen victim at fit time.

The split hierarchy is:

```text
train          detector/denoiser fit and defense attack exposure
validation     clean conformal calibration and all defense/model selection
attacker_train per-defense adaptive attacker fit
test           one frozen final audit; never used for fitting or selection
```

Episode and scenario cohorts are mutually disjoint. A temporal window cannot
cross an episode, scenario, or split boundary. Normalization statistics,
innovation scales, denoiser targets, detector preprocessing, fallback
configuration, thresholds, and stopping choices are derived without test
access.

Every adaptive attacker is trained for one frozen victim and one frozen
defense bundle. Its strict manifest must bind the attacker-training episode and
scenario cohort, attack hyperparameters, spatial/temporal/query budgets,
victim, defense, detector threshold, anchor contract, purifier, certificate,
fallback, and optional shield. The final audit pins that manifest and rejects
an incomplete or mismatched binding.

Runtime construction accepts a verified complete bundle, not a loose detector
artifact plus caller-selected components. Before an episode begins it
cross-checks the deployed detector preprocessing and history bootstrap,
semantic projector, temporal envelope, proposal transform, anchor rule,
certificate, fallback/critic, and optional shield against their bundle hashes.
A component that is individually well formed but belongs to another bundle is
still rejected. Replacement tests are required for every component binding.

## Falsifiable hypotheses

The proposed method is evaluated through three pre-registered hypotheses.
None is established by unit tests, a training manifest, or a synthetic
integration run.

### H1 — detector

At a validation-selected clean false-positive operating point, the fused
detector improves held-out test attack detection over each active single
channel. Report episode-aware TPR, FPR, precision-recall, ROC, and calibration
results by seen and unseen attack family.

H1 fails if the pre-registered clean false-positive limit is exceeded, the
paired confidence interval for the chosen detector endpoint does not clear its
registered baseline, or performance is confined to attack families used for
fusion fitting.

### H2 — purifier

On detected, held-out attacked observations, semantic-temporal purification
improves a pre-registered recovery endpoint, such as frozen-policy clean-action
agreement or a fixed safety-cost metric, relative to no purification and to
the minimum-envelope repair. It must also satisfy a registered clean
distortion and clean-performance limit.

H2 fails if improvement disappears under defense-aware attacks, the confidence
interval includes the no-improvement boundary, or clean distortion,
intervention, fallback, or task cost exceeds its registered limit. A lower
denoiser training loss alone cannot establish H2.

### H3 — complete Guard

Against the complete P1/P3/P4 matrix, the full Guard improves both a registered
worst-case safety endpoint and a registered worst-case utility endpoint over
Vanilla PPO and the validation-selected P2 baseline, subject to clean-cost and
latency limits.

H3 fails if any required attack cell is absent or invalid, either endpoint
misses its registered effect/CI threshold, the clean-cost or latency constraint
is violated, or a converged defense-aware attack removes the claimed gain.
Safety and utility are independent endpoints; success on one may not substitute
for failure on the other.

All thresholds, non-inferiority margins, effect directions, bootstrap
replicates, and multiple-comparison correction must be frozen before test
rows are opened. The maintained offline audit currently distinguishes
implementation evidence from statistical evidence; point summaries alone do
not pass H1--H3.

## Attack matrix and gradient-masking audit

Every frozen test seed requires the following complete 13-cell matrix:

```text
Clean

FGSM / PGD / MAD /
Robust-Sarsa / PA-AD / STFA
    x non_adaptive / defense_aware
```

P1 attacks remain correctness and local-gradient baselines. P3 attacks prevent
weak-attack overestimation. P4 STFA supplies a semantic, temporally budgeted
safety-oriented attack. `Defense-aware` means that the attacker has declared
access to the frozen defense, not merely the victim; the audit requires actual
defense forward or backward query evidence.

Hard detector, certificate, fallback, and shield gates are non-differentiable.
The shipped BPDA adapter covers a fixed-anchor purifier surrogate only and
explicitly is not an exact end-to-end gradient. A robustness result therefore
requires:

- direct gradients through every supported differentiable path;
- an explicit BPDA surrogate for the purifier and declared hard-gate
  treatment;
- genuine multi-sample EOT for any randomized component;
- transfer and score-query checks that do not depend on defense gradients;
- step, restart, EOT-sample, and query-budget convergence curves;
- adaptive optimization with the same anchor/history transition semantics as
  deployment.

Persistent transfer or black-box strength over the purported adaptive
white-box attack, decreasing attack strength as budget grows, or sensitivity
to a single random sample is evidence of an unconverged attack or gradient
masking. In that case the robustness claim fails; it is not attributed to the
defense.

## Episode-wise reporting and failure rules

Clean and attacked cells use the same victim and episode seed. For each episode,
worst utility and safety are selected independently:

```text
utility worst: min episode return
safety worst:  max collision count
               max near-miss count
               max registered safety cost
```

The attack producing worst return need not produce worst safety. Its source
cell is retained for every endpoint. A failed, missing, duplicate, wrong-split,
wrong-budget, wrong-binding, or non-finite row invalidates the complete robust
summary rather than being dropped.

Clean cost includes paired return and safety-cost deltas against the frozen
undefended anchor, detector false-positive rate, purifier distortion,
intervention and fallback rates, and certificate abstention. Driving tasks
also report success/collision/near-miss and only those TTC/DRAC metrics supplied
by the frozen metric contract. Highway observations do not create TTC/DRAC.

Formal comparisons use episode-level paired bootstrap confidence intervals,
median and tail quantiles, and a pre-registered family-wise or false-discovery
correction when multiple methods/endpoints are claimed. Victim and defense
training seeds are hierarchical experimental units and cannot be treated as
independent step samples.

Latency is measured after warm-up at batch size one with device synchronization.
Record hardware/software, end-to-end and component p50/p95/p99, sample counts,
and whether simulator time is included. Compare latency by clean/attack cell;
pooling every cell into one percentile is diagnostic only.

## Non-fungible accounting

The online exporter retains exact per-step and episode counts for:

- victim policy, detector, and detector-internal policy calls;
- proposal-model calls, projection calls, and purification attempts;
- certificate calls, certificate-internal policy calls, and IBP bound passes;
- verified safety-critic, fallback, shield, and overall Guard calls;
- attacker victim/defense forward and backward calls;
- EOT samples, BPDA surrogate calls, and simulator calls.

These currencies remain separate. A diagnostic total may be printed, but no
single “query” number can be used as a shared attack/defense budget. Spatial,
temporal, discrete-edit, attack-query, and defense-compute budgets are also
distinct.

## Artifact hash chain

A formal defense bundle and test audit pin:

- resolved configuration, source commit, dirty-state policy, dependency lock,
  runtime versions, and deterministic settings;
- victim checkpoint bytes and complete in-memory policy-state hash;
- observation/action spaces, action ontology, normalization, environment
  registry, scenario assets, cost definition, and metric-version contracts;
- raw fit/calibration datasets, adjacent manifests, episode/scenario split
  registry, collector semantics, and recomputation evidence;
- detector preprocessing, fusion state, active-channel ablation, calibration
  scores/threshold/alpha, history bootstrap/anchor evolution, and certificate
  epsilon/scope;
- semantic projector fingerprint, temporal envelope, denoiser proposal
  checkpoint/state/data/preprocessing, and purifier contract;
- anchor update/reset contract, fallback preference/critic binding, legal
  action semantics, and optional safety-shield artifact;
- each adaptive attacker checkpoint, manifest, attacker-training split,
  defense binding, and spatial/temporal/query contract;
- immutable episode-row export, audit configuration, output files, and output
  lock manifest.

Checkpoint and sidecar publication is transactional and refuses replacement
unless a training command explicitly requests it. Formal audit output never
overwrites an existing directory. Dependency-injected rows or any
`test_scope=true` record produce integration evidence only.

## Evidence scope

Evidence fields are independent:

```text
algorithm_contract
SB3 categorical integration
public-driving integration
public-driving empirical effectiveness
SUMO contract integration
SUMO empirical effectiveness
```

The implementation and contract fields can be true while every empirical field
is false. At present no stable, immutable SUMO PPO victim and official RAPID
bundle exist. Therefore `sumo_empirical_effectiveness` is forced `false`.
Synthetic SB3 or SUMO-schema tests cannot establish SUMO robustness, safety,
attack superiority, or the later PPO / PPO + Safety Shield / PPO + Safety
Shield + ACCVP comparison.

## Required empirical work

Before any “robust”, “safer”, “strongest”, or innovation-win claim:

- train independent frozen victim and defense cohorts;
- pre-register H1--H3, clean/latency limits, attack budgets, and statistics;
- train every per-defense adaptive attacker without test access;
- run all detector/purifier/adv-training/certificate/fallback/shield ablations;
- complete gradient-masking and attack-convergence checks;
- publish a complete paired matrix with confidence intervals and corrected
  comparisons;
- keep public-driving and SUMO evidence labels separate.

Until those artifacts exist, P5 is an implemented proposed defense and audit
contract, not an empirically validated defense.
