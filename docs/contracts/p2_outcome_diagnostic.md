# P2 CartPole outcome diagnostic contract

Status: post-hoc engineering diagnostic. This contract does not authorize a
formal attack or defense effectiveness claim.

## Purpose

The ratio-6 screening changed many policy actions without reducing the clean
Vanilla PPO return. This gate separates three explanations before more effort
is spent on P4 or P5:

1. CartPole's capped survival reward can hide state degradation;
2. the observation attack may flip causally unimportant actions or times; and
3. an iterative solver may spend queries after it has already crossed the
   policy decision boundary.

It also closes the provenance chain between each evaluated defense, its
training threat, and the already completed ratio-6 P12 screening. It is not a
new benchmark and must not be used to rank defenses.

## Frozen inputs

The tracked configuration is
`configs/experiments/p2_cartpole_outcome_diagnostic_eps600.yaml`. It binds by
SHA-256 to:

- the tracked P12 ratio-6 configuration;
- the complete P12 ratio-6 output manifest; and
- the Vanilla PPO, Adv-PPO, SA-PPO, and CAR-PPO defense configurations.

All ten episode seeds are a declared post-hoc slice of the prior validation
cohort: `25000..25009`. The epsilon ratio remains `6.0`; it is not tuned using
these outcomes. A missing, changed, incomplete, or unverifiable source bundle
must fail closed.

## Forced-action interventions

Each victim is evaluated deterministically on the same seeds under exactly
five paired conditions:

- `clean`: execute the victim action;
- `opposite_all`: replace every binary CartPole action with `1 - action`;
- `opposite_first_1`: replace only the first eligible action;
- `opposite_first_5`: replace the first five eligible actions; and
- `opposite_first_20`: replace the first twenty eligible actions.

The intervention count is based on actual environment steps, not attempted
model calls. A non-binary action space is outside this contract and must be
rejected. These interventions answer whether action changes can cause task
damage; they are controls, not observation attacks.

## Outcome and safety-margin measurements

In addition to total reward and episode length, every condition records:

- terminated and truncated separately;
- maximum absolute cart position;
- maximum absolute pole angle;
- minimum cart margin `2.4 - abs(cart_position)`;
- minimum pole margin `0.20943951023931953 - abs(pole_angle)`; and
- the minimum normalized margin across those two constraints.

Margins are evaluated on the authoritative observation returned by the
environment. Negative values mean that a limit was crossed. The run records
per-episode rows and paired differences; it does not substitute action-flip
rate for outcome harm.

## PGD iteration trace

The diagnostic uses an independent trace implementation that mirrors the
production P12 `pgd_ce` update, random stream, projection, and final-only
restart selection under the same ratio-6 epsilon profile. For each clean
trajectory it selects at most eight states at deterministic, uniformly spaced
indices spanning the trajectory, rather than simply taking the first eight
steps. The initial candidate and every one of 20 iterations for each of 5
restarts are retained. Each trace point records the objective, logit margin,
selected action, action-flip state, perturbation norm, restart, iteration, and
the production candidate-selection state.

Production equivalence is a hard requirement: the terminal/best traced
candidate, accounting totals, random seeds, projection, and selected action
must equal a normal production-solver call on the same frozen observation.
Tracing must not introduce additional solver randomness. `run` checks
bit-parity between the traced final-only winner and a production solver call;
`verify` reruns the complete trace and production solver from the frozen model,
state, bounds, and seed. Failure of either equivalence invalidates the run
rather than producing a partial conclusion.

## Defense-threat closure

For every victim, the diagnostic records and cross-checks:

- defense method and training mode;
- the pinned defense YAML and its SHA-256;
- configured training epsilon, PGD steps, step size, restarts, and schedule;
- corresponding effective values reported by the victim training manifest;
- the evaluation epsilon vector implied by ratio `6.0`; and
- whether train and evaluation threat models are matched.

A mismatch is reported, not silently normalized away. In particular, results
outside a defense's training radius are stress observations and cannot be
phrased as failure of matched-threat robustness. A mismatch between a defense
YAML and its training manifest is instead an integrity failure and must fail
closed.

Defense closure is embedded in `plan.json`; it is not a separate result
artifact. Vanilla PPO is the undefended reference, so threat matching for it is
`not_applicable_reference`, not a defense success or failure. Adv-PPO, SA-PPO,
and CAR-PPO must all match the evaluation threat before a comparative defense
interpretation is allowed. At ratio 6 their effective evaluation bounds
(`0.30` and `0.06` by feature) exceed the scalar training epsilon (`0.02`), so
that interpretation gate is false by construction.

## Explicit diagnostic gates

`summary.json` reports exactly four gates. Each contains `passed`, frozen
`thresholds`, measured `evidence`, and machine-readable `reasons`:

1. `environment_outcome_sensitive` uses Vanilla PPO paired
   `opposite_all - clean` evidence. It passes if the mean paired return drop is
   at least `1.0`, or if the mean normalized joint-margin decrease is at least
   `0.05`.
2. `observation_attack_outcome_aligned` passes only when the environment gate
   passes and at least one verified P12 attack group for that same Vanilla PPO
   victim among FGSM, PGD, and MAD has mean paired return drop of at least
   `1.0`. Other victims remain reported as context but cannot be borrowed to
   pass the Vanilla alignment decision. The group means are recomputed from the
   pinned P12 `episodes.json`; its path and SHA-256 are recorded as evidence.
   P12 summary tables are not trusted as the source of this result.
3. `pgd_incremental_value` compares the final-only PGD winner with the best
   iteration-1 candidate across restarts for Vanilla PPO only. It passes if
   that Vanilla aggregate mean objective increase is at least `0.001`, or its
   final flip-rate increase is at least `0.05`. Other victims are reported only
   as contextual counts and cannot be borrowed to pass this gate. The
   best-seen-minus-final objective difference is also reported to diagnose
   final-only candidate-selection loss.
4. `defense_comparison_interpretable` passes only if Adv-PPO, SA-PPO, and
   CAR-PPO all have a matched evaluation threat. Vanilla remains an N/A
   reference. This gate is false for the ratio-6 stress run and therefore
   forbids a defense-ranking conclusion.

These thresholds are routing criteria, not formal hypothesis tests. A failed
environment gate calls for a different task or outcome metric; a failed
observation-alignment gate calls for attack objective/timing work; a failed PGD
incremental-value gate indicates that iterative queries are not earning their
cost; and a failed defense-interpretability gate prevents comparative defense
claims.

## Artifacts and verification

`run` writes to a new, empty output directory only. The bundle includes the
resolved config, plan, per-episode intervention rows, PGD trace rows,
state-bank rows, summaries, and a manifest containing SHA-256 pins for every
artifact. The plan contains the defense-closure evidence. Existing output is
never overwritten. The destination must be outside the source P12 bundle and
all pinned-input parent trees; direct or ancestor symlink/reparse paths are
rejected. A sibling lock directory reserves the destination during publication.

`verify` re-hashes every pinned input and artifact, reproduces the plan,
reloads model identities, recomputes state margins, PGD candidate selection,
query accounting, source P12 observation-attack group means, all four gates,
summaries, and bootstrap intervals, and rejects semantic or byte-level
tampering. It does not claim an independent simulator replay; raw episode rows
remain bound by their artifact digest, aggregate closure equations, exact
state-index contract, and closed-world semantic checks. The manifest states
the integrity boundary explicitly: internal JSON hashes detect corruption;
scientific-source SHA-256 values and pinned-input revalidation bind the local
execution, while an externally retained manifest or output digest supplies the
external trust anchor. The P2 manifest does not itself record a Git commit.

## Statistical and claim boundary

Paired summaries use 1,000 deterministic bootstrap replicates at 95%
confidence. With only ten post-hoc seeds, all intervals are diagnostic.
The configuration, plan, summary, and manifest must retain these flags:

```yaml
post_hoc: true
formal_eligible: false
diagnostic_only: true
```

Allowed conclusions concern mechanism diagnosis, such as whether forced action
changes reduce reward or safety margins and how quickly PGD crosses a decision
boundary. The bundle does not authorize defense ranking, robustness claims,
publication-level significance, P4 final-seed use, or P5 effectiveness claims.
