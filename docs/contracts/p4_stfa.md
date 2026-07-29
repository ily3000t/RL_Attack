# P4 STFA contract

Status: implemented（已落实）as an algorithm, artifact, training-plumbing, and
strict audit contract. This status does not mean that STFA has been shown to
outperform P3 attacks, and it does not establish SUMO empirical effectiveness.

## Scope and threat model

`STFA` is a sequential, white-box observation attack for a frozen categorical
policy. At every environment step, a director may select an attack and declare
one legal target action together with its lateral and longitudinal factors. The
inner optimizer then searches the declared policy-input feasible set. The
target is an optimization goal; the audit separately records the target action,
the action actually produced by the victim, target hit, and action flip.

The maintained implementation includes six objective variants:

- `full`: expected safety cost plus joint and factorized target margins;
- `flat`: expected safety cost plus the joint target margin;
- `factor`: expected safety cost plus lateral/longitudinal margins;
- `safety`: expected safety cost only;
- `ce` and `mad`: maintained categorical comparison objectives.

Continuous perturbations are projected after every optimizer update. Optional
discrete search uses a deterministic planner and is accounted separately.
Continuous budget, discrete edit cost, temporal selection, policy forwards,
input gradients, projection calls, critic calls, director calls, and defense
transform calls are not interchangeable currencies.

## Action and temporal contracts

SUMO uses the repository-owned, zero-based 3 × 3 ontology: lateral
`{-1, 0, +1}` crossed with longitudinal `{-1, 0, +1}`. HighwayEnv uses its
canonical five meta-actions and a sparse factorization; a runtime descriptor
whose action-name/index mapping differs from the canonical mapping is rejected.
Availability masks are part of each decision, and unavailable actions cannot
be selected as targets.

Every attacked episode owns one `TemporalBudgetLedger`. Its immutable contract
contains:

```text
K, minimum gap, optional rolling-window size, optional rolling-window K
```

The ledger, not an attack probability, authorizes selection. A selected step
consumes one temporal token even when projection produces a zero perturbation
or the victim action does not change. The director checkpoint is bound to the
same full temporal contract and episode horizon used for its labels.

## Semantic projection boundary

The generic projector enforces the declared policy-input shape, finite values,
valid lower/upper bounds, per-feature L-infinity budget, and immutable mask.
`ProjectionResult` binds the clean observation to the projected observation,
recomputable perturbation, norms, discrete edits, and edit cost.

The SUMO `sumo_merge_core_v1` projector additionally enforces the frozen
52-feature layout, physical-unit-to-policy-input budget conversion, categorical
grids, binary flags, positive vehicle dimensions, ordered neighbor slots, and
unchanged zero-padding slots. Its discrete planner requires an explicit feature
allowlist and enumerates deterministic, single-field legal-grid neighbors. It
does not enumerate all multi-edit combinations and it never mutates simulator
state.

The Highway projector is constructed from the actual runtime descriptor,
preserves C-row-major flattening, freezes presence columns and padding rows, and
does not invent TTC or DRAC from an underspecified observation.

These projectors guarantee policy-input schema consistency only. They do not
prove that a projected vector corresponds to a dynamically or geometrically
realizable simulator state.

## Learned artifact and dataset binding

The safety critic estimates one cost per action at the clean observation. Its
checkpoint and adjacent strict-JSON sidecar bind:

- the exact frozen PPO checkpoint and complete policy-state hashes;
- observation/action spaces and action ontology;
- dataset file and dataset-sidecar hashes;
- environment, normalization, and cost-definition contracts;
- deterministic victim-probability recomputation and training metadata.

The temporal/factor director consumes the clean observation, victim action
probabilities, safety costs, budget state, and time features. Its checkpoint
also binds the critic checkpoint/state/space, full temporal budget, horizon,
labeler, dataset provenance, and action factorization. Runtime loading fails
closed on any binding mismatch. Saving checkpoint plus sidecar is transactional
and refuses overwrite unless it is explicitly requested.

`rl-attack-train-stfa critic` and `rl-attack-train-stfa director` consume pinned
NPZ files with adjacent manifests. They do not collect rollouts. The loader
uses `allow_pickle=False`, requires exact field names, shapes, and dtypes, and
recomputes victim probabilities (and director safety costs) before training.
A training run manifest is evidence that this fixed-data pipeline executed; it
is not a robustness statistic or a paper result.

## Defense-aware modes

The implementation distinguishes transfer, victim-adaptive, exact
differentiable, EOT, and BPDA declarations. EOT requires multiple genuine
stochastic transform samples. BPDA requires an explicit surrogate. The audit
records transform and surrogate use; merely choosing a mode does not prove
that an adaptive attack is converged.

## Strict audit gate

`rl-attack-p4-audit CONFIG --output-dir RUN_DIR` accepts a closed YAML schema,
pins the victim checkpoint and in-memory policy state, uses deterministic
categorical argmax, pairs clean/attacked episode seeds, and gives every attacked
episode a hard temporal ledger. It verifies the environment registry identity,
spaces, normalization, scenario assets, factorization, semantic projector,
safety-cost definition, critic, director, and discrete planner before
evaluation.

Injected factories exist only for contract tests. Any injected run is marked
`test_scope=true`, is ineligible for a robust summary, and cannot be presented
as production evidence. Invalid attack output publishes only a strict invalid
manifest. Complete production outputs retain target-versus-actual action,
perturbation/edit accounting, ledger history, paired episodes, provenance, and
artifact hashes.

P4 evidence must be labelled independently:

```text
algorithm_contract
SB3 nine-action integration
SUMO contract integration
SUMO empirical effectiveness
```

The final field is forced false until a stable, immutable SUMO PPO victim and
official learned artifacts exist. A synthetic nine-action smoke can establish
the first two fields; it cannot establish the latter two.

The checked-in
`configs/experiments/p4_synthetic_9action_smoke.yaml` and
`configs/experiments/p4_sumo_stfa9_implementation_gate.yaml` files are strict,
parseable unresolved templates. All-zero SHA-256 strings are deliberate
sentinels for missing external artifacts, not fabricated digests. The SUMO
template pins the current scenario files, identity normalization, semantic
projector, safety-cost definition, infinite Box-bound encoding, and disabled
evidence claims, but the audit deliberately refuses a production SUMO
construction until that registry path and a stable victim are ready.

## Required empirical work

The implementation release still requires frozen victim cohorts and
pre-registered experiments for:

- P3 versus STFA under matched spatial, temporal, and measured query budgets;
- semantic/plain, random/learned timing, flat/factorized, and objective
  ablations;
- convergence over steps, restarts, EOT samples, and discrete candidates;
- non-adaptive and adaptive attacks against each defense;
- independently trained victims and attackers, paired test seeds, confidence
  intervals, and episode-wise worst-over-attacks.

Until those runs exist, no claim of attack superiority, defense robustness,
SUMO effectiveness, or innovation win is eligible.
