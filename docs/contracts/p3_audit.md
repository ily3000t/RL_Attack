# P3 reproduced strong-attack audit contract

Status: implemented (已落实) as the executable audit boundary for maintained
P1 PGD-CE/MAD-PGD baselines and reproduced Robust-Sarsa/PA-AD attacks. This
contract evaluates attacks; it does not claim that an upstream paper result
has been reproduced merely because an adapter runs.

## Matrix and pairing

`rl-attack-strong-audit CONFIG --output-dir RUN_DIR` executes the full Cartesian
product

```text
victim checkpoint × epsilon ratio × paired episode seed × attack
```

The checked-in CartPole formal matrix contains exactly `pgd_ce`,
`categorical_mad_pgd`, `robust_sarsa`, and `pa_ad`. They all reference the same
epsilon profile, mutability mask, episode and solver seeds, hard budget, victim
action mode, and episode-wise `worst_over_attacks` reporting rule. The labels
are intentionally precise:

- PGD-CE and categorical MAD-PGD are `maintained_p1_attack_baseline`;
- Robust-Sarsa is a
  `clean_room_categorical_robust_sarsa_adaptation`, not a paper-exact copy;
- PA-AD is a
  `clean_room_stochastic_pa_ad_with_pgd_actor_extension`, because the maintained
  inner actor supports bounded multi-step/restart PGD.

Each victim/episode seed is first evaluated cleanly. Every attacked row stores
that exact clean return and the paired drop. Attack opportunity streams are
shared across attacks for the same victim, epsilon ratio, and episode seed;
solver randomness additionally includes the attack name. Thus algorithms see
the same temporal opportunities without accidentally sharing their optimizer
randomness.

`fairness.reporting_protocol.victim_action_mode` is mandatory. In
`stochastic` mode the runner samples actions itself with inverse-CDF sampling;
it never relies on SB3's process-global RNG. The clean and attacked episode use
the same SHA-256-derived uniform variate at each timestep (common random
numbers), while each observation retains its own categorical probabilities.
In `deterministic` mode both paths use categorical argmax. The action mode is
passed to every attack factory and must match the learned attacker checkpoint.
The formal categorical PA-AD reproduction currently accepts only `stochastic`;
requesting an unsupported deterministic PA-AD variant fails closed rather than
being mislabeled as the paper method.

The audited observation is the flattened, C-row-major policy input. A config
must provide one base epsilon and one Boolean mutability entry per policy input
feature. Effective bounds are

```text
epsilon_i(ratio) = base_epsilon_i × ratio × mutable_mask_i
```

Immediately after loading every PPO victim, the runner compares both the model
and policy observation/action spaces against the policy-facing agent
environment. Box shape, dtype, lower/upper bounds and Discrete action
count/start/dtype must match exactly. If the environment needed
`FlattenObservation`, the manifest records the source shape and C-row-major
mapping; a victim trained for an unflattened or differently flattened space is
rejected. The policy-state hash, eval mode, and `requires_grad` freeze invariant
are recorded before evaluation and checked again after the full matrix. A
factory or stateful attack cannot silently mutate the victim and still produce
a completed manifest.

The runner rejects shape changes, non-finite output, invalid observation bounds,
epsilon violations, and any change to an immutable feature.

## Exact seed derivation

The identifier is `sha256_u63_canonical_json_v1`. Its payload is UTF-8 canonical
JSON with sorted keys, compact separators, ASCII escaping, and no NaN:

```json
{
  "algorithm": "sha256_u63_canonical_json_v1",
  "base_seed": 31000000,
  "components": ["..."],
  "namespace": "attack_solver"
}
```

SHA-256 is computed over those bytes. The first eight digest bytes are decoded
as an unsigned big-endian integer and the sign bit is cleared. Epsilon ratios
are components encoded with 17 significant digits. Opportunity components are
`checkpoint_sha256, episode_seed, epsilon_ratio`; solver components append
`attack_name`.

## Hard accounting boundary

During `attack.generate`, the attack receives only an
`InstrumentedCategoricalPolicy`. Every `logits` forward is charged as one policy
query. Every autograd traversal through a returned logits tensor is charged as
one gradient evaluation. The proxy raises before a query exceeds its per-step
limit and raises inside autograd when a gradient would exceed its limit.

The `AttackResult` declaration must exactly equal the instrumented counts.
Self-reported cost cannot silently replace measured cost. Clean-action and
post-attack audit predictions are reporting operations and are deliberately
outside the solver budget.

For learned PA-AD, these counters measure victim-policy forwards and
victim-input autograd traversals only. Director inference is not a victim query
or victim gradient, and this runner does not currently report a separate
director-latency or end-to-end wall-clock metric. Formal results must not claim
otherwise.

Any non-empty attack metadata field whose name contains `fallback` immediately
invalidates the entire run. The runner writes an `invalid` manifest with
`robust_return_eligible=false`, emits no episode/summary/worst-return artifacts,
and raises `InvalidAttackEvaluation`. A zero perturbation may be a valid result
only when the attack labels it as such without fallback metadata (for example,
at epsilon zero); numerical, disconnected-gradient, degenerate-direction, or
no-improvement fallbacks can never inflate reported robust return.

An injected factory has this interface:

```python
def factory(context: AttackBuildContext) -> ObservationAttack: ...
```

The context supplies immutable bounds, shared budget metadata, victim
checkpoint and full policy-state hashes, victim action mode, spaces, epsilon
ratio, and method-specific `factory_kwargs`. Built-in factories accept only a
closed algorithm-parameter schema, reject ambiguous step-size shapes, and
prove the planned query/gradient counts fit the shared budget before execution.
They also require exact observation/action dimensions and exact epsilon/mask
bindings. A stateful learned
attacker may optionally implement `reset_episode`, `observe_transition`, and
`end_episode`; the runner invokes them without giving mutation access to the
environment.

## Metrics

Episode artifacts retain return, length, attack count, actual perturbation
norms, measured query/gradient counts, and action flips under the declared
execution mode. Stochastic evaluation compares clean and adversarial sampled
actions using the same inverse-CDF common random number; deterministic
evaluation compares argmax actions. Method-configured success, clean pairing,
and safety information are reported separately.
Safety event rates and minimum values are reported only if the environment
actually emitted their configured `info` keys. Missing values remain JSON
`null` with a reason and are never filled with zero.

For every victim/ratio/attack condition, the runner reports:

- mean attacked return and paired return-drop bootstrap confidence intervals;
- lower-tail return CVaR at the configured alpha;
- action-flip and attack-specific-success rates with episode bootstrap;
- policy queries and gradients per attacked step;
- maximum realized L-infinity perturbation and available safety metrics.

`worst_over_attacks` is computed per paired episode seed by taking the minimum
return across the complete attack set, then aggregating those episode-wise
minima. It is not the minimum of independently averaged tables.

## Artifacts and provenance

One completed run contains:

```text
resolved_config.json
episodes.json / episodes.csv
summaries.json / summaries.csv
worst_over_attacks.json / worst_over_attacks.csv
manifest.json
```

All JSON is strict (`NaN`/infinity forbidden). The manifest records the source
config SHA-256, every victim checkpoint and policy-state SHA-256, full matrix cardinality,
epsilon/mask/budget/seed/reporting contracts, attack fidelity labels and factory
arguments, runtime versions, Git commit/dirty state, lock hashes, and hashes of
all non-self-referential artifacts.

Each attack references the single shared budget, epsilon profile, seed protocol,
and reporting protocol. The loader rejects mismatched references or per-method
schema additions, preventing a nominal comparison from changing the mask,
seeds, solver budget, or headline metric.

Robust-Sarsa critics and PA-AD directors are accepted only when their pinned
checkpoint hash, separately pinned adjacent-manifest hash, frozen-victim
evidence, checkpoint/policy-state provenance, observation shape, action count,
and victim action mode all match the live audit victim. Missing provenance is
an error; there is no compatibility fallback. Robust-Sarsa must record a
positive-size transition set, positive gradient training, finite losses, and a
positive robust-training coefficient. Its adjacent manifest must equal the
manifest embedded in the checkpoint. PA-AD's sidecar must record a positive
maintained PAMDP training run, negative-victim-reward contract,
collected-step/query/gradient evidence, unchanged victim hashes, and
payload/sidecar agreement for architecture, initialization seed, victim
provenance, and training manifest. The embedded training manifest pins the exact
policy-input observation shape, epsilon vector, lower/upper bounds, and mutable
mask. Every nonzero audit epsilon must equal that contract element-wise. A
director whose state still equals its seeded random initialization is rejected
even if a sidecar claims training.

The epsilon-zero PA-AD condition is an identity control: it neither loads nor
calls the director. For nonzero conditions, one director checkpoint is valid
for exactly its training epsilon. The checked-in template therefore contains
only ratios `[0.0, 1.0]`. To evaluate another nonzero epsilon, copy the template,
train a separate director with that exact per-feature vector and bounds/mask,
pin the new checkpoint and sidecar hashes, and run a separate resolved audit.
Reusing one director across epsilon values is a transfer experiment and is not
eligible for a strongest-attack claim under this contract.

## Formal CartPole artifact paths

The formal config consumes the actual P2 and learned-attacker CLI defaults:

```text
outputs/p2_defenses/CartPole-v1_vanilla_ppo_seed0/model.zip
outputs/p3_reproduced_attack_training/CartPole-v1_robust-sarsa_seed0/robust_sarsa.pt
outputs/p3_reproduced_attack_training/CartPole-v1_pa-ad_seed0/pa_ad.pt
```

Before a formal run, copy the checked-in YAML to a run-specific resolved config
and replace all four digest placeholders with lowercase values from:

```powershell
Get-FileHash -Algorithm SHA256 <robust_sarsa.pt>
Get-FileHash -Algorithm SHA256 <robust_sarsa.pt.manifest.json>
Get-FileHash -Algorithm SHA256 <pa_ad.pt>
Get-FileHash -Algorithm SHA256 <pa_ad.pt.manifest.json>
```

Then execute `rl-attack-strong-audit <resolved-config.yaml> --output-dir
<run-directory>`. The runner re-hashes all four resources before loading them
and writes the fully resolved contract and artifact hashes into the audit
manifest.
