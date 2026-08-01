# Architecture and isolation boundaries

`RL_Attack` is the independent third research block. It owns attack/defense
algorithms and experiment protocols; it does not own WCDT or ACCVP.

```text
WCDT_ACCVP_Attack (read-only producer)
        |
        | immutable scenario/victim snapshot + hashes
        v
RL_Attack
  src/rl_attack/             maintained algorithms and adapters
  scenarios/.../vN/          immutable SUMO snapshots
  artifacts/victims/.../vN/  immutable victim bundles (future)
  configs/                   versioned experiment contracts
  third_party/upstream/      ignored, detached paper-code checkouts
  third_party/environments/  dependency records, never core dependencies
  outputs/                   ignored generated runs
```

## Dependency direction

Allowed:

```text
experiment -> attack/defense -> core policy protocol
experiment -> environment adapter
SB3 adapter -> stable_baselines3
SUMO environment -> gymnasium + numpy + traci
```

Forbidden:

```text
src/rl_attack -> WCDT_ACCVP_Attack
src/rl_attack -> safe_rl
src/rl_attack -> third_party/upstream
SUMO environment -> SB3 or torch
```

The `scripts/check_isolation.py` test enforces the first three rules. The SUMO
package has an additional import test.

## P4 STFA boundaries

The maintained P4 path keeps simulator state, policy input, and learned
artifacts distinct:

```text
environment info -> safety-signal adapter -> audit metrics
policy observation -> semantic projector -> frozen PPO policy
fixed NPZ + sidecar -> critic training -> critic checkpoint + sidecar
fixed NPZ + critic -> director training -> director checkpoint + sidecar
```

The projector may enforce policy-input schema and budget constraints, but it
does not write simulator state or certify physical realizability. SUMO discrete
search produces allowlisted policy-input candidates only. The audit owns the
hard temporal ledger and validates the environment registry, scenario assets,
spaces, normalization, cost definition, action ontology, victim, critic, and
director before a production run.

Training commands consume pre-built immutable datasets; they do not collect
rollouts. Generated datasets, checkpoints, sidecars, and audit outputs stay
under ignored output/artifact locations and are referenced by exact hashes in
run-specific resolved configs. The checked-in P4 experiment files are templates,
not mutable pointers into WCDT.

## P5 RAPID-Guard boundaries

P5 wraps, but never mutates, a frozen categorical victim:

```text
raw fixed cohorts + sidecars
        |
        +--> frozen victim recomputation --> detector fit
        |                                  + clean conformal calibration
        |
        +--> clean targets + trusted inputs --> residual proposal fit
                                                   |
                                                   v
environment policy input --> detector --> semantic-temporal purifier
                                  |                 |
                                  +-----------------+
                                           |
                                           v
                             victim / legal fallback / optional shield
                                           |
                                           v
                              immutable episode-row audit export
```

The raw datasets do not cache policy probabilities, logits, IBP bounds, or
detector features. These are recomputed from the exact frozen victim. Fit and
calibration datasets, episode/scenario splits, normalizer, action ontology,
environment, projector, IBP radius, anchor rule, purifier, and fallback are
cross-bound into the bundle.

The Guard owns one transactional anchor/history per episode. Detection occurs
before any state update. Calibrated detection requires a caller-attested,
attack-free prefix containing at least two consecutive trusted frames from the
current episode; a repeated-first window is never synthesized. With no or only
one trusted frame, preflight fails closed before victim-policy or IBP queries
and reports the fallback cost outside H1 coverage. Only validated pass-through
and accepted purification can commit trusted state. Fallback preserves the last
trusted anchor but invalidates temporal continuity until an explicit, bound
trusted-prefix rebootstrap. Paired clean, non-adaptive, and defense-aware
rollouts must use independent Guard instances.

The denoiser produces an unprojected proposal only. The purifier's strongest
guarantee is policy-input schema and temporal-envelope consistency. The
fallback guarantees legal action selection only. IBP certifies a supported
actor's one-step clean greedy action only. None proves physical realizability,
episode return, collision avoidance, or closed-loop safety.

The P5 offline audit is downstream of a real evaluation harness. It aggregates
hash-pinned rows and cannot manufacture online SUMO evidence. Dependency
injection is a test seam and permanently disables robustness eligibility. See
[`contracts/p5_rapid_guard.md`](contracts/p5_rapid_guard.md) for the complete
claim and artifact boundary.

## Victim import contract

A future WCDT checkpoint is accepted only as a versioned victim bundle:

```text
artifacts/victims/wcdt/<version>/
  model.zip
  vecnormalize.pkl        # when used
  policy_metadata.json
  observation_schema.json
  source_manifest.json    # WCDT SHA and hashes
```

Never point experiments at a mutable file inside WCDT. A new checkpoint creates
a new version; it never changes attack or defense implementations.

## Third-party policy

Paper repositories are operational references, not Python dependencies of this
package. `third_party/upstream-lock.json` is authoritative. The sync script
checks out detached commits and verifies submodules. Repositories whose root
license is unknown remain read-only.
