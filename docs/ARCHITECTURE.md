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
