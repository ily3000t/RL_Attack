# P1/P2 paired benchmark contract

Status: implemented. An empirical claim is valid only when the complete bundle
passes `verify` and its manifest says `formal_result_eligible: true`.

`rl-attack-p12-benchmark` evaluates frozen SB3 categorical PPO victims. It
does not train policies and must not expose validation or test episode seeds to
the defense-training path.

## Commands

```powershell
rl-attack-p12-benchmark plan <resolved-config.yaml> --device cpu
rl-attack-p12-benchmark run <resolved-config.yaml> `
  --output-dir <new-output-directory> --device cpu
rl-attack-p12-benchmark run <resolved-config.yaml> `
  --output-dir <interrupted-output-directory> --device cpu --resume
rl-attack-p12-benchmark run <resolved-config.yaml> `
  --output-dir <interrupted-output-directory> --device cpu --resume `
  --max-new-shards <positive-integer>
rl-attack-p12-benchmark verify <complete-output-directory>
```

There are no CLI overrides for victims, epsilon, attacks, solver budgets,
episode seeds, or statistics. These values are frozen in the strict YAML input.
`run` requires an empty new directory; `--resume` requires the exact interrupted
directory and exact original inputs.

`--max-new-shards N` is an execution-only control for long runs. It counts only
newly written, complete shards; existing validated shards do not consume the
quota. If work remains after the quota is reached, the command leaves
`run_state.status: in_progress`, publishes neither summaries nor
`manifest.json`, and returns a progress record that requires a later
`--resume`. The option is not part of the scientific config, plan, fingerprint,
or final manifest, so slicing does not change the result. A shard remains the
smallest pause unit: this is not a wall-clock timeout and never interrupts an
episode cohort midway. `verify` continues to accept only complete bundles.

## Matrix and pairing

- P1 permits only `vanilla_ppo` victims.
- P2 requires exactly `vanilla_ppo`, `adv_ppo`, `sa_ppo`, and `car_ppo`, all
  with the same set of training seeds. Victim names, checkpoint paths,
  checkpoint SHA-256 values, manifest paths, and method/seed identities are
  unique.
- The attack set contains exactly one each of Random-Uniform, FGSM-CE, PGD-CE,
  and categorical MAD-PGD, plus a separately evaluated clean condition. Unique
  display names cannot be used to duplicate an attack kind.
- `base_per_feature`, `mutable_mask`, and every epsilon ratio are expressed in
  policy-input coordinates. Their vector length must equal the flattened policy
  input length.
- The primary victim action is deterministic. Every method shares episode
  seeds, attack-opportunity seeds, and attack-solver seeds. Seed derivation does
  not include method or victim identity, so comparisons use common randomness.

Gymnasium N-D Box observations are wrapped with C-order
`FlattenObservation`. The loaded model and its policy must exactly match the
resulting Box and Discrete spaces.

## Claim tiers and protocol gates

`claim_tier` is one of `smoke`, `development`, or `final`. Smoke requires at
least one training seed, development at least five, and final at least ten.
Smoke is always non-formal.

Every development/final configuration, and every configuration whose cohort
role is `test`, must also satisfy all of the following:

- at least five training seeds, or ten for the final tier;
- at least 200 paired episode seeds;
- at least 10,000 hierarchical-bootstrap replicates;
- `attack_probability: 1.0`;
- at least one mutable feature with positive base epsilon;
- epsilon ratios containing both `0` and `1`;
- PGD-CE and categorical MAD-PGD using at least 20 steps and 5 restarts;
- policy-query and gradient budgets large enough for the frozen solver settings.

Passing those design gates is necessary but not sufficient for a formal result.
Formal eligibility additionally requires a `test` cohort, a clean repository,
matching installed dependency locks, no injected environment/model test hooks,
CPU execution, CPU-trained victims from clean repositories, immutable and
unique loaded policy states, and (for Highway) a formal audited runtime.
CUDA execution or CUDA-trained victims are deliberately marked non-formal;
CUDA determinism is not claimed by this contract.

The formal gate also requires fresh defense training: the training manifest
must record `loaded: false` and `input_checkpoint: null`. The robust
configuration observed on the actual loaded model—at minimum
the full `robust_config`, `model.num_timesteps`, the policy class, and effective
PPO hyperparameters—must equal the frozen configuration and training manifest,
rather than merely matching requested training metadata. Clean evaluation is
formal only when its rows satisfy the strict clean-evaluation schema and include
at least the tier-specific minimum number of episodes.

## Frozen defense-training inputs

Every victim pins both its checkpoint and an
`rl_attack.defense_run.v2` training manifest by SHA-256. The manifest is parsed
with duplicate-key rejection and an exact-key schema. Validation binds:

- method identity, catalog metadata, and the exact robust training mode;
- requested and effective training seeds, devices, robust configuration, and
  PPO hyperparameters;
- raw observation, C-order policy observation, adapter, and Discrete action
  spaces;
- the embedded checkpoint path and SHA-256, plus the manifest's own resolved
  path;
- full Git commit, clean/dirty training state, core and upstream lock hashes,
  and Python/Gymnasium/SB3/Torch runtime versions.

Requested and effective seeds must both equal the victim's configured training
seed. Requested/effective robust configurations must be canonical and equal.
Unknown, missing, stale, or merely self-consistent forged fields are rejected.
Inputs are checked while planning, immediately before every model load, during
resume, and again by `verify`. Policy-state SHA-256 values are recorded before
and after evaluation; mutation or two victims resolving to the same loaded
policy state invalidates the bundle.

## Audited Highway binding

Highway P12 is restricted to the repository's audited `highway-fast-v0`
factory, fixed 30-step episode limit, fixed C-order observation contract, action
ontology, and safety-info wrapper. An injected environment factory is forbidden.
The configuration pins the audited Highway runtime manifest and the repository
Highway dependency lock.

The provenance boundary uses three related SHA layers:

1. training artifacts: each victim checkpoint SHA and strict defense-training
   manifest SHA;
2. audited runtime envelope: runtime-manifest file SHA and its canonical payload
   SHA;
3. replayed runtime contracts: dependency-lock SHA plus effective environment
   configuration, policy-observation contract, action ontology, and safety-info
   contract SHAs.

Freeze/verify re-probes the installed runtime and requires these records to
match. Highway rows must expose valid `crashed`, `collision`, and `on_road`
signals; unavailable safety fields invalidate the audit.

## Shards, resume, and verification

The atomic resume unit is one complete episode cohort for a
victim/condition/attack/epsilon-ratio cell. Every shard binds the run
fingerprint, expected shard identity, exact episode-seed list, exact row count,
and canonical payload SHA-256. Each row is then validated field by field against
the frozen config: identity, derived seeds, epsilon and mask bounds, clean
pairing, return drop, termination, safety values, attack counts, action flips,
solver-query counts, gradients, and perturbation summaries.

Resume accepts only `in_progress`, `finalizing`, or `complete` states with the
same resolved config, plan, runtime/code fingerprint, victim inputs, and strict
shards. Completed shards are reused only after full validation. Finalization is
reconstructible: validated shards are the sole scientific source, so a crash
while writing summaries or after marking the state complete but before
publishing `manifest.json` can be resumed safely. Ordinary exceptions,
including the complete-before-manifest publication window, are recoverable by
resume. A hard process termination can instead leave temporary files, backup
files, or empty directories; those leftovers require manual review and cleanup
before proceeding. This contract does not promise automatic recovery from every
possible crash.

`verify` does not trust summary files or manifest eligibility claims. It
revalidates the frozen source config, runtime, code/lock records, victims, plan,
state, and every shard row; rebuilds every JSON/CSV scientific artifact from the
strict shards; recomputes formal eligibility; and requires the rebuilt bytes and
claims to match the bundle. Updating an internal digest after changing a row or
summary is therefore insufficient.

## Statistical outputs

The bundle contains episode rows, per-checkpoint summaries, hierarchical method
summaries, episode-wise worst-over-attacks rows, and P2 defense comparisons in
both JSON and CSV. `paired_comparisons` reports each defense minus its
training-seed-matched Vanilla PPO for every matrix cell and for
worst-over-attacks. Positive return contrast favors the defense; negative return
drop and collision contrasts favor the defense.

Method and defense-comparison intervals first resample training seeds. Within a
bootstrap replicate they then use one shared episode-index draw across every
sampled training seed, because episode seeds are a crossed blocking factor.
They must not independently resample episode indices per model. Checkpoint
intervals use an episode bootstrap.

`policy_queries` and `policy_queries_per_attacked_step` count only policy-logit
queries made internally by the attack solver. They exclude the clean/attacked
victim action selections and environment steps. The frozen per-applied-attack
accounting is Random-Uniform `0`, FGSM-CE `3`, and each iterative attack
`1 + restarts * (steps + 1)`; gradient evaluations are respectively `0`, `1`,
and `restarts * steps`.

## Filesystem and integrity boundary

Bundle paths are canonical relative POSIX paths. Traversal, absolute/drive
paths, backslashes, NULs, unsafe or overlong components, Windows reserved names,
and trailing dots/spaces are rejected. The output may not alias or contain a
pinned input. Symlinks, junctions, reparse points, unexpected top-level files,
and unexpected shard files are rejected during run, resume, and verify.

Internal SHA-256 records detect corruption and inconsistent rewrites; they do
not authenticate an adversary because no secret key or signature is involved.
For external tamper evidence, publish the final `manifest.json` SHA-256 through
an independent trusted channel. The manifest intentionally has no recursive
self-hash.
