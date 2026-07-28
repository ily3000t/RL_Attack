# PA-AD clean-room reproduction contract

## Reference

Yanchao Sun, Ruijie Zheng, Yongyuan Liang, and Furong Huang, “Who Is the
Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL,”
ICLR 2022.

- Paper: <https://arxiv.org/abs/2106.05087>
- Authors' repository: <https://github.com/umd-huang-lab/paad_adv_rl>
- Pinned revision: `ef04e7912abc0937531ba95920a1b78688cd023e`
  in `third_party/upstream-lock.json`

Any upstream checkout is reference-only. Its licensing status is unresolved,
it is never imported at runtime, and no source was copied into the maintained
package.

## Implemented scope（已落实）

The maintained discrete-action path reproduces the paper's two-level
decomposition:

1. a trainable stochastic-PAMDP director observes the clean environment state and proposes
   a zero-sum, unit-length direction in the victim's categorical policy space;
2. the actor uses victim input gradients to find an observation inside the
   explicit per-feature L-infinity box that moves the policy in that direction;
3. the attacked categorical distribution is sampled (never replaced by
   argmax), the environment reward is negated, and the resulting transition is
   used to train the director;
4. victim parameters and persistent buffers are frozen and hashed before and
   after collection; they are never owned by the director optimizer or stored
   inside director checkpoints.

For a stochastic categorical victim, the clean-room actor maximizes

```text
||pi(s_adv) - pi(s)||_2
  + lambda * cosine(pi(s_adv) - pi(s), director_direction)
```

The default one-step solver is the paper's reported FGSM approximation.
Multi-step/restart PGD is supported only as an explicitly labelled extension.

## Runtime and perturbation contracts

- Victim interface: differentiable `CategoricalPolicy`; SB3 PPO with a
  `Discrete` action space is supported through `SB3CategoricalPolicyAdapter`.
- Observation interface: `observation_shape` is mandatory. A tensor with that
  exact shape is one sample and only one additional leading axis denotes a
  batch. Thus a `(vehicles, features)` observation is never mistaken for a
  batch. Flattening, when desired, belongs to an explicitly recorded
  environment adapter; the attack performs no implicit flattening.
- Perturbation interface: finite epsilon, lower validity bound, upper validity
  bound, boolean mutable mask, and optional step size must each have the exact
  `observation_shape` (no scalar or accidental batch broadcasting). The clean
  observation must already lie inside the declared validity box.
- Budget: victim forward queries and victim input-gradient evaluations are
  checked before any victim query. Director inference is not a victim-policy
  query; its latency is currently excluded and is not separately reported, so
  P3 makes no wall-clock-efficiency claim for PA-AD.
- Determinism: evaluation fixes the director mode, checkpoint hash, and attack
  seed. Random starts and stochastic director samples use the supplied
  `torch.Generator`.
- Identity and fallback: epsilon zero is a valid identity evaluation point,
  not a fallback. Degenerate directions and only the narrow cases of a
  disconnected or non-finite input gradient return the clean observation and
  record an invalid fallback. Victim shape/logit and sensor-contract errors
  propagate and invalidate the run rather than being swallowed.

## Victim action-mode contract

The maintained P3 route supports only `victim_action_mode: stochastic`. Both
director training and evaluation must sample from the attacked categorical
distribution, matching the stochastic objective `J`. A deterministic victim
requires the paper's separate D-PAMDP: its director emits a target action and
its actor maximizes the targeted margin `J_D`. That branch is intentionally not
implemented; requesting it raises a fail-closed error rather than silently
mixing stochastic `J` with argmax execution.

## Director training and artifacts

`collect_pa_ad_rollout` and `train_pa_ad_from_sb3` provide the executable
PAMDP loop: sample a director direction, solve the actor, sample the attacked
victim action, step the environment, negate the victim reward, compute GAE,
and update the director with maintained PPO. GAE distinguishes `terminated`
(which disables value bootstrap) from `episode_end` (which also includes
truncation and resets the recursive trace). Every rollout field is detached
inside the trainer before PPO graph construction.

`PAADDirectorTrainer` is a maintained PPO implementation over the lower
dimensional policy-direction action space. An orthonormal Helmert basis maps its
`|A|-1` dimensional Gaussian latent action into the zero-sum tangent space of
the categorical probability simplex. Its rollout batch stores that latent
action so PPO log probabilities remain well-defined. The mapped direction is
normalized only for the actor. `generalized_advantage_estimate` consumes
adversary rewards, which must equal negative victim rewards.

Director checkpoints use a versioned weights-only format and embed the frozen
victim's checkpoint SHA-256 plus a shared P3 SHA-256 of all policy parameters
and persistent buffers. The loader can require the director artifact hash and
both victim hashes, rejecting any cross-victim reuse. Saving also emits a
strict JSON manifest containing those bindings, architecture, seeds, training
contract, fidelity label, and explicit confirmation that no victim checkpoint
or victim optimizer update is included.

## Fidelity label and limitations

All maintained results use:

```text
method: pa_ad
reproduction_level: clean_room_algorithmic
paper_exact_reproduction: false
upstream_runtime_dependency: false
```

This is not an “official PA-AD” or paper-exact run. The original experiments
used legacy environments, networks, hyperparameters, and an upstream training
stack; this project instead targets maintained SB3 PPO victims and explicit
sensor-level contracts. The paper's optimality theorem concerns an optimal
director and exact actor formulation. It does not make a finite-training,
finite-gradient implementation provably optimal.
