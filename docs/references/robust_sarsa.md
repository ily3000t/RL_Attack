# Robust-Sarsa reference and fidelity

## Primary method

Huan Zhang et al., *Robust Deep Reinforcement Learning against Adversarial
Perturbations on State Observations*, NeurIPS 2020:

- paper: <https://papers.nips.cc/paper/2020/hash/f0eb6568ea114ba6e293f903c34d7488-Abstract.html>
- supplemental material: <https://papers.nips.cc/paper_files/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Supplemental.pdf>

The maintained implementation separates robust-critic training from the
observation attack. For a categorical victim, let
`x = concat(s, one_hot(a))`. The critic objective is:

```text
L_RS = mean((r + gamma Q(s_next, a_next) - Q(s, a))^2)
       + lambda_RS * mean(max_[s',u'] (Q(s', u') - Q(s, one_hot(a)))^2)

s' in clip([s - eps_state, s + eps_state], valid_state_domain)
u' in clip([one_hot(a) - eps_action, one_hot(a) + eps_action], [0, 1]^A)

s_adv = argmin_{s' in B(s_clean)} Q_RS(s_clean, pi(s'))
```

The first two lines are the **training regularizer**: both state and action
coordinates of the critic input are in the inner neighborhood. This matches
the pinned upstream implementation's construction of one bounded tensor from
`torch.cat((sel_states, sel_actions), dim=1)` at
`SA_PPO@7f5193e:src/policy_gradients/agent.py:924-952`.

The final line is the **evasion attack**. There, the first state argument of
`Q_RS` remains the clean simulator state and only the observation consumed by
the frozen victim policy is perturbed. Minimizing `V(s_adv)` is not treated as
Robust-Sarsa. These two uses of state must not be conflated.

## Categorical PPO adaptation

The paper evaluates continuous-action PPO/DDPG. This project targets SB3 PPO
with a `Discrete` action space, including the nine-action SUMO contract.
Actions in the SARSA critic are one-hot vectors. During critic training, the
inner optimizer relaxes each one-hot vector to its coordinatewise `[0, 1]`
epsilon box; it does not claim that intermediate vectors are a probability
simplex. At attack time the differentiable objective is the categorical
expected value:

```text
sum_a pi(a | s_adv) Q_RS(s_clean, a)
```

The maintained inner optimizer performs finite multi-restart projected-gradient
ascent jointly over state and relaxed one-hot-action inputs. It supports a
scalar state radius or a flat per-feature radius, a separate action radius,
and projects states to both the epsilon neighborhood and the valid environment
observation bounds. Both radii share the configured warmup scale. The worst
candidate is selected separately for every minibatch sample.

After the inner search, candidates are detached. The outer squared-deviation
loss performs fresh critic evaluations at adversarial and clean inputs, so
gradients still flow to critic parameters. The original method computes an
IBP/CROWN convex-relaxation upper bound for the concatenated input
neighborhood. Finite PGD is non-convex and may underestimate the true inner
maximum; it is neither a certificate nor a paper-exact bound. Consequently all
maintained artifacts and reports use:

```text
reproduction_level = clean_room_categorical_adaptation
```

They must not be labeled an official or exact paper-code reproduction. An
action-only or state-only regularizer is an ablation, not a maintained
Robust-Sarsa reproduction. The training configuration, checkpoint writer, and
checkpoint loader reject such an artifact if it is labeled `robust_sarsa`.

## Victim action-mode contract

The victim action rule is part of every training, checkpoint, attack, and
evaluation contract. Two modes are supported and may not be pooled:

- `stochastic_sample`: rollouts and execution sample from the categorical
  policy. The optimized categorical expected Q is exact in expectation. This
  is the only mode used by the formal P3 Robust-Sarsa configuration.
- `deterministic_greedy`: rollouts and execution use `argmax(logits)`. Because
  argmax is non-differentiable, the attack is explicitly named
  `softmax_expected_q_surrogate_for_deterministic_greedy`; it is a smooth
  surrogate, not an execution-exact objective.

Reports must include `victim_action_mode`, `objective_contract.name`, and
`objective_contract.execution_alignment`. A critic trained under one mode is
rejected when an attack requests the other.

## Source and dependency isolation

`third_party/upstream-lock.json` pins the authors' `SA_PPO` repository at
commit `7f5193e770bc4b31dd7c1ddc6a866b28ba816659`. Its root license is unresolved,
so it is reference-only. Maintained code does not import it and was written
clean-room from the paper, supplemental algorithm, and the pinned semantic
reference above.

## Artifact requirements

Every critic checkpoint contains:

- the critic architecture and one-hot action encoding;
- the complete training configuration and deterministic seed;
- transition count and transition-data SHA-256;
- externally expected and observed victim-checkpoint SHA-256 values;
- the checkpoint-loaded and in-memory complete policy-state SHA-256 values,
  covering parameters and persistent buffers, plus separate parameter and
  buffer hashes/counts;
- an explicit action mode and `frozen: true` record with pre/post state hashes,
  evaluation mode, and proof that no parameter requires gradients;
- final and mean TD/robustness losses;
- a `training.regularizer` record binding the effective flattened state
  radius, separate action radius, valid state bounds and their source, joint
  PGD steps and restarts, per-sample selection, shared warmup, inner-detach and
  outer-gradient semantics, and the explicit
  `finite_nonconvex_pgd_approximation_not_certified_upper_bound` claim;
- the declared fidelity differences above.

The critic file is loaded only when its caller supplies the expected file
SHA-256. Saving also writes a mandatory adjacent `<checkpoint>.manifest.json`
using strict JSON (`NaN`/`Infinity` forbidden); loading verifies the external
digest, sidecar binding, embedded/sidecar equality, joint-regularizer contract,
and critic-state hash before returning the frozen critic.

An attack result records actual policy queries and gradient evaluations. Hard
budgets are checked before the first victim query. Only the custom numerical or
disconnected-gradient failure classes may return a zero perturbation. Such a
result always has `fallback_occurred=true`, `result_valid=false`,
`evaluation_status=invalid_fallback`, and a stable `fallback_reason_code`; an
audit must invalidate it rather than count it as robustness. Shape, device,
out-of-memory, and arbitrary policy/runtime errors propagate fail-closed. At
epsilon zero, the identity result is valid and is not a fallback.
