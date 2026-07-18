# P2 defense fidelity contract

This document defines what each maintained defense name means. A result table
must include the `method_key` and `reproduction_level`; the display name alone
is not sufficient.

## Vanilla PPO

`vanilla_ppo` is the locked Stable-Baselines3 2.3.2 PPO implementation with no
robustness loss. It is both the clean-performance reference and the parent
implementation for the maintained robust variants.

## Adv-PPO

`adv_ppo` is an engineering adversarial-training baseline. For every PPO
minibatch it constructs a bounded adversarial observation and adds its PPO
loss to the clean PPO objective:

```text
L = L_PPO(clean) + lambda_adv L_PPO(adversarial)
```

The resolved budget is fixed and recorded for each run; budget selection is a
separate validation sweep. The rollout always comes from the simulator's clean
state; only the actor observation used by the training loss is perturbed.

There is no single external implementation that this project calls the unique
"official Adv-PPO." All results therefore use reproduction level
`engineering_baseline`.

## SA-PPO

The NeurIPS 2020 SA-MDP method adds a robust policy regularizer that constrains
the worst policy divergence in a bounded state neighborhood. The maintained
categorical form is:

```text
R_SA(s) = max_{s' in B(s)} KL(pi(.|s) || pi(.|s'))
L = L_PPO + lambda_sa mean(R_SA)
```

The clean distribution is detached in the inner maximization. Random
initialization is required because the KL input gradient is zero at `s'=s`.
The maintained implementation uses projected gradient ascent and linearly
warms epsilon from zero to the resolved target during the first 75% of
training. The original paper also studies SGLD and convex-relaxation solvers;
those remain paper-fidelity checks in the locked `SA_PPO` checkout.

Method tables must use `sa_ppo` with reproduction level
`clean_room_objective`, not "official SA-PPO."

Primary reference:
<https://proceedings.neurips.cc/paper/2020/hash/f0eb6568ea114ba6e293f903c34d7488-Abstract.html>

## CAR-PPO

CAR-RL targets an infinity measurement error rather than an average
state-divergence penalty. The maintained categorical clean-room objective:

1. uses PGD to maximize each sample's negative clipped PPO surrogate;
2. forms the per-sample score from that adversarial loss and clean policy
   entropy;
3. computes detached `alpha = softmax(score / lambda)` across the minibatch;
4. optimizes clean PPO plus `kappa * sum(alpha * score)`;
5. linearly warms epsilon during the first 75% of training.

Optional PGD restarts are reduced by selecting the worst candidate separately
for every sample before the minibatch weighting. Finite PGD and sampled
minibatches approximate, but cannot equal, the state-space supremum. The method
is therefore recorded as `car_ppo` with reproduction level
`clean_room_objective`, not as a paper-code run.

Primary reference: <https://arxiv.org/abs/2502.16734>

## IBP greedy-action certificate

`ibp_certificate` propagates an input interval through the categorical actor
and audits the clean greedy action's lower-bounded logit margin. In P2 this is
an evaluation component, not a fifth training recipe. Its certificate is local
and one-step:

```text
lower(logit_y) > max_{a != y} upper(logit_a)
```

This proves that the greedy action is unchanged within the specified box for a
supported feed-forward network. It does not certify episode return, collision
avoidance, or closed-loop dynamics.

## Required comparison

Every P2 report includes, on identical victim and episode seeds:

```text
vanilla_ppo
adv_ppo
sa_ppo
car_ppo
```

Clean return, attack return, actual perturbation, training time, and
worst-over-attacks are mandatory. A defense cannot be selected using test
episodes. Report `ibp_certificate` coverage and margin as auxiliary metrics
when the actor architecture is supported; do not present it as an IBP-trained
PPO model.
