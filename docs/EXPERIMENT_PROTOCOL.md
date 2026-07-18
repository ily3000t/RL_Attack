# Experiment protocol

## Dataset split and seeds

- Victim training seeds: at least 5 during development, 10 for final results.
- Learned attacker/defense seeds: 3 during development, 5 for final results.
- Traffic/task seeds are split once into train, validation, and test cohorts.
- Attack hyperparameters are selected only on validation seeds.
- Every clean/attack comparison uses paired victim and episode seeds.

## Evaluation matrix

Phase P1:

```text
Vanilla PPO x
  Clean / Random-Uniform / Random-Sign /
  FGSM-CE / PGD-CE / Categorical-MAD
```

Spatial budget multipliers:

```text
0, 0.25, 0.5, 1, 2
```

Temporal attack fractions:

```text
0.05, 0.10, 0.25, 0.50, 1.00
```

PGD development audit uses 10 steps/1 restart; final audit uses at least
20 steps/5 restarts after a step-size convergence check.

## Metrics

All tasks report return, length, actual perturbation norms, attack count,
policy queries, gradient evaluations, and latency.

Driving tasks additionally report:

- merge success, collision, near-miss, and merge failure rates;
- TTC lower quantiles and DRAC upper quantiles;
- attack success rate and conditional ASR over episodes cleanly successful;
- safety loss per attacked decision;
- clean/robust Pareto curve and worst-over-attacks score.

Results use mean with confidence interval plus median and tail quantiles.
Episode-level paired bootstrap is the default comparison; multiple-method
claims require family-wise or false-discovery correction.

## Fairness rules

- The victim checkpoint and observation normalization are frozen.
- Attacks compared in one table share spatial, temporal, and query budgets.
- White-box, transfer, score-query, and simulator-query attacks are separate.
- Observation, action, reward, training-data, parameter, and backdoor attacks
  are separate leaderboards.
- The deterministic policy result is primary; stochastic evaluation uses a
  shared action-randomness stream and is supplementary.
- A defense is tuned against a declared training attack and audited against
  unseen/adaptive attacks.
