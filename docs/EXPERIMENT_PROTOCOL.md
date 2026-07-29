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

Phase P4 adds a sequential, hard-budget matrix:

```text
frozen victim x paired episode seed x
  P3 strongest attacks / STFA objective and semantic ablations
```

STFA must report its full temporal contract (`K`, minimum gap, optional rolling
window), selected and non-zero steps, target and actual actions, continuous
norms, discrete edit cost, and measured policy/gradient/projector/critic/
director/transform calls. A selected zero-delta step still consumes one token.
Random-timing ablations use a pre-sampled fixed-K schedule; Bernoulli attack
probability is not a substitute for the ledger.

Training datasets for the safety critic and director are frozen before the
corresponding fit. Dataset, adjacent manifest, victim, normalization,
environment, cost/labeler, critic, temporal-budget, and horizon hashes are
part of the learned-artifact contract. Training-pipeline manifests are not
formal evaluation results.

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
- Semantic projection and simulator-state attacks are distinct threat models.
  A schema-consistent projected observation is not labelled physically
  realizable without an independent simulator-level guarantee.
- P4 implementation, SB3 nine-action integration, SUMO contract integration,
  and SUMO empirical effectiveness are separate evidence fields. Synthetic or
  dependency-injected runs cannot establish SUMO empirical effectiveness.
