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

Phase P5 adds one stateful defense matrix:

```text
frozen victim x frozen defense seed x paired test episode seed x
  Clean
  (FGSM / PGD / MAD /
   Robust-Sarsa / PA-AD / STFA)
    x (non-adaptive / defense-aware)
```

Every test seed has all 13 cells. A missing, failed, duplicate, wrong-budget,
or wrong-binding cell invalidates the robust summary. Each defense-aware
attacker is trained for one frozen victim/defense bundle on the separate
`attacker_train` cohort. Its manifest binds the defense, thresholds, anchor,
purifier, certificate, fallback/shield, attack budgets, and attacker-training
episode/scenario split.

RAPID-Guard uses four disjoint seed roles:

```text
train          detector/denoiser fitting and defense attack exposure
validation     clean conformal threshold and every model/hyperparameter choice
attacker_train per-defense adaptive attacker fitting
test           one frozen final audit
```

Temporal detector windows remain inside one episode, scenario, and split.
Normalizer and detector preprocessing statistics are train-only. No test
trajectory, outcome, detector score, anchor state, or attack result may feed
training, calibration, early stopping, model selection, or attack selection.

Calibrated online detection requires two consecutive, caller-attested
attack-free trusted frames from the current episode. Repeating the first frame
to manufacture a three-frame window is forbidden. No-history, single-frame,
and post-gap decisions fail closed before victim-policy or IBP queries, use the
legal fallback, and are reported separately outside H1 coverage. A fallback
invalidates history continuity until an explicit bound rebootstrap.

P5 has three pre-registered hypotheses:

- H1: fused detection improves a held-out attack endpoint at a fixed clean
  false-positive operating point;
- H2: purification improves a held-out policy/safety recovery endpoint subject
  to clean-distortion and task-cost limits;
- H3: the complete Guard improves independently selected episode-wise worst
  safety and worst utility subject to clean-cost and latency limits.

Each hypothesis has a frozen baseline, effect direction, minimum effect or
non-inferiority margin, paired confidence level, and failure rule before test
data are opened. An implementation test or training manifest cannot satisfy a
hypothesis.

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

For P5, worst utility and safety are selected independently for every paired
episode:

```text
min return
max collision count
max near-miss count
max registered safety cost
```

The source attack/adaptivity cell is retained per endpoint. Clean cost includes
paired return and safety-cost deltas against the undefended frozen anchor,
detector false positives, purifier distortion, intervention/fallback rates,
and certificate abstention. H1 additionally reports episode-aware TPR,
precision-recall, ROC, and seen/unseen-family results; H2 reports recovery
success conditional on a detected attacked opportunity.

RAPID-Guard latency uses synchronized, batch-one measurements after warm-up.
Report end-to-end and detector/proposal/projection/certificate/critic/fallback/
shield p50, p95, and p99 by clean/attack cell together with hardware, software,
and sample counts. A percentile pooled over the entire attack matrix is
diagnostic, not a comparative latency claim.

Defense accounting retains policy, detector, detector-policy, proposal,
projection, purification-attempt, certificate, certificate-policy, IBP-bound,
critic, fallback, shield, and Guard calls. Attack accounting separately retains
victim/defense forward and backward calls, EOT samples, BPDA surrogate calls,
and simulator calls. These are non-fungible currencies.

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
- P5 algorithm/contract, SB3 integration, public-driving integration,
  public-driving empirical effectiveness, SUMO contract integration, and SUMO
  empirical effectiveness are also separate evidence fields. Until a stable,
  immutable SUMO PPO victim exists, the last field is forced false.
- Hard detector/certificate/fallback/shield gates require a gradient-masking
  audit. Use direct gradients where supported, explicit BPDA for a declared
  surrogate, genuine multi-sample EOT for randomness, and transfer/score-query
  checks. Attack steps, restarts, EOT samples, and query budget must show
  convergence. A black-box or transfer attack that persistently exceeds the
  purported adaptive white-box attack invalidates, rather than strengthens,
  the robustness claim.
- A one-step IBP margin certifies only invariance of the clean greedy actor
  action inside the exact policy-input box for supported MLPs. It never
  certifies the Guard, episode return, collision avoidance, dynamics, or
  physical realizability.
