# P12 CartPole attack/defense smoke — 2026-08-09

## Status and claim boundary

This run completed the full train → freeze → attack → aggregate → rebuild/verify
loop on commit `5e615859c9ac608f57acdc0ea54b5d5d22783ae7` with a clean worktree.
`verify` returned `status=verified`, 36 shards, 180 episode rows, and run
fingerprint
`869fc9eac5337314390e3d143aafcef0c1d392edd6f908dfe4201395bb00cd78`.

This is deliberately a **non-formal smoke result**. Its only formal
ineligibility reasons are `claim_tier_is_smoke`, `claim_tier_is_not_formal`,
and `cohort_is_not_test`. It uses one training seed and five paired evaluation
episodes, so confidence intervals are diagnostic only and no publication-level
robustness claim is made.

The final local bundle manifest SHA-256 is
`20de7e9e7b13d7dc89a30d444ae5cebd9265c21126ae5826fd8c8d575884ce16`.
Internal SHA-256 records detect corruption and inconsistent rewrites; they are
not a cryptographic signature.

## Environment

- Saved environment name: `RL_Attack_Core_Py310`
- Interpreter: `E:\RL_Attack\.venv\Scripts\python.exe`
- Python 3.10.16, PyTorch 2.0.0 CPU, Stable-Baselines3 2.3.2,
  Gymnasium 0.29.1, NumPy 1.23.0
- `.venv` has `include-system-site-packages=false`; no package was installed in
  or changed in the user's active Conda `pytorch` environment.
- CPU thread pools were limited to one thread per process. CUDA was not used.

## Frozen protocol

- Environment: `CartPole-v1`, deterministic actions, maximum 500 steps.
- Training: seed 0, requested 100,000 steps, actual 100,352 steps because PPO
  completes whole 1,024-step rollouts; 100 clean evaluation episodes using
  seeds 10,000–10,099.
- Test cohort: paired seeds 20,000–20,004.
- Per-feature policy-input L∞ base epsilon: `[0.05, 0.05, 0.01, 0.01]`;
  ratios `[0, 1]`; every feature mutable.
- Attacks: Random Uniform, FGSM-CE, PGD-CE 5 steps × 2 restarts, and
  categorical MAD-PGD 5 steps × 2 restarts.
- Attack probability: 1.0. Maximum attack-internal cost per attacked step:
  16 policy queries and 10 gradient evaluations.
- Statistics: 1,000 crossed hierarchical bootstrap replicates, 95% interval,
  CVaR α=0.10.

The policy-query count includes only logits forwards inside the attack solver.
It excludes clean action selection and the final action selected after attack.
FGSM used 3 queries/1 gradient per attacked step; PGD and MAD used
13 queries/10 gradients; Random used 0/0.

## Implemented method semantics

- **Vanilla PPO**: standard maintained SB3 PPO without a robust loss.
- **Adv-PPO**: PGD observations are used in an adversarial PPO clipped-surrogate
  objective with coefficient 1.0. This is an engineering baseline, not an exact
  reproduction of one uniquely defined paper method.
- **SA-PPO**: clean-room state-adversarial policy-consistency objective with a
  PGD inner solver, coefficient 1.0, and epsilon ramp over the first 75% of
  training. This is not paper-exact SGLD/convex-relaxation code.
- **CAR-PPO**: clean-room categorical, per-sample adversarial clipped loss with
  detached soft-CAR weights (`lambda=0.1`) and a finite PGD approximation. It
  is neither paper-exact nor a certified defense.
- **Random Uniform**: black-box uniform perturbation inside the feature-wise
  L∞ envelope.
- **FGSM-CE / PGD-CE**: maximize cross-entropy against the clean greedy action,
  in one step or projected multi-step restarts respectively.
- **Categorical MAD-PGD**: maximize categorical policy KL divergence inside the
  same projected envelope.

## Training clean performance

| Method | Clean episodes | Mean return | Median | Std. dev. |
|---|---:|---:|---:|---:|
| Vanilla PPO | 100 | 500.00 | 500.00 | 0.00 |
| Adv-PPO | 100 | 500.00 | 500.00 | 0.00 |
| SA-PPO | 100 | 499.87 | 500.00 | 1.20 |
| CAR-PPO | 100 | 477.29 | 500.00 | 52.35 |

CAR-PPO already has a clean-performance deficit and high dispersion at this
configuration. It is therefore not on the same clean/robust Pareto frontier as
the other three policies in this seed.

## Paired attack result at epsilon ratio 1

The clean return below uses the five paired test seeds, not the 100-episode
training-time clean cohort.

| Method | Paired clean return | Worst-over-attacks return | Paired drop | CVaR₀.₁ | Random flip | FGSM flip | PGD flip | MAD flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla PPO | 500.0 | 500.0 | 0.0 | 500.0 | 10.32% | 44.92% | 44.92% | 36.32% |
| Adv-PPO | 500.0 | 500.0 | 0.0 | 500.0 | 9.40% | 15.64% | 15.64% | 13.48% |
| SA-PPO | 500.0 | 500.0 | 0.0 | 500.0 | 9.24% | 16.96% | 16.96% | 15.68% |
| CAR-PPO | 419.8 | 412.6 | 7.2 | 311.0 | 8.54% | 16.77% | 16.73% | 13.83% |

Observed smoke trends:

- Adv-PPO and SA-PPO reduce PGD action flips from Vanilla's 44.92% to 15.64%
  and 16.96% respectively (about 65% and 62% relative reductions).
- Vanilla, Adv-PPO, and SA-PPO still achieve the CartPole return ceiling in all
  five attacked episodes. Return saturation therefore hides the substantial
  action changes and does **not** prove equivalent robustness.
- CAR-PPO is unstable on the paired cohort. Some individual attacks even raise
  its return relative to its weak clean trajectory; the per-episode
  worst-over-attacks aggregate is 412.6 with paired drop 7.2 and CVaR 311.
  The diagnostic 95% bootstrap interval for that drop is `[-1.2, 19.4]` and
  includes zero. This is evidence against the selected CAR hyperparameters,
  not evidence that attacks generally improve CAR-PPO or that CAR is
  significantly more attack-sensitive.
- At this epsilon and 5×2 smoke budget, PGD-CE does not improve the return or
  flip result over FGSM-CE. A 20×5 development run is required before comparing
  strong-attack quality.
- Collision, crashed, and on-road fields are unavailable under the CartPole
  contract. This run provides no driving-safety evidence.

## Reproduction commands

Run from `E:\RL_Attack` in PowerShell. These commands use only the project
`.venv` and write ignored artifacts under `outputs/`.

The bundle freezes code commit `5e61585`; check out that commit before exact
reproduction or re-verification. A later documentation-only commit changes the
repository fingerprint by design even though it does not change scientific
code.

```powershell
$py = 'E:\RL_Attack\.venv\Scripts\python.exe'
$out = 'E:\RL_Attack\outputs\p12_cartpole_smoke_20260809'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

& $py -m rl_attack.cli.defense_baseline --method vanilla_ppo --env-id CartPole-v1 --timesteps 100000 --eval-episodes 100 --eval-seed-start 10000 --seed 0 --device cpu --output-dir $out --run-name vanilla_ppo_seed0 --attack none --epsilon 0 --epsilon-schedule-fraction 0 --adversarial-loss-coef 0 --policy-consistency-coef 0 --value-consistency-coef 0 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-epochs 10 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0 --vf-coef 0.5 --max-grad-norm 0.5

& $py -m rl_attack.cli.defense_baseline --method adv_ppo --env-id CartPole-v1 --timesteps 100000 --eval-episodes 100 --eval-seed-start 10000 --seed 0 --device cpu --output-dir $out --run-name adv_ppo_seed0 --attack pgd --epsilon 0.02 --attack-steps 10 --attack-step-size 0.005 --attack-restarts 1 --attack-random-start --epsilon-schedule-fraction 0 --adversarial-loss-coef 1 --policy-consistency-coef 0 --value-consistency-coef 0 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-epochs 10 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0 --vf-coef 0.5 --max-grad-norm 0.5

& $py -m rl_attack.cli.defense_baseline --method sa_ppo --env-id CartPole-v1 --timesteps 100000 --eval-episodes 100 --eval-seed-start 10000 --seed 0 --device cpu --output-dir $out --run-name sa_ppo_seed0 --attack pgd --epsilon 0.02 --attack-steps 10 --attack-step-size 0.005 --attack-restarts 1 --attack-random-start --epsilon-schedule-fraction 0.75 --adversarial-loss-coef 0 --policy-consistency-coef 1 --value-consistency-coef 0 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-epochs 10 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0 --vf-coef 0.5 --max-grad-norm 0.5

& $py -m rl_attack.cli.defense_baseline --method car_ppo --env-id CartPole-v1 --timesteps 100000 --eval-episodes 100 --eval-seed-start 10000 --seed 0 --device cpu --output-dir $out --run-name car_ppo_seed0 --attack pgd --epsilon 0.02 --attack-steps 10 --attack-step-size 0.005 --attack-restarts 1 --attack-random-start --epsilon-schedule-fraction 0.75 --car-soft-lambda 0.1 --adversarial-loss-coef 1 --policy-consistency-coef 0 --value-consistency-coef 0 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-epochs 10 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0 --vf-coef 0.5 --max-grad-norm 0.5

& $py -m rl_attack.cli.p12_benchmark plan "$out\benchmark.yaml" --device cpu
& $py -m rl_attack.cli.p12_benchmark run "$out\benchmark.yaml" --output-dir "$out\benchmark_bundle" --device cpu
& $py -m rl_attack.cli.p12_benchmark verify "$out\benchmark_bundle"
```

The exact ignored runtime configuration is
`E:\RL_Attack\outputs\p12_cartpole_smoke_20260809\benchmark.yaml`. It pins all
four model and training-manifest SHA-256 values.

## Next formal experiment

A development result must use at least five training seeds per method, at least
200 held-out paired episodes, PGD/MAD at least 20 steps × 5 restarts, attack
probability 1.0, and at least 10,000 crossed bootstrap replicates. The planned
epsilon sweep should restore the project grid `[0, 0.25, 0.5, 1, 2]` and report
both return degradation and policy-level action flips. CAR-PPO should first be
tuned on the separate validation cohort; test seeds must remain inaccessible
during selection.
