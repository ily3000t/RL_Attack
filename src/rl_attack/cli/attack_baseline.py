from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from rl_attack.attacks.observation import (
    CategoricalMADPGDAttack,
    FGSMCEAttack,
    PGDCEAttack,
    PerturbationBounds,
    RandomSignAttack,
    RandomUniformAttack,
)
from rl_attack.evaluation import evaluate_sb3_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/load and smoke-test a categorical PPO victim")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument(
        "--attack",
        choices=["clean", "random-uniform", "random-sign", "fgsm-ce", "pgd-ce", "mad-pgd"],
        default="clean",
    )
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--attack-probability", type=float, default=1.0)
    return parser


def _build_attack(args, env: gym.Env):
    if args.attack == "clean":
        return None
    space = env.observation_space
    if not isinstance(space, gym.spaces.Box):
        raise TypeError("observation attacks require a Box observation space")
    mutable = np.ones(space.shape, dtype=bool)
    bounds = PerturbationBounds(
        epsilon=np.full(space.shape, args.epsilon, dtype=np.float32),
        lower=space.low,
        upper=space.high,
        mutable_mask=mutable,
    )
    if args.attack == "random-uniform":
        return RandomUniformAttack(bounds)
    if args.attack == "random-sign":
        return RandomSignAttack(bounds)
    if args.attack == "fgsm-ce":
        return FGSMCEAttack(bounds)
    if args.attack == "pgd-ce":
        return PGDCEAttack(bounds, steps=args.steps, restarts=args.restarts)
    return CategoricalMADPGDAttack(
        bounds,
        steps=args.steps,
        restarts=args.restarts,
    )


def main() -> None:
    args = _parser().parse_args()

    def env_factory():
        return gym.make(args.env_id)

    probe_env = env_factory()
    try:
        if not isinstance(probe_env.action_space, gym.spaces.Discrete):
            raise TypeError("this baseline CLI requires a Discrete action space")
        attack = _build_attack(args, probe_env)
    finally:
        probe_env.close()

    if args.load_model:
        model = PPO.load(args.load_model, device=args.device)
    else:
        training_env = env_factory()
        model = PPO(
            "MlpPolicy",
            training_env,
            seed=args.seed,
            verbose=0,
            device=args.device,
        )
        model.learn(total_timesteps=args.timesteps)
        training_env.close()
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)

    results = evaluate_sb3_policy(
        model,
        env_factory,
        episode_seeds=range(10_000, 10_000 + args.episodes),
        attack=attack,
        attack_probability=args.attack_probability,
        attack_seed=args.seed + 1_000_000,
        deterministic=True,
    )
    returns = np.asarray([result.episode_return for result in results], dtype=np.float64)
    summary = {
        "env_id": args.env_id,
        "attack": args.attack,
        "episodes": len(results),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "attack_count": int(sum(result.attack_count for result in results)),
        "policy_queries": int(sum(result.policy_queries for result in results)),
        "gradient_evaluations": int(
            sum(result.gradient_evaluations for result in results)
        ),
        "perturbation_linf_max": float(
            max((result.perturbation_linf_max for result in results), default=0.0)
        ),
        "episode_results": [result.to_dict() for result in results],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
