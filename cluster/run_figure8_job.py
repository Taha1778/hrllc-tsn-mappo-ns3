#!/usr/bin/env python3
"""Run one isolated Figure 8 point from the source-Python reference suite.

Each invocation owns one method/K combination and writes exactly one JSON
result.  It deliberately never reads or writes the normal ``runs/`` or
``weights/`` folders, so Ray workers can run these jobs concurrently.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from paper_reference_full_suite import (  # noqa: E402
    BCDBaseline,
    CPPOBaseline,
    DQNBaseline,
    Evaluator,
    MADDPGBaseline,
    MAPPOTrainer,
    RandomBaseline,
    SystemConfig,
)

METHODS = ("MAPPO-H", "MAPPO-M", "CPPO", "DQN", "MADDPG", "BCD", "Random")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate_point(method: str, k: int, seed: int, episodes: int, scenarios: int) -> dict:
    """Preserve Figure 8's method-specific seeds while isolating one point."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    config = SystemConfig(random_seed=seed)
    config.K_equipments = k
    evaluator = Evaluator(config, seed=seed + k * 1000)

    if method == "MAPPO-H":
        trainer = MAPPOTrainer(config, variant="H", use_mse=False, seed=seed + k * 2000)
        trainer.train(num_episodes=episodes, verbose=False)
        return evaluator.evaluate_mappo(trainer.agent, num_scenarios=scenarios)
    if method == "MAPPO-M":
        trainer = MAPPOTrainer(config, variant="M", use_mse=True, seed=seed + k * 3000)
        trainer.train(num_episodes=episodes, verbose=False)
        return evaluator.evaluate_mappo(trainer.agent, num_scenarios=scenarios)
    if method == "CPPO":
        return evaluator.evaluate_baseline(
            CPPOBaseline(config, seed=seed + k * 4000), "cppo", num_scenarios=scenarios
        )
    if method == "DQN":
        return evaluator.evaluate_baseline(
            DQNBaseline(config, seed=seed + k * 5000), "dqn", num_scenarios=scenarios
        )
    if method == "MADDPG":
        return evaluator.evaluate_baseline(
            MADDPGBaseline(config, seed=seed + k * 6000), "maddpg", num_scenarios=scenarios
        )
    if method == "BCD":
        return evaluator.evaluate_baseline(BCDBaseline(config), "bcd", num_scenarios=scenarios)
    if method == "Random":
        return evaluator.evaluate_baseline(
            RandomBaseline(config, seed=seed + k * 7000), "random", num_scenarios=scenarios
        )
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated source-Python Figure 8 point")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--k", required=True, type=int, choices=(2, 4, 6, 8, 10, 12))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.episodes < 1 or args.scenarios < 1:
        parser.error("--episodes and --scenarios must both be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "figure8_point.json"
    if result_path.exists():
        raise SystemExit(f"Refusing to overwrite an existing result: {result_path}")

    metrics = evaluate_point(args.method, args.k, args.seed, args.episodes, args.scenarios)
    payload = {
        "provenance": "source_python_reference",
        "figure": 8,
        "method": args.method,
        "k": args.k,
        "seed": args.seed,
        "episodes": args.episodes,
        "scenarios": args.scenarios,
        "git_commit": git_commit(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
