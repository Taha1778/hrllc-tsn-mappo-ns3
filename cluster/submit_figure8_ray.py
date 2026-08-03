#!/usr/bin/env python3
"""Submit Figure 8's independent method/K points to an existing Ray cluster."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import ray

METHODS = ("MAPPO-H", "MAPPO-M", "CPPO", "DQN", "MADDPG", "BCD", "Random")
K_VALUES = (2, 4, 6, 8, 10, 12)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = PROJECT_ROOT / "cluster" / "run_figure8_job.py"


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported methods: {', '.join(invalid)}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribute source-Python Figure 8 points using Ray")
    parser.add_argument("--address", default="auto", help="Ray address; use auto from the Ray head pod")
    parser.add_argument("--results-root", required=True, help="Shared persistent path mounted on every Ray node")
    parser.add_argument("--methods", type=parse_methods, default=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--cpus-per-job", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [
        {
            "method": method,
            "k": k,
            "output_dir": str(Path(args.results_root) / method.lower() / f"k-{k:02d}"),
        }
        for method, k in itertools.product(args.methods, K_VALUES)
    ]
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return

    ray.init(address=args.address)

    @ray.remote(num_cpus=args.cpus_per_job)
    def run_job(job: dict) -> dict:
        import subprocess

        command = [
            sys.executable,
            str(JOB_SCRIPT),
            "--method",
            job["method"],
            "--k",
            str(job["k"]),
            "--seed",
            str(args.seed),
            "--episodes",
            str(args.episodes),
            "--scenarios",
            str(args.scenarios),
            "--output-dir",
            job["output_dir"],
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            **job,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    outcomes = ray.get([run_job.remote(job) for job in jobs])
    failed = [outcome for outcome in outcomes if outcome["returncode"] != 0]
    print(json.dumps(outcomes, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} Figure 8 jobs failed; see the returned stderr fields.")


if __name__ == "__main__":
    main()
