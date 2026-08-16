#!/usr/bin/env python3
"""Submit Figure 8's independent method/K points to an existing Ray cluster."""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

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


def parse_k_values(value: str) -> tuple[int, ...]:
    try:
        k_values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers") from error
    invalid = sorted(set(k_values) - set(K_VALUES))
    if not k_values or invalid:
        supported = ", ".join(str(item) for item in K_VALUES)
        raise argparse.ArgumentTypeError(
            f"Unsupported K values: {invalid}; supported values are {supported}"
        )
    return k_values


def source_commit() -> str:
    configured_commit = os.environ.get("PROJECT_GIT_COMMIT", "").strip()
    if configured_commit:
        return configured_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribute source-Python Figure 8 points using Ray")
    parser.add_argument("--address", default="auto", help="Ray address; use auto from the Ray head pod")
    parser.add_argument("--results-root", required=True, help="Shared persistent path mounted on every Ray node")
    parser.add_argument("--methods", type=parse_methods, default=METHODS)
    parser.add_argument("--k-values", type=parse_k_values, default=K_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--cpus-per-job", type=float, default=1.0)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum concurrent Ray tasks; 0 uses the cluster CPU capacity",
    )
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=14400,
        help="Fail one point if its subprocess exceeds this time",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cpus_per_job <= 0:
        parser.error("--cpus-per-job must be positive")
    if args.max_in_flight < 0:
        parser.error("--max-in-flight cannot be negative")
    if args.job_timeout_seconds <= 0:
        parser.error("--job-timeout-seconds must be positive")

    jobs = [
        {
            "method": method,
            "k": k,
            "git_commit": source_commit(),
            "output_dir": str(Path(args.results_root) / method.lower() / f"k-{k:02d}"),
        }
        for method, k in itertools.product(args.methods, args.k_values)
    ]
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return

    import ray

    ray.init(address=args.address)

    @ray.remote(num_cpus=args.cpus_per_job)
    def run_job(job: dict) -> dict:
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
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.job_timeout_seconds,
                env={**os.environ, "PROJECT_GIT_COMMIT": job["git_commit"]},
            )
        except subprocess.TimeoutExpired as error:
            return {
                **job,
                "returncode": 124,
                "stdout": error.stdout or "",
                "stderr": f"Job timed out after {args.job_timeout_seconds} seconds",
            }
        return {
            **job,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    cluster_cpus = max(1, int(ray.cluster_resources().get("CPU", 1)))
    max_in_flight = args.max_in_flight or max(1, int(cluster_cpus / args.cpus_per_job))
    max_in_flight = min(max_in_flight, len(jobs))
    pending_jobs = iter(jobs)
    active = {}
    outcomes = []

    for _ in range(max_in_flight):
        job = next(pending_jobs, None)
        if job is None:
            break
        active[run_job.remote(job)] = job

    while active:
        ready, _ = ray.wait(list(active), num_returns=1)
        task = ready[0]
        job = active.pop(task)
        outcome = ray.get(task)
        outcomes.append(outcome)
        print(
            f"progress={len(outcomes)}/{len(jobs)} method={job['method']} "
            f"k={job['k']} returncode={outcome['returncode']}",
            flush=True,
        )
        next_job = next(pending_jobs, None)
        if next_job is not None:
            active[run_job.remote(next_job)] = next_job

    outcomes.sort(key=lambda outcome: (METHODS.index(outcome["method"]), outcome["k"]))
    failed = [outcome for outcome in outcomes if outcome["returncode"] != 0]
    print(json.dumps(outcomes, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} Figure 8 jobs failed; see the returned stderr fields.")


if __name__ == "__main__":
    main()
