#!/usr/bin/env python3
"""Run isolated native ns-3 MAPPO extension experiments through Ray.

Every task receives a private workspace below RESULTS_ROOT.  Source and the
already-built ns-3 executable stay read-only in /workspace; runs/ and weights/
are never shared between Ray workers.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default_config.json"
EXTENSION_CONFIG = ROOT / "configs" / "extension_qos.json"
K_VALUES = (2, 4, 6, 8, 10, 12)
RADII = (100, 150, 200, 250, 300, 350)
POWERS = (20, 30, 40, 50, 60)
GAMMAS = (0, 3, 5, 7, 10)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def qos_score(metrics: dict) -> float:
    return sum(float(metrics[key]) for key in (
        "failure_rate", "latency_violation_rate", "reliability_violation_rate"
    )) / 3.0


def task_id(task: dict) -> str:
    return "__".join(f"{key}-{task[key]}" for key in ("phase", "variant", "axis", "value", "seed"))


def make_config(base: dict, candidate: dict, variant: str) -> dict:
    config = copy.deepcopy(base)
    config["enforce_paper_configuration"] = False
    config["extension_label"] = "policy_tuning_qos_extension"
    config.update({key: value for key, value in candidate.items() if key != "name"})
    config["experiment_variant"] = variant
    config["critic_loss"] = "mse" if variant == "MAPPO-M" else "huber"
    config["random_gcl"] = variant == "MAPPO-R"
    config["simple_reward"] = variant == "MAPPO-S"
    return config


def run_local_task(task: dict, results_root: str, timeout_seconds: int, commit: str) -> dict:
    job_dir = Path(results_root) / "native-ns3" / task_id(task)
    result_path = job_dir / "result.json"
    if result_path.exists():
        existing = load_json(result_path)
        if existing.get("task") == task and existing.get("git_commit") == commit:
            return existing
        raise RuntimeError(f"Refusing incompatible resume at {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(task["base_config"], task["candidate"], task["variant"])
    axis = task["axis"]
    if axis == "k": config["k"] = task["value"]
    if axis == "radius": config["radius_m"] = task["value"]
    if axis == "power": config["total_power_w"] = task["value"]
    if axis == "gamma": config["gamma_threshold_db"] = task["value"]
    config_path = job_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [
        sys.executable, str(ROOT / "python" / "hrllc_tsn_trainer.py"),
        "--config", str(config_path), "--workspace", str(job_dir), "--fresh-start",
        "--rounds", str(task["rounds"]), "--iterations", str(task["iterations"]),
        "--k", str(config["k"]), "--seed", str(task["seed"]),
        "--radius", str(config["radius_m"]), "--power", str(config["total_power_w"]),
        "--gamma-db", str(config["gamma_threshold_db"]),
        "--validation-seeds", str(task["validation_seeds"]),
        "--final-test-seeds", str(task["final_test_seeds"]),
    ]
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_seconds)
    if completed.returncode == 0:
        evaluation = subprocess.run(
            [*command[0:2], "--config", str(config_path), "--workspace", str(job_dir),
             "--k", str(config["k"]), "--seed", str(task["seed"]),
             "--radius", str(config["radius_m"]), "--power", str(config["total_power_w"]),
             "--gamma-db", str(config["gamma_threshold_db"]),
             "--final-test-seeds", str(task["final_test_seeds"]), "--final-evaluate-best"],
            text=True, capture_output=True, timeout=timeout_seconds,
        )
        completed = subprocess.CompletedProcess(
            command, evaluation.returncode, completed.stdout + evaluation.stdout, completed.stderr + evaluation.stderr
        )
    metrics_path = job_dir / "runs" / "final_test_metrics.json"
    payload = {
        "task": {key: value for key, value in task.items() if key not in {"base_config", "candidate"}},
        "candidate": task["candidate"], "git_commit": commit,
        "source": "native_ns3_cpp_with_python_mappo", "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3), "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode == 0 and metrics_path.exists():
        payload["metrics"] = load_json(metrics_path)
        payload["qos_score"] = qos_score(payload["metrics"])
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit isolated native ns-3 MAPPO extension tasks through Ray")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--commit", default=os.environ.get("PROJECT_GIT_COMMIT", "unknown"))
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--validation-seeds", type=int, default=20)
    parser.add_argument("--final-test-seeds", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-in-flight", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_in_flight < 1 or args.timeout_seconds < 1:
        parser.error("max-in-flight and timeout-seconds must be positive")
    base, extension = load_json(DEFAULT_CONFIG), load_json(EXTENSION_CONFIG)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    candidates = extension["selection_candidates"]
    tasks = [
        {"phase": "selection", "variant": "MAPPO-H", "axis": "k", "value": 8, "seed": seed,
         "candidate": candidate, "base_config": base, "rounds": args.rounds, "iterations": args.iterations,
         "validation_seeds": args.validation_seeds, "final_test_seeds": args.final_test_seeds}
        for candidate in candidates for seed in seeds
    ]
    if args.dry_run:
        print(json.dumps([{key: value for key, value in task.items() if key not in {"base_config", "candidate"}} for task in tasks], indent=2))
        return
    import ray
    ray.init(address=args.address)
    worker = ray.remote(num_cpus=1)(run_local_task)
    active, outcomes = {}, []
    pending = iter(tasks)
    for _ in range(min(args.max_in_flight, len(tasks))):
        task = next(pending, None)
        if task: active[worker.remote(task, args.results_root, args.timeout_seconds, args.commit)] = task
    while active:
        ready, _ = ray.wait(list(active), num_returns=1)
        ref = ready[0]
        outcome = ray.get(ref)
        outcomes.append(outcome)
        print(f"progress={len(outcomes)}/{len(tasks)} id={task_id(active.pop(ref))} rc={outcome['returncode']}", flush=True)
        task = next(pending, None)
        if task: active[worker.remote(task, args.results_root, args.timeout_seconds, args.commit)] = task
    failed = [item for item in outcomes if item["returncode"] != 0 or "metrics" not in item]
    if failed:
        raise SystemExit(f"{len(failed)} selection tasks failed")
    winners = {}
    for candidate in candidates:
        measurements = [item for item in outcomes if item["candidate"]["name"] == candidate["name"]]
        winners[candidate["name"]] = sum(item["qos_score"] for item in measurements) / len(measurements)
    selection = {"git_commit": args.commit, "metric": extension["selection_metric"], "scores": winners,
                 "winner": min(winners, key=winners.get), "tasks": outcomes}
    destination = Path(args.results_root) / "native-ns3" / "selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
