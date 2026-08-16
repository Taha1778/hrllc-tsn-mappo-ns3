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
    candidate_name = task.get("candidate", {}).get("name", "default")
    fields = ("phase", "variant", "axis", "value", "seed")
    return "__".join([f"candidate-{candidate_name}", *(f"{key}-{task[key]}" for key in fields)])


def persisted_task(task: dict) -> dict:
    """Return the stable task record stored alongside each isolated result."""
    return {key: value for key, value in task.items() if key not in {"base_config", "candidate"}}


def make_config(base: dict, candidate: dict, variant: str) -> dict:
    config = copy.deepcopy(base)
    config["enforce_paper_configuration"] = False
    config["extension_label"] = "paper_method_hyperparameter_tuning_extension"
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
        if (
            existing.get("task") == persisted_task(task)
            and existing.get("candidate") == task["candidate"]
            and existing.get("git_commit") == commit
        ):
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
    metrics_path = job_dir / "weights" / "best_metrics.json"
    payload = {
        "task": persisted_task(task),
        "candidate": task["candidate"], "git_commit": commit,
        "source": "native_ns3_cpp_with_python_mappo_validation", "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3), "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode == 0 and metrics_path.exists():
        payload["metrics"] = load_json(metrics_path)
        payload["qos_score"] = qos_score(payload["metrics"])
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_final_evaluation(task: dict, results_root: str, timeout_seconds: int, commit: str) -> dict:
    """Evaluate only the selected candidate's best validation checkpoint on held-out seeds."""
    job_dir = Path(results_root) / "native-ns3" / task_id(task)
    result_path = job_dir / "final_test_result.json"
    if result_path.exists():
        existing = load_json(result_path)
        if (
            existing.get("task") == persisted_task(task)
            and existing.get("candidate") == task["candidate"]
            and existing.get("git_commit") == commit
        ):
            return existing
        raise RuntimeError(f"Refusing incompatible final-test resume at {job_dir}")
    config = load_json(job_dir / "config.json")
    command = [
        sys.executable, str(ROOT / "python" / "hrllc_tsn_trainer.py"),
        "--config", str(job_dir / "config.json"), "--workspace", str(job_dir),
        "--k", str(config["k"]), "--seed", str(task["seed"]),
        "--radius", str(config["radius_m"]), "--power", str(config["total_power_w"]),
        "--gamma-db", str(config["gamma_threshold_db"]),
        "--final-test-seeds", str(task["final_test_seeds"]), "--final-evaluate-best",
    ]
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_seconds)
    metrics_path = job_dir / "runs" / "final_test_metrics.json"
    payload = {
        "task": persisted_task(task), "candidate": task["candidate"], "git_commit": commit,
        "source": "native_ns3_cpp_with_python_mappo_held_out_test", "returncode": completed.returncode,
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
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run only the configured primary MAPPO-H candidate after validation has selected it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_in_flight < 1 or args.timeout_seconds < 1:
        parser.error("max-in-flight and timeout-seconds must be positive")
    base, extension = load_json(DEFAULT_CONFIG), load_json(EXTENSION_CONFIG)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    candidates = extension["selection_candidates"]
    if args.primary_only:
        primary_name = extension["primary_candidate"]
        candidates = [candidate for candidate in candidates if candidate["name"] == primary_name]
        if len(candidates) != 1:
            parser.error("primary_candidate must name exactly one selection candidate")
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
    winner = min(winners, key=winners.get)
    winner_tasks = [item for item in tasks if item["candidate"]["name"] == winner]
    final_worker = ray.remote(num_cpus=1)(run_final_evaluation)
    final_tests = ray.get([
        final_worker.remote(task, args.results_root, args.timeout_seconds, args.commit)
        for task in winner_tasks
    ])
    if any(item["returncode"] != 0 or "metrics" not in item for item in final_tests):
        raise SystemExit("held-out final test failed")
    metric_keys = final_tests[0]["metrics"].keys()
    final_metrics = {
        key: sum(float(item["metrics"][key]) for item in final_tests) / len(final_tests)
        for key in metric_keys
    }
    selection = {
        "git_commit": args.commit, "selection_metric": extension["selection_metric"],
        "scores": winners, "winner": winner, "selection_tasks": outcomes,
        "final_test_candidate": winner, "final_test_metrics": final_metrics,
        "final_tests": final_tests,
    }
    destination = Path(args.results_root) / "native-ns3" / "selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
