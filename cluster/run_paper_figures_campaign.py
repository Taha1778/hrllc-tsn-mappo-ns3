#!/usr/bin/env python3
"""Run the complete, provenance-safe Figures 7--13 campaign on Ray."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cluster"))
from submit_ns3_campaign_ray import load_json, run_final_evaluation, run_local_task, task_id  # noqa: E402

K_VALUES = (2, 4, 6, 8, 10, 12)
RADII = (100, 150, 200, 250, 300, 350)
POWERS = (20, 30, 40, 50, 60)
GAMMAS = (3, 4, 5, 6, 7)
VARIANTS = ("MAPPO-H", "MAPPO-M", "MAPPO-R", "MAPPO-S")
REFERENCE_METHODS = ("CPPO", "DQN", "MADDPG", "BCD", "Random")


def native_id(task: dict) -> str:
    return "__".join(str(task[key]) for key in ("variant", "axis", "value", "seed"))


def run_native(task: dict, root: str, timeout: int, commit: str) -> dict:
    outcome = run_local_task(task, root, timeout, commit)
    if outcome.get("returncode") != 0:
        return outcome
    final = run_final_evaluation(task, root, timeout, commit)
    final["training_metrics"] = outcome.get("metrics")
    final["training_elapsed_seconds"] = outcome.get("elapsed_seconds")
    final["native_id"] = native_id(task)
    final["job_dir"] = str(Path(root) / "native-ns3" / task_id(task))
    final["point_key"] = task["point_key"]
    return final


def run_reference(task: dict, root: str, commit: str) -> dict:
    sys.path.insert(0, str(ROOT / "python"))
    from paper_reference_full_suite import (  # noqa: E402
        BCDBaseline, CPPOBaseline, DQNBaseline, Evaluator,
        MADDPGBaseline, RandomBaseline, SystemConfig,
    )
    out = Path(root) / "reference" / f"{task['axis']}-{task['value']}" / task["method"].lower()
    result = out / "result.json"
    expected = {**task, "git_commit": commit, "source": "python_reference_baseline"}
    if result.exists():
        saved = load_json(result)
        if saved.get("task") == task and saved.get("git_commit") == commit:
            return saved
        raise RuntimeError(f"Refusing incompatible reference resume at {result}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = SystemConfig(random_seed=task["seed"])
    if task["axis"] == "k":
        cfg.K_equipments = int(task["value"])
    else:
        cfg.area_radius = float(task["value"])
    evaluator = Evaluator(cfg, seed=task["seed"] + int(task["value"]) * 100)
    method = task["method"]
    if method == "CPPO": model, key = CPPOBaseline(cfg, seed=task["seed"]), "cppo"
    elif method == "DQN": model, key = DQNBaseline(cfg, seed=task["seed"]), "dqn"
    elif method == "MADDPG": model, key = MADDPGBaseline(cfg, seed=task["seed"]), "maddpg"
    elif method == "BCD": model, key = BCDBaseline(cfg), "bcd"
    else: model, key = RandomBaseline(cfg, seed=task["seed"]), "random"
    started = time.perf_counter()
    metrics = evaluator.evaluate_baseline(model, key, num_scenarios=task["scenarios"])
    payload = {
        **expected, "returncode": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "scenario_count": task["scenarios"],
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def native_tasks(base: dict, candidate: dict, args) -> list[dict]:
    points = set()
    for variant in ("MAPPO-H", "MAPPO-M"):
        for k in K_VALUES: points.add((variant, "k", k))
        for radius in RADII: points.add((variant, "radius", radius))
    for power in POWERS:
        for gamma in GAMMAS: points.add(("MAPPO-H", "power-gamma", f"{power}:{gamma}"))
        for variant in ("MAPPO-R", "MAPPO-S"): points.add((variant, "power", power))
    tasks = []
    for variant, axis, value in sorted(points):
        for seed in args.seeds:
            task_axis, task_value = axis, value
            task_base = json.loads(json.dumps(base))
            phase = "paper-figures"
            if axis == "power-gamma":
                power, gamma = map(float, value.split(":"))
                task_base["gamma_threshold_db"] = gamma
                task_axis, task_value = "power", power
                phase = f"paper-figures-gamma-{gamma:g}"
            tasks.append({
                "phase": phase, "variant": variant, "axis": task_axis,
                "value": task_value, "seed": seed, "candidate": candidate,
                "base_config": task_base, "rounds": args.rounds,
                "iterations": args.iterations, "validation_seeds": args.validation_seeds,
                "final_test_seeds": args.final_test_seeds,
                "point_key": f"{variant}|{axis}|{value}|{seed}",
            })
    return tasks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="auto")
    p.add_argument("--results-root", required=True)
    p.add_argument("--commit", default=os.getenv("PROJECT_GIT_COMMIT", "unknown"))
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--iterations", type=int, default=256)
    p.add_argument("--validation-seeds", type=int, default=20)
    p.add_argument("--final-test-seeds", type=int, default=100)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--reference-scenarios", type=int, default=100)
    p.add_argument("--max-in-flight", type=int, default=3)
    p.add_argument("--timeout-seconds", type=int, default=14400)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.seeds = tuple(int(x) for x in args.seeds.split(","))
    base = load_json(ROOT / "configs/default_config.json")
    ext = load_json(ROOT / "configs/extension_qos.json")
    selected = ext.get("selected_figure_candidate", "deep-wide-h3-d128-r2")
    candidate = next(x for x in ext["selection_candidates"] if x["name"] == selected)
    natives = native_tasks(base, candidate, args)
    refs = [
        {"method": method, "axis": axis, "value": value, "seed": 42,
         "scenarios": args.reference_scenarios}
        for axis, values in (("k", K_VALUES), ("radius", RADII))
        for value in values for method in REFERENCE_METHODS
    ]
    if args.dry_run:
        print(json.dumps({"native_tasks": len(natives), "reference_tasks": len(refs),
                          "selected_candidate": candidate}, indent=2))
        return
    import ray
    ray.init(address=args.address)
    native_worker = ray.remote(num_cpus=1)(run_native)
    ref_worker = ray.remote(num_cpus=1)(run_reference)
    work = [("native", task) for task in natives] + [("reference", task) for task in refs]
    total = len(work)
    work_iter = iter(work)
    pending, results = {}, []
    def submit(kind, task):
        if kind == "native":
            ref = native_worker.remote(task, args.results_root, args.timeout_seconds, args.commit)
        else:
            ref = ref_worker.remote(task, args.results_root, args.commit)
        pending[ref] = (kind, task)
    for _ in range(min(args.max_in_flight, total)):
        item = next(work_iter, None)
        if item: submit(*item)
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        ref = ready[0]
        kind, task = pending.pop(ref)
        outcome = ray.get(ref)
        results.append(outcome)
        print(f"progress={len(results)}/{total} type={kind} rc={outcome.get('returncode')}", flush=True)
        item = next(work_iter, None)
        if item: submit(*item)
    failed = [x for x in results if x.get("returncode") != 0 or "metrics" not in x]
    if failed: raise SystemExit(f"{len(failed)} figure tasks failed")
    manifest = {
        "schema": "paper_figures_campaign_v1", "git_commit": args.commit,
        "selected_candidate": candidate, "native_task_count": len(natives),
        "reference_task_count": len(refs), "results": results,
    }
    path = Path(args.results_root) / "campaign_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__": main()
