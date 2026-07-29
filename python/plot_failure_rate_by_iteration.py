"""Create paper-style result figures from completed HRLLC-TSN MAPPO runs.

The current campaign contains one evaluated configuration: MAPPO-H, K=8,
radius=250 m, power=40 W, and required SNR=5 dB. The script therefore draws
the matching measured point in Figures 8-13 and never invents missing methods
or sweep points. Add optional runs/paper_sweep_results.csv to extend a figure
with additional completed experiments.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
FIGURES_DIR = RUNS_DIR / "paper_figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.json"
SWEEP_RESULTS_PATH = RUNS_DIR / "paper_sweep_results.csv"


def newest_summary_path():
    summaries = list(RUNS_DIR.glob("summary*.csv"))
    if not summaries:
        raise FileNotFoundError("No training summary CSV was found in runs/.")
    return max(summaries, key=lambda path: path.stat().st_mtime)


def save_figure(figure, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES_DIR / f"{filename}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def optional_sweep_results():
    if not SWEEP_RESULTS_PATH.exists():
        return pd.DataFrame()
    required = {"figure", "series", "x", "failure_rate"}
    data = pd.read_csv(SWEEP_RESULTS_PATH)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{SWEEP_RESULTS_PATH.name} is missing columns: {sorted(missing)}")
    return data


def plot_convergence(summary):
    data = summary.sort_values("round").copy()
    data["moving_average_return"] = data["reward_mean"].rolling(9, min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(data["round"] + 1, data["moving_average_return"], color="#1f77b4", linewidth=1.8,
              label="MAPPO-H, K=8")
    axis.set(title="Fig. 7-style: Moving Average Return Versus Episode", xlabel="Episode", ylabel="Moving average return (window = 9)")
    axis.legend(frameon=True)
    save_figure(figure, "fig07_moving_average_return_vs_episode")


def plot_failure_by_axis(figure_number, title, x_label, x_value, failure_rate, filename, sweeps):
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    matches = sweeps[sweeps["figure"] == figure_number] if not sweeps.empty else pd.DataFrame()
    if not matches.empty:
        for series, subset in matches.groupby("series"):
            subset = subset.sort_values("x")
            axis.plot(subset["x"], 100 * subset["failure_rate"], marker="o", linewidth=1.6, label=series)
    axis.plot([x_value], [100 * failure_rate], marker="o", markersize=7, color="#1f77b4",
              linestyle="None", label="MAPPO-H (this run)")
    axis.set(title=title, xlabel=x_label, ylabel="Failure rate (%)")
    axis.legend(frameon=True)
    save_figure(figure, filename)


def plot_violation_rates(config, final_metrics, sweeps):
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    matches = sweeps[sweeps["figure"] == 9] if not sweeps.empty else pd.DataFrame()
    if not matches.empty and {"latency_violation_rate", "reliability_violation_rate"}.issubset(matches.columns):
        for series, subset in matches.groupby("series"):
            subset = subset.sort_values("x")
            axis.plot(subset["x"], 100 * subset["latency_violation_rate"], marker="o", label=f"{series}-L")
            axis.plot(subset["x"], 100 * subset["reliability_violation_rate"], marker="s", linestyle="--", label=f"{series}-R")
    k = int(config["k"])
    axis.plot([k], [100 * final_metrics["latency_violation_rate"]], marker="o", markersize=7,
              color="#1f77b4", linestyle="None", label="MAPPO-H-L (this run)")
    axis.plot([k], [100 * final_metrics["reliability_violation_rate"]], marker="s", markersize=7,
              color="#d62728", linestyle="None", label="MAPPO-H-R (this run)")
    axis.set(title="Fig. 9-style: QoS Violation Rate Versus K", xlabel="Number of industrial equipments K",
             ylabel="Violation rate (%)")
    axis.legend(frameon=True, ncols=2)
    save_figure(figure, "fig09_latency_and_reliability_violation_vs_k")


def measure_policy_inference_ms(config):
    checkpoint = WEIGHTS_DIR / "best_model.npz"
    if not checkpoint.exists():
        return None
    sys_path = str(PROJECT_ROOT / "python")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import hrllc_tsn_trainer as trainer

    model = trainer.load_model_npz(checkpoint)
    observations = [torch.zeros((1, model["obs_dim"]), dtype=torch.float32) for _ in model["actors"]]
    for actor, observation, valid_count in zip(model["actors"], observations, model["actor_n_action_dims"]):
        actor(observation, valid_count)
    repetitions = 2000
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(repetitions):
            for actor, observation, valid_count in zip(model["actors"], observations, model["actor_n_action_dims"]):
                actor(observation, valid_count)
    return 1000.0 * (time.perf_counter() - started) / repetitions


def plot_computation_time(config, sweeps):
    inference_ms = measure_policy_inference_ms(config)
    if inference_ms is None:
        return
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    matches = sweeps[sweeps["figure"] == 13] if not sweeps.empty else pd.DataFrame()
    if not matches.empty and "computation_time_ms" in matches.columns:
        for series, subset in matches.groupby("series"):
            subset = subset.sort_values("x")
            axis.plot(subset["x"], subset["computation_time_ms"], marker="o", linewidth=1.6, label=series)
    axis.plot([int(config["k"])], [inference_ms], marker="o", markersize=7, color="#1f77b4",
              linestyle="None", label="MAPPO-H actor inference (this CPU)")
    axis.set(title="Fig. 13-style: Computation Time Versus K", xlabel="Number of industrial equipments K",
             ylabel="Policy inference time per decision (ms)")
    axis.legend(frameon=True)
    save_figure(figure, "fig13_computation_time_vs_k")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style plots from completed results")
    parser.add_argument("--summary", type=Path, default=None, help="Training summary CSV; defaults to newest summary*.csv")
    args = parser.parse_args()

    apply_style()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summary = pd.read_csv(args.summary or newest_summary_path())
    final_metrics = json.loads((RUNS_DIR / "final_test_metrics.json").read_text(encoding="utf-8"))
    sweeps = optional_sweep_results()

    plot_convergence(summary)
    plot_failure_by_axis(8, "Fig. 8-style: Failure Rate Versus K", "Number of industrial equipments K",
                         int(config["k"]), final_metrics["failure_rate"], "fig08_failure_rate_vs_k", sweeps)
    plot_violation_rates(config, final_metrics, sweeps)
    plot_failure_by_axis(10, "Fig. 10-style: Failure Rate Versus Area Radius", "Circular-area radius (m)",
                         float(config["radius_m"]), final_metrics["failure_rate"], "fig10_failure_rate_vs_radius", sweeps)
    plot_failure_by_axis(11, "Fig. 11-style: Failure Rate Versus HRLLC Power", "Total HRLLC power (W)",
                         float(config["total_power_w"]), final_metrics["failure_rate"], "fig11_failure_rate_vs_power", sweeps)
    plot_failure_by_axis(12, "Fig. 12-style: MAPPO Failure Rate Versus HRLLC Power", "Total HRLLC power (W)",
                         float(config["total_power_w"]), final_metrics["failure_rate"], "fig12_mappo_failure_rate_vs_power", sweeps)
    plot_computation_time(config, sweeps)
    print(f"figures={FIGURES_DIR}")


if __name__ == "__main__":
    main()
