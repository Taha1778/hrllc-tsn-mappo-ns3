#!/usr/bin/env python3
"""Validate and combine isolated Figure 8 points after all Ray jobs finish."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

METHODS = ("MAPPO-H", "MAPPO-M", "CPPO", "DQN", "MADDPG", "BCD", "Random")
K_VALUES = (2, 4, 6, 8, 10, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed Figure 8 Ray jobs")
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    root = Path(args.results_root).resolve()

    points = {}
    for method in METHODS:
        for k in K_VALUES:
            path = root / method.lower() / f"k-{k:02d}" / "figure8_point.json"
            if not path.exists():
                raise SystemExit(f"Missing required result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("provenance") != "source_python_reference" or payload.get("method") != method or payload.get("k") != k:
                raise SystemExit(f"Invalid or mismatched result: {path}")
            points[(method, k)] = payload

    commits = {payload["git_commit"] for payload in points.values()}
    if len(commits) != 1:
        raise SystemExit(f"Results come from different Git commits: {sorted(commits)}")

    output = {
        "provenance": "source_python_reference",
        "figure": 8,
        "git_commit": commits.pop(),
        "k_values": list(K_VALUES),
        "results": {
            method: [points[(method, k)]["metrics"]["failure_rate"] for k in K_VALUES]
            for method in METHODS
        },
    }
    (root / "figure8_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    with (root / "figure8_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", *METHODS])
        for index, k in enumerate(K_VALUES):
            writer.writerow([k, *[output["results"][method][index] for method in METHODS]])

    figure, axis = plt.subplots(figsize=(12, 7))
    for method in METHODS:
        axis.plot(K_VALUES, output["results"][method], marker="o", label=method)
    axis.set(xlabel="Number of Industrial Equipments K", ylabel="Failure Rate", title="Figure 8: Failure Rate vs K")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(root / "figure8_failure_rate_vs_k.png", dpi=150)
    print(root / "figure8_results.json")


if __name__ == "__main__":
    main()
