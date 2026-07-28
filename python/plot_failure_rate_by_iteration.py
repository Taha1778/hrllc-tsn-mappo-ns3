"""Aggregate the held-out ns-3 CSVs into one failure-rate point per iteration."""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_CSV = RUNS_DIR / "failure_rate_by_iteration.csv"
OUTPUT_SVG = RUNS_DIR / "failure_rate_by_iteration.svg"


def main():
    files = sorted(RUNS_DIR.glob("final_test_round_*.csv"))
    if not files:
        raise FileNotFoundError("No final-test CSV files were found in runs/.")

    frames = []
    for path in files:
        frames.append(pd.read_csv(path, usecols=["step", "frames", "successes"]))
    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby("step", as_index=False)[["frames", "successes"]].sum()
    grouped["failure_rate"] = 1.0 - grouped["successes"] / grouped["frames"]
    grouped["failure_rate_percent"] = 100.0 * grouped["failure_rate"]
    grouped.to_csv(OUTPUT_CSV, index=False)

    mean_percent = grouped["failure_rate_percent"].mean()
    width, height = 1200, 640
    left, right, top, bottom = 95, 50, 75, 100
    plot_width, plot_height = width - left - right, height - top - bottom
    y_max = max(100.0, float(grouped["failure_rate_percent"].max()) * 1.05)

    def x(value):
        return left + (value / max(1, len(grouped) - 1)) * plot_width

    def y(value):
        return top + (1.0 - value / y_max) * plot_height

    points = " ".join(
        f"{x(float(step)):.2f},{y(float(rate)):.2f}"
        for step, rate in zip(grouped["step"], grouped["failure_rate_percent"])
    )
    grid = []
    for tick in range(0, 101, 20):
        grid.append(
            f'<line x1="{left}" y1="{y(tick):.2f}" x2="{width-right}" y2="{y(tick):.2f}" class="grid"/>'
        )
        grid.append(f'<text x="{left-12}" y="{y(tick)+5:.2f}" text-anchor="end" class="axis">{tick}%</text>')
    x_ticks = []
    for tick in range(0, len(grouped), 32):
        x_ticks.append(f'<text x="{x(tick):.2f}" y="{height-bottom+30}" text-anchor="middle" class="axis">{tick + 1}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Failure rate for every simulation iteration">
<style>
text {{ font-family: Arial, sans-serif; fill: #1f2937; }}
.title {{ font-size: 22px; font-weight: 600; }} .axis {{ font-size: 14px; }}
.grid {{ stroke: #d1d5db; stroke-width: 1; }} .series {{ fill: none; stroke: #2563eb; stroke-width: 2; }}
.mean {{ stroke: #dc2626; stroke-width: 2; }} .paper {{ stroke: #16a34a; stroke-width: 2; stroke-dasharray: 7 5; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="38" text-anchor="middle" class="title">Failure Rate at Every Iteration Across 100 Held-Out Scenarios</text>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#374151"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#374151"/>
<polyline points="{points}" class="series"/>
<line x1="{left}" y1="{y(mean_percent):.2f}" x2="{width-right}" y2="{y(mean_percent):.2f}" class="mean"/>
<line x1="{left}" y1="{y(0.04):.2f}" x2="{width-right}" y2="{y(0.04):.2f}" class="paper"/>
<text x="{width-right-4}" y="{y(mean_percent)-8:.2f}" text-anchor="end" class="axis">Our mean: {mean_percent:.2f}%</text>
<text x="{width-right-4}" y="{y(0.04)-8:.2f}" text-anchor="end" class="axis">Paper MAPPO-H: about 0.04%</text>
<text x="{width/2}" y="{height-25}" text-anchor="middle" class="axis">Simulation iteration (10 ms each)</text>
<text x="24" y="{height/2}" text-anchor="middle" class="axis" transform="rotate(-90 24 {height/2})">Failure rate (%)</text>
{''.join(x_ticks)}
</svg>'''
    OUTPUT_SVG.write_text(svg, encoding="utf-8")
    print(f"csv={OUTPUT_CSV}")
    print(f"svg={OUTPUT_SVG}")


if __name__ == "__main__":
    main()
