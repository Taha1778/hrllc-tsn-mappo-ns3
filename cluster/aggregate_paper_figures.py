#!/usr/bin/env python3
"""Validate a complete campaign and render provenance-labelled Figures 7--13."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

KS=(2,4,6,8,10,12); RADII=(100,150,200,250,300,350); POWERS=(20,30,40,50,60); GAMMAS=(3,4,5,6,7)

def mean(rows, key): return sum(float(r["metrics"][key]) for r in rows)/len(rows)
def reference_task(row):
    """Return reference-task metadata from both supported result schemas."""
    return row.get("task", row)
def save(fig, out, stem):
    fig.tight_layout(); fig.savefig(out/f"{stem}.png", dpi=180); fig.savefig(out/f"{stem}.pdf"); plt.close(fig)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--results-root", required=True); a=p.parse_args()
    root=Path(a.results_root); data=json.loads((root/"campaign_manifest.json").read_text())
    rows=data["results"]; commits={r["git_commit"] for r in rows}
    if commits!={data["git_commit"]}: raise SystemExit(f"Mixed commits: {commits}")
    if len(rows)!=data["native_task_count"]+data["reference_task_count"]: raise SystemExit("Incomplete manifest")
    out=root/"figures"; out.mkdir(exist_ok=True); native=defaultdict(list); ref=defaultdict(list)
    for r in rows:
        if r["source"].startswith("native_ns3"):
            t=r["task"]; native[(t["variant"],t["axis"],str(t["value"]),str(t.get("base_config",{}).get("gamma_threshold_db","")))].append(r)
        else:
            t=reference_task(r)
            ref[(t["method"],t["axis"],str(t["value"]))].append(r)
    # Native lookup tolerates numeric JSON formatting and identifies gamma via saved config when present.
    def nrows(variant, axis, value, gamma=None):
        found=[]
        for r in rows:
            if not r["source"].startswith("native_ns3"): continue
            t=r["task"]
            if t["variant"]==variant and t["axis"]==axis and float(t["value"])==float(value):
                if gamma is None or float(r.get("configured_gamma_db", gamma))==float(gamma): found.append(r)
        if not found: raise SystemExit(f"Missing native point {variant} {axis} {value} gamma={gamma}")
        return found
    tables={}
    # Fig 7: true native training trajectories, averaged over seeds and binned every 50 rounds.
    fig,ax=plt.subplots(figsize=(10,6)); table=[]
    for v in ("MAPPO-H","MAPPO-M"):
        for k in (4,8,12):
            bins=defaultdict(list)
            for r in nrows(v,"k",k):
                summaries=list(Path(r["job_dir"]).joinpath("runs").glob("summary*.csv"))
                if not summaries: raise SystemExit(f"Missing convergence summary: {r['job_dir']}")
                with summaries[0].open(encoding="utf-8") as f:
                    for row in csv.DictReader(f): bins[(int(row["round"])//50)*50].append(float(row["reward_mean"]))
            xs=sorted(bins); ys=[sum(bins[x])/len(bins[x]) for x in xs]
            ax.plot(xs,ys,label=f"{v} K={k} (ns-3)"); table += [[v,k,x,y] for x,y in zip(xs,ys)]
    ax.set(xlabel="Training round",ylabel="Mean reward",title="Figure 7: Native MAPPO training convergence"); ax.grid(alpha=.3); ax.legend(ncol=2); save(fig,out,"figure7_training_convergence"); tables[7]=(["method","k","training_round_bin","reward_mean"],table)
    # Fig 8
    fig,ax=plt.subplots(figsize=(10,6)); table=[]
    for method in ("MAPPO-H","MAPPO-M"):
        ys=[mean(nrows(method,"k",k),"failure_rate") for k in KS]; ax.plot(KS,ys,"o-",label=f"{method} (ns-3)"); table += [[method,"native_ns3",k,y] for k,y in zip(KS,ys)]
    for method in ("CPPO","DQN","MADDPG","BCD","Random"):
        ys=[mean(ref[(method,"k",str(k))],"failure_rate") for k in KS]; ax.plot(KS,ys,"o--",label=f"{method} (Python reference)"); table += [[method,"python_reference",k,y] for k,y in zip(KS,ys)]
    ax.set(xlabel="K",ylabel="Failure rate",title="Figure 8: Failure rate vs K"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.3); ax.legend(ncol=2); save(fig,out,"figure8_failure_rate_vs_k"); tables[8]=(["method","source","k","failure_rate"],table)
    # Fig 9
    fig,axs=plt.subplots(1,2,figsize=(12,5)); table=[]
    for metric,title,axis in (("latency_violation_rate","Latency",axs[0]),("reliability_violation_rate","Reliability",axs[1])):
        for method in ("MAPPO-H","MAPPO-M"):
            ys=[mean(nrows(method,"k",k),metric) for k in KS]; axis.plot(KS,ys,"o-",label=f"{method} (ns-3)"); table += [[method,"native_ns3",metric,k,y] for k,y in zip(KS,ys)]
        ys=[mean(ref[("CPPO","k",str(k))],metric) for k in KS]; axis.plot(KS,ys,"o--",label="CPPO (Python reference)"); table += [["CPPO","python_reference",metric,k,y] for k,y in zip(KS,ys)]
        axis.set(xlabel="K",ylabel=f"{title} violation rate",title=title); axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.3); axis.legend()
    save(fig,out,"figure9_qos_violations_vs_k"); tables[9]=(["method","source","metric","k","value"],table)
    # Fig 10
    fig,ax=plt.subplots(figsize=(10,6)); table=[]
    for method in ("MAPPO-H","MAPPO-M"):
        ys=[mean(nrows(method,"radius",r),"failure_rate") for r in RADII]; ax.plot(RADII,ys,"o-",label=f"{method} (ns-3)"); table += [[method,"native_ns3",r,y] for r,y in zip(RADII,ys)]
    for method in ("CPPO","DQN","MADDPG","BCD","Random"):
        ys=[mean(ref[(method,"radius",str(r))],"failure_rate") for r in RADII]; ax.plot(RADII,ys,"o--",label=f"{method} (Python reference)"); table += [[method,"python_reference",r,y] for r,y in zip(RADII,ys)]
    ax.set(xlabel="Radius (m)",ylabel="Failure rate",title="Figure 10: Failure rate vs radius"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.3); ax.legend(ncol=2); save(fig,out,"figure10_failure_rate_vs_radius"); tables[10]=(["method","source","radius_m","failure_rate"],table)
    # Figure 11 uses independently trained native power/SNR points. Gamma is encoded in point_key.
    fig,ax=plt.subplots(figsize=(10,6)); table=[]
    for gamma in GAMMAS:
        ys=[]
        for power in POWERS:
            rs=[r for r in rows if r["source"].startswith("native_ns3") and r.get("point_key","").startswith(f"MAPPO-H|power-gamma|{power}:{gamma}|")]
            if not rs: raise SystemExit(f"Missing power/SNR point {power}/{gamma}")
            ys.append(mean(rs,"failure_rate"))
        ax.plot(POWERS,ys,"o-",label=f"SNR threshold {gamma} dB"); table += [[power,gamma,y] for power,y in zip(POWERS,ys)]
    ax.set(xlabel="Total power (W)",ylabel="Failure rate",title="Figure 11: MAPPO-H (ns-3) power/SNR sensitivity"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.3); ax.legend(); save(fig,out,"figure11_power_snr_sensitivity"); tables[11]=(["power_w","gamma_db","failure_rate"],table)
    # Fig 12
    fig,ax=plt.subplots(figsize=(9,6)); table=[]
    for method in ("MAPPO-H","MAPPO-R","MAPPO-S"):
        ys=[]
        for power in POWERS:
            if method=="MAPPO-H":
                rs=[r for r in rows if r["source"].startswith("native_ns3") and r.get("point_key","").startswith(f"MAPPO-H|power-gamma|{power}:5|")]
            else: rs=nrows(method,"power",power)
            ys.append(mean(rs,"failure_rate"))
        ax.plot(POWERS,ys,"o-",label=f"{method} (ns-3)"); table += [[method,p,y] for p,y in zip(POWERS,ys)]
    ax.set(xlabel="Total power (W)",ylabel="Failure rate",title="Figure 12: Native MAPPO variants"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.3); ax.legend(); save(fig,out,"figure12_mappo_variants_vs_power"); tables[12]=(["method","power_w","failure_rate"],table)
    # Fig 13 reports measured campaign compute time normalized per simulated decision.
    fig,ax=plt.subplots(figsize=(9,6)); table=[]
    m=[]
    for k in KS:
        rs=nrows("MAPPO-H","k",k)
        m.append(sum(r["training_elapsed_seconds"]*1000/(r["task"]["rounds"]*r["task"]["iterations"]) for r in rs)/len(rs))
    b=[sum(r["elapsed_seconds"]*1000/r["scenario_count"] for r in ref[("BCD","k",str(k))])/len(ref[("BCD","k",str(k))]) for k in KS]
    ax.plot(KS,m,"o-",label="MAPPO-H native campaign time/decision"); ax.plot(KS,b,"o--",label="BCD Python-reference time/scenario")
    table=[[k,x,y] for k,x,y in zip(KS,m,b)]; ax.set(xlabel="K",ylabel="Milliseconds",title="Figure 13: Computation-time comparison (normalized units differ)"); ax.grid(alpha=.3); ax.legend(); save(fig,out,"figure13_decision_time_vs_k"); tables[13]=(["k","mappo_native_campaign_ms_per_decision","bcd_python_ms_per_scenario"],table)
    for number,(header,body) in tables.items():
        with (out/f"figure{number}.csv").open("w",newline="",encoding="utf-8") as f: w=csv.writer(f); w.writerow(header); w.writerows(body)
    report={"git_commit":data["git_commit"],"selected_candidate":data["selected_candidate"],"figures":list(range(7,14)),"provenance":{"native":"C++ ns-3 with Python MAPPO","reference":"Python reference baselines; legends label these explicitly"}}
    (out/"combined_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"PROVENANCE.md").write_text(f"# Figures 7-13 provenance\n\nCommit: `{data['git_commit']}`\n\nMAPPO series are native C++ ns-3 measurements. CPPO, DQN, MADDPG, BCD, and Random are labelled Python-reference baselines. Figure 13 normalizes MAPPO campaign wall time per simulated decision and BCD Python time per evaluated scenario; these are not identical timing units and the figure states that limitation.\n")
    print(out)
if __name__=="__main__": main()
