# Run 3 Full Paper Figure Suite

Run 3 executes the complete experiment runner supplied in `python paper/code1.py`.
It produces the source runner's Figures 7 through 13: MAPPO-H/M convergence,
method comparison across K, latency/reliability violations, radius sweep,
power/SNR sweep, MAPPO-H/R/S comparison, and MAPPO/BCD timing.

This is a Python-reference experiment suite. Its environment is implemented in
Python by the supplied code, so its outputs must not be presented as C++ ns-3
results and must not be combined with Run 2 held-out metrics. The source
runner's methods and experiment settings are retained as supplied; generated
files are isolated under `runs/run03_paper_reference_suite/`.

Run with `scripts/start_run03_paper_reference_suite.ps1 -Background`.
