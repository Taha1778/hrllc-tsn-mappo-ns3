# HRLLC-TSN MAPPO project instructions

- Preserve the paper-fixed values in `configs/default_config.json` unless the
  task explicitly creates a separately labelled extension experiment.
- Before a campaign, use `scripts/start_reproduction_campaign.ps1 -Mode Preflight`.
- For a full campaign, use the background launcher. Do not keep an interactive
  Codex session polling a long run; inspect `runs/workflow_status.json`, logs,
  and final CSV/JSON artifacts after it ends.
- Windows Python orchestrates training and reports. WSL/Linux runs ns-3. The
  shared project folder transfers artifacts automatically; no manual copying.
- Search project source before `external/`; never broad-scan generated ns-3 or
  CMake content unless the task specifically concerns the dependency.
