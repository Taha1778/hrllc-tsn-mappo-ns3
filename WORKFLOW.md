# Low-Overhead Paper-Reproduction Workflow

This workflow preserves the implemented paper configuration. It does not tune
the MAPPO learning rates, change the reward, replace MAPPO, or alter the
scenario distribution.

## Run once, without keeping Codex active

From PowerShell in this project folder:

```powershell
.\scripts\start_reproduction_campaign.ps1 -Mode Preflight
.\scripts\start_reproduction_campaign.ps1 -Mode Full -FreshStart -Background
```

The first command checks the fixed paper parameters, Python dependencies, and
the ns-3 build. The second starts the full 1,000-episode campaign in a hidden
PowerShell process and returns immediately. No manual Windows-to-WSL copying is
needed: the project folder is shared by both environments.

## What runs where

| Part | Location | Output |
| --- | --- | --- |
| Campaign orchestration, MAPPO update, checkpoints, reports | Windows Python | `weights/` and `runs/` |
| Discrete-event simulation | WSL/Linux ns-3 | round and validation CSVs in `runs/` |
| Monitoring | Windows | `runs/workflow_status.json` and campaign logs |

## Monitoring after the process has started

- `runs/workflow_status.json` is the authoritative short status: `running`,
  `completed`, `failed`, or `preflight_passed`.
- `runs/summary.csv` gains one row per completed training episode.
- `runs/validation_summary.csv` contains the fixed-seed checkpoint evaluations.
- The launcher prints the two log-file paths. Errors are written there and to
  `workflow_status.json`; the workflow never plays a sound.
- Checkpoints and simulator weights are published atomically and retry for up
  to 60 seconds if OneDrive or an indexer temporarily locks a target file.

## Resume safely

If a campaign stops, first run preflight, then restart without `-FreshStart`:

```powershell
.\scripts\start_reproduction_campaign.ps1 -Mode Full -Background
```

The trainer resumes from `weights/current_model.npz` and the next round number.
Use `-FreshStart` only when deliberately starting a new 1,000-episode campaign.

## Final evaluation

After training has completed, use the existing final evaluator. It keeps the
100 final seeds separate from training and validation:

```powershell
.\scripts\start_reproduction_campaign.ps1 -Mode Final
```

Do not alter `configs/default_config.json` for a paper-reproduction result.
Any MAPPO, reward, scheduling, hyperparameter-search, or curriculum change
belongs in a separately labelled extension experiment.
