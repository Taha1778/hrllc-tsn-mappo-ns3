# Automated Run Change Log

## Run 1: Current PyTorch ns-3 baseline

- Branch: `run/01-current-pytorch-ns3`
- Parent: `main`
- Change set: PyTorch MAPPO learner, ReLU C++ inference, orthogonal initialization, four Type-1 then four Type-2 agents, and unrestricted paper action range.
- Validation result: 15.89% total failure at training round 999 across 20 fixed scenarios.
- Held-out result: 14.31% total failure across 100 scenarios; 12.15% latency violations and 4.37% reliability violations.
- Decision: below neither the 1% compatibility goal nor the paper-level target. Run 2 will address documented Python/C++ environment mismatches only.

## Run Template

- Branch:
- Parent branch:
- Researched compatibility mismatch:
- Implemented change:
- Validation result:
- Held-out result:
- Decision:

## Run 2: Python-Compatible C++ ns-3 Baseline

- Branch: `run/02-python-compatibility-baseline`
- Parent branch: `run/01-current-pytorch-ns3`
- Researched compatibility mismatch: C++ previously used bit-scaled payload service, one random exponential fading draw per frame, pre-reserved retransmission time, common retry-head width, and reconstructed PPO behavior log-probabilities.
- Implemented change: C++ now follows supplied Python source data-unit timing, source HARQ attempt semantics, Python-provided Gamma gains, source boundary reset, and writes action-time log-probabilities. Python uses source-sized retry heads and consumes the recorded PPO behavior values.
- Config: unchanged paper-fixed values; `python_compatibility_mode=true`, target failure rate `0.01`, and three permitted changed runs after Run 1.
- Validation result: preflight passed JSON validation, Python compile/import, ns-3 build, one-step smoke trajectory, and C++/PyTorch action log-probability equivalence.
- Held-out result: pending full training and isolated 100-scenario evaluation.
- Decision: start the Python-compatible baseline campaign.

### 2026-07-28 held-out completion

- Result: passed the `0.01` total-failure target with `0.005853794642857144` (0.5854%) over 100 held-out scenarios.
- Violation rates: latency `0.003515625` (0.3516%); reliability `0.0033970424107142862` (0.3397%).
- Decision: retain the Python-compatible baseline; no additional mismatch correction or campaign is required.
