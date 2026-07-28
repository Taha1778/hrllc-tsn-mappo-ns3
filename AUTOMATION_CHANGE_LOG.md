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
