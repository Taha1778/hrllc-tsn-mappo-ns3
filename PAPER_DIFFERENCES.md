# Python-Compatibility Differences From The Paper

This repository has a Python-compatibility baseline. Its target is the supplied
`python paper/code1.py` behavior while retaining C++ ns-3 as the environment.
It is not a strict reproduction of the published paper.

| Area | Classification | Paper | Compatibility baseline | Reason |
| --- | --- | --- | --- |
| Payload service units | Direct deviation | Equations (4), (5), (8), and (9) use `rho / R`. | Uses the supplied Python implementation's numeric `data_size / rate` behavior. | Match the reference source result. |
| E2E latency | Paper-consistent interpretation | Equation (10) uses the DS-TT gate departure time. | Uses that fixed gate departure time; negative `L_W` identifies late air delivery. | No contradiction claimed. |
| Air-interface HARQ time | Direct deviation | Equation (8) reserves retransmission time. | Simulates the supplied source's sequential attempts and stops after first decoding success. | Match the reference source result. |
| Small-scale fading RNG | Paper-unspecified assumption | The paper derives a distribution but does not prescribe RNG implementation. | Python-generated reference Gamma gains are passed to C++ for the same seeded source behavior. | Cross-language deterministic equivalence. |
| Mobility boundary | Paper-unspecified assumption | Random Direction mobility is stated without boundary detail. | Uses the source's 0.99-radius reset, fresh direction, and fresh speed. | Source assumption. |
| Critic target | Direct deviation | Equations (44)-(45) define Huber loss on TD error. | Uses the supplied source's GAE-return target. | Match the reference source result. |
| Neural details | Paper-unspecified assumption | Hidden-layer count and dimension are stated; activation, initialization, optimizer details, and clipping are not. | ReLU, orthogonal initialization, Adam, and 0.5 gradient clipping. | Source assumptions. |
| Antenna count | Paper-consistent | The normalized precoder in Theorem 1 yields `G(n,1)`. | `N=64` remains a scenario label; it is not an additional array-gain multiplier. | No contradiction. |

The automation may change only documented C++/Python compatibility mismatches.
It may not alter paper parameters, reward coefficients, or the 1% acceptance
threshold merely to improve a result.

## Run 3 Experiment Boundary

Run 3 executes the full experiment suite supplied by the user in Python. It is
kept separate from Run 2 because it does not use the C++ ns-3 environment.
Its results are source-reference results, not evidence that every paper baseline
has been reproduced in ns-3.
