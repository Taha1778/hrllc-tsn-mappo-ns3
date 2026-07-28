# Python-Compatibility Differences From The Paper

This repository has a Python-compatibility baseline. Its target is the supplied
`python paper/code1.py` behavior while retaining C++ ns-3 as the environment.
It is not a strict reproduction of the published paper.

| Area | Paper | Compatibility baseline | Reason |
| --- | --- | --- | --- |
| Payload service units | Equations (4), (5), (8), and (9) use `rho / R`. | Uses the supplied Python implementation's numeric `data_size / rate` behavior. | Match the reference source result. |
| E2E latency | Equation (10) uses the DS-TT gate departure time. | Sets E2E latency from the fixed gate departure time, as the supplied source does; late air-interface completion is identified by negative `L_W`. | Match the reference source result. |
| Air-interface HARQ time | Equation (8) reserves retransmission time. | Simulates the supplied source's sequential HARQ attempts and stops after first successful decoding. | Match the reference source result. |
| Small-scale fading | The paper derives a distribution but does not prescribe RNG implementation. | Python-generated reference Gamma gains are passed to C++ for the same seeded source behavior. | Cross-language deterministic equivalence. |
| Mobility boundary | Random Direction model is stated without boundary detail. | Uses the supplied source's 0.99-radius reset, fresh direction, and fresh speed. | Source assumption. |
| Critic target | Equations (44)-(45) define Huber loss on TD error. | Uses the supplied source's GAE-return target. | Source differs from paper. |
| Neural details | Hidden-layer count and dimension are stated; activation, initialization, optimizer details, and clipping are not. | ReLU, orthogonal initialization, Adam, and 0.5 gradient clipping. | Source assumptions. |
| Antenna count | The normalized precoder in Theorem 1 yields `G(n,1)`. | `N=64` remains a scenario label; it is not an additional array-gain multiplier. | Matches both source and derived expression. |

The automation may change only documented C++/Python compatibility mismatches.
It may not alter paper parameters, reward coefficients, or the 1% acceptance
threshold merely to improve a result.
