# Assumptions for the Paper Reimplementation

This project reproduces the published equations and stated simulation settings
from `Toward Deterministic 6G HRLLC`. The paper does not publish source code,
so the following implementation choices are necessary. They are recorded here
to keep the results reproducible and to distinguish assumptions from claims
made by the paper.

## Stated by the paper and implemented directly

- Eight moving industrial equipments in a circular area of radius 250 m.
- Two flow types, a 10 ms hyperperiod, 256 steps per trajectory, 1,000
  training episodes, and 10 parameter updates per trajectory.
- Total power 40 W, total HRLLC bandwidth 20 MHz, carrier 3.5 GHz,
  mini-slot duration 0.25 ms, retransmission delay 1 ms, and four TSN hops.
- Large-scale fading follows the paper's equation
  `lambda_k = 10^(PL_k/10) * 10^(SH_k/10)`, with `SH_k ~ N(0, 8^2)`.
- MAPPO uses two 64-unit hidden layers, PPO clip 0.2, gamma 0.9, GAE lambda
  0.9, Huber delta 1, actor learning rate `1e-4`, and critic learning rate
  `5e-3`.
- Performance is aggregated across 100 independently generated wireless IIoT
  scenarios, as stated in the performance-evaluation section.

## Assumptions required because the paper is silent

| Topic | Assumption used here | Why it is needed |
| --- | --- | --- |
| Shadow-fading time correlation | `SH_k` is redrawn every simulation step. | The paper specifies the marginal distribution `N(0, 8^2)` but does not give a decorrelation model. The active config uses independent per-step redraws to follow the interpretation that fading should change at each time step. |
| Initial UE locations | Radius is sampled uniformly from `0` to `r_c`; angle is uniform. | The paper says only “randomly distributed in a circular area” and does not specify an area-uniform, radial-uniform, or fixed topology distribution. This preserves the existing project convention and is recorded so it can be changed later. |
| DRL optimizer | Adam with beta1 `0.9`, beta2 `0.999`, and epsilon `1e-8`; its state is saved with every checkpoint. | The paper gives learning rates but does not name the optimizer. Adam is the standard stable optimizer for PPO-style neural-network training; saving its moments makes a resumed campaign equivalent to a continuous one. |
| Initial model | Xavier-uniform weights and zero output biases. | The paper lists “initial network parameters” but does not define their initialization. Xavier initialization is appropriate for the project's tanh layers and zero biases avoid preferring a power or retransmission action before training. |
| Evaluation action choice | Use argmax actions from each learned policy on fixed scenario seeds. | The paper defines action selection as sampling from `pi_theta_k`; it does not state a final-evaluation override. Argmax removes evaluation-time action randomness and is used here as a reliability-oriented choice. |
| Retransmission boundary | Use the action set in equation (25): `N_k` from `0` through `floor(L_k / d_r)`. | Constraint (20d) is strict, but equation (25) is the later, direct MAPPO action-space definition and includes the boundary. The actual frame deadline is still checked independently through `L_W >= 0`. |
| Retry feasibility mask | Mask retry counts that cannot fit one mini-slot plus the largest possible shared TSN egress service time within `L_ND`. | The paper gives the `L_W >= 0` feasibility condition but does not say whether actions known to violate it should be sampled. The mask is derived from equations (5), (7), and (8), reduces a reward-scale conflict, and remains conservative: it removes only actions that cannot meet latency even under the best possible air-interface duration. |
| Massive-MIMO antenna count | Use the paper's final Gamma-distribution reliability expression directly. | The paper introduces `N` antennas but does not give a numeric value for `N`; its final reliability expression is independent of a chosen antenna count. Adding an array-gain constant would be an unsupported assumption. |

## What this does not claim

This is an equation-level ns-3 execution of the paper model. It is not a
packet-level 5G-LENA or TSN queue-disc reproduction, and it should not be
described as the authors' unpublished source code.

## Active reliability-penalty extension

The active configuration keeps the paper-stated model and optimization
settings of two 64-unit hidden layers, 10 PPO updates per trajectory, and a
critic learning rate of `5e-3`. It uses the paper-unspecified Xavier-uniform
initialization with zero output biases, plus a reliability penalty multiplier
of `2.0`. The multiplier is an extension because it changes the paper-mapped
reward scale. The trainer's mini-batch implementation uses the full 256-step
trajectory as one batch. Validation uses 20 fixed scenarios for checkpoint
selection; the final result remains a separate 100-scenario evaluation.

## Acceptance rule for this project

The trainer selects checkpoints using fixed held-out validation scenarios and
reports a separate final result across 100 unseen scenarios. A near-paper
result means a final failure rate below `0.10%`, with latency and reliability
violation rates reported separately.
