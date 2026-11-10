# Week 5 results: encoder and decoder, open loop

Setup: male-CNS v1.0 full network, Numba engine, w_scale 0.7, laptop (WSL, 4 GB), run 2026-11-09/10, 36 min in total.
Replay: 60 successful validation episodes from the week 4 dataset, 3,877 control steps, 167 s of brain time. Each episode starts from a fresh brain, is held 200 ms at its first state, then gets one logged state per 40 ms step.
Encoder: 4 signals (cube − gripper x, y, z; gripper width) → 335 JO-C/E neurons. Each signal drives a push channel at 150 + 125·s Hz and a pull channel at 150 − 125·s Hz, so the total input rate is the same in every state. Positions use tanh squashing (5 cm in x, y; 8 cm in z).
Decoder: rates over the last 20, 40 or 50 ms. Primary readout: the 68 week 1 DN candidates.
All R² values are out-of-episode (5 folds split by whole episode, ridge penalty chosen inside each training fold).

## Verdict: PASS (rule fixed before the run)
Rule: every signal beats all 20 episode-swap nulls, and the median R² across the 4 signals is at least 0.5.

| readout, 40 ms | x | y | z | gripper width | median |
|---|---|---|---|---|---|
| **DN candidates (68)** | **0.64** | **0.84** | **0.88** | **0.85** | **0.84** |
| all DNs | 0.76 | 0.89 | 0.95 | 0.90 | 0.89 |
| front-leg MNs (81) | −0.01 | 0.14 | 0.04 | 0.03 | 0.03 |
| JO input spikes (ceiling) | 0.95 | 0.97 | 0.98 | 0.98 | 0.97 |
| null, max of 20 (DN candidates) | 0.04 | 0.08 | 0.52 | 0.45 | |

- The null is high for z and gripper width. Every episode grasps at about the same time, so time-in-episode alone predicts those two signals. DN candidates still clear the null by 0.36 for z and 0.40 for gripper width.
- Decoded cube position, when the cube is within 5 cm of the gripper in x and y: median error 4.3 mm (x), 3.1 mm (y), 9.9 mm (z). For comparison, the expert's tolerances are 12 mm to start descending and 15 mm to close.
- Leg motor neurons carry nothing. This confirms the week 2 decision to read out from descending neurons.

## Encoder choice: wiring (rule fixed in advance; decided on tuning sweeps, scored on replays)
| encoder | output overlap between channels | sweep R², DN candidates | sweep R², all DNs |
|---|---|---|---|
| **wiring** (JO neurons grouped by whom they synapse onto) | 0.16 | **0.76** | 0.80 |
| random split | 0.82 | 0.26 | 0.38 |

The grouping matters because of the constant-total-drive code. Push and pull channels whose neurons reach the same partners (as with the random split) largely cancel, so the brain receives almost nothing. Wiring-based grouping puts each signal's two halves onto different partners.

## Timing
- Window: 20 / 40 / 50 ms gives median R² 0.73 / 0.84 / 0.86. Keep 40 ms, so one decoder window equals one control step.
- Step response: the DN-candidate readout reaches 90% of a step change 25 ms after the input changes. All DNs take 35 ms.
- Lag: R² peaks at lag 0 for every readout. One 40 ms brain advance per control step is enough, with no extra delay.
- Beyond 3 steps of lag, the brain readouts score above the input ceiling. So the brain holds some memory of earlier states beyond 40 ms. It is small.

## Shuffled-wiring control (same input, same activity)
Degree-preserving shuffle of the full network, w_scale adjusted to match activity. The shuffled network was quieter than the real one, so its scale went up to 1.14, the opposite of the week 2 subnetworks. JO input spikes were 99.98% identical in the two replays, because the stimulus random numbers don't depend on the wiring. So the comparison holds the input fixed.

| | real | shuffled |
|---|---|---|
| non-input neurons firing ≥ 1 Hz | 4.8% | 4.5% |
| DN candidates, mean rate | 40.7 Hz | 1.2 Hz |
| median R², DN candidates | **0.84** | **0.28** |
| R² of all recorded outputs, x / y / z / width* | 0.79 / 0.89 / 0.96 / 0.90 | 0.65 / 0.76 / 0.89 / 0.88 |
| same rule for both: the 68 most active outputs, x / y / z / width* | 0.63 / 0.76 / 0.85 / 0.82 (median 0.79) | 0.41 / 0.60 / 0.86 / 0.85 (median 0.73) |

\* Checked afterwards from the replay files. The targets are signals reconstructed from the JO input spikes (push minus pull), because the replay files don't store the states. These targets reproduce the notebook's real-network R² to within 0.03. "Most active" was chosen on half the episodes, from rates only.

What this shows:
- **The connectome decides where the task information lands.** Real wiring routes it onto the week 1 DN candidates. Random wiring leaves those 68 neurons nearly silent.
- **Some information reaching the DN population does not need the connectome.** Across all of the shuffled network's descending neurons, the state is still decodable well above the null (x 0.65, y 0.76; null at most 0.08).
- **Selecting by connectivity beats selecting by activity.** The week 1 DN candidates (0.84) beat the real network's own 68 most active outputs (0.79).
- **With the same selection rule, real still beats shuffled, mainly in x and y** (0.63 vs 0.41, 0.76 vs 0.60). Height and gripper width are equal.

## Action and phase (week 6 and 7 preview)
Expert action R² from a linear readout, all phases:

| | dx | dy | dz | grip |
|---|---|---|---|---|
| DN candidates | 0.21 | 0.55 | 0.24 | 0.69 |
| all DNs | 0.26 | 0.56 | 0.26 | 0.73 |
| the 4 encoded signals, noise-free | 0.31 | 0.60 | 0.22 | 0.81 |
| raw 24-value state | 0.57 | 0.87 | 0.77 | 0.86 |

The brain readout comes close to the noise-free encoded signals. The weak scores come from **what the encoder includes**, not from the brain.

A follow-up on 120 episodes of the same expert (no brain), with the same linear readout plus a small MLP:
- **Reach + descend** (week 6 scope): the 4 signals predict dx and dy well with a linear readout (0.92, 0.97). dz is the weak axis: 0.42 linear, 0.62 MLP. The expert descends only once it is aligned within 12 mm, and a linear readout can't produce that cutoff.
- **Full task:** 4 signals aren't enough even with an MLP (dx 0.13, dz 0.36). Adding goal − cube x and y, gripper height and cube height (8 signals) raises the MLP to dy 0.89, dz 0.72 and grip 0.98. dx stays low (0.36) in this small test, so check it again on the full dataset.

Phase (8 classes, chance 0.125), balanced accuracy: DN candidates 0.34, all DNs 0.40, noise-free encoded signals 0.32. The 4 signals don't separate the phases. All DNs do slightly better than the noise-free signals, which suggests the brain's nonlinearity helps a little. The effect is small.

## Other numbers
- Neurons active during the replay: 10,952, close to the week 3 active set (10,677). The active-set fallback is still viable. Rebuild it from `neuron_hz` in `replay_real_laptop.npz`.
- Brain runtime is consistent with week 3 (about 4.6 s per simulated second).

## Decisions for weeks 6–7
- **Week 6 readout:** DN candidates, 40 ms window, one 40 ms brain advance per control step, no extra delay. Keep the wiring encoder (`encoder_wiring.json`).
- **Week 6 risk is dz, not x and y.** Report reach success separately for the travel and descend parts. If descent fails, first train the readout on brain rates to target the action (the brain may supply the nonlinearity). If that isn't enough, add gripper height as a signal.
- **Before week 7: expand the encoder to 8 signals** (add goal − cube x, y; gripper height; cube height). Then there are 16 channels of about 21 JO neurons, so check them on tuning sweeps first. The current input ceiling (0.97) leaves room for the extra noise.
- **Week 7 random-network baseline must use the same readout rule as the connectome.** Use the 68 most active (or most informative) outputs, chosen on training episodes. Reading the random network out from the week 1 DN candidates would rig the comparison: those neurons are nearly silent in it.
- **Week 7 expectation:** with a fair readout, the random network decodes the state only somewhat worse (median 0.73 vs 0.79), mainly in lateral position. The connectome's advantage is likely to show up as a gap in reaching accuracy in x and y, not as a gap between working and not working.
