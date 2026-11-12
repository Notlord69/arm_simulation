# Week 6 results: reservoir readout and first closed loop

Setup: male-CNS v1.0 full network, frozen, Numba engine, w_scale 0.7, wiring encoder from week 5 (4 signals → 335 JO-C/E). Laptop (WSL, 4 GB), run 2026-11-10, 20 min in total, 254 s of brain time.
Loop, one 40 ms control step: arm state → encoder → 40 ms of brain → DN-candidate rates (68 neurons, 40 ms window) → ridge readout → dx, dy, dz → week 4 IK controller → 40 ms of physics. The gripper stays open (reaching only).
Training: DAgger, 4 rounds × 30 episodes. In round 0 the expert drives; in rounds 1–3 the brain drives. Half the episodes in each round add 5 mm per step of action noise. In total 2,769 labelled steps. Ridge penalty chosen by cross-validation grouped by episode (alpha 100).
Test: 50 fresh seeds (90000–90049), never used in training, the same seeds for every policy.

## Verdict: PASS (rule fixed before the run)
Rule: brain reach success ≥ 70%, and better than the cut-feedback control.

| policy, 50 test seeds | reached | 95% CI | travel | knocked | median time | closest x-y (median) |
|---|---|---|---|---|---|---|
| expert | 50/50 | 93–100% | 100% | 0% | 15 steps (0.60 s) | 0.2 mm |
| linear readout on the 4 signals, no brain | 50/50 | 93–100% | 100% | 0% | 17 steps | 0.1 mm |
| **fly brain, DN candidates** | **50/50** | **93–100%** | **100%** | **0%** | **17.5 steps (0.70 s)** | **1.2 mm** |
| control: brain sees only the first state (20 seeds) | 0/20 | 0–16% | 30% | 0% | n/a | 29.7 mm |

Reached = gripper within 15 mm of the cube in x-y and below 4.2 cm, held for 0.2 s. These are the expert's own tolerances for closing the gripper.

- **The fly brain moves the arm.** Every test reach succeeded. They took about 2.5 steps (0.1 s) longer than the expert's.
- **The loop really runs through the brain.** With the input frozen at the first state, the same brain and readout never reach. The readout did not just learn a drift toward where cubes usually are.
- **The fallback (all DNs as the readout) was not needed.**

## How the readout learned (DAgger)
| round | who drives | success while collecting | knocked | rows after refit |
|---|---|---|---|---|
| 0 | expert (brain watches) | 87% | 13% | 672 |
| 1 | brain | 97% | 3% | 1,390 |
| 2 | brain | 97% | 3% | 2,092 |
| 3 | brain | 100% | 0% | 2,769 |

Round 0 knocks come from the 5 mm action noise during the expert's descent. The brain drove successfully from its first round.

## How well the brain imitates the expert (action R²)
| | dx | dy | dz |
|---|---|---|---|
| DN candidates, cross-validated on DAgger data | 0.73 | 0.89 | 0.55 |
| all DNs, same data | 0.81 | 0.90 | 0.57 |
| 4 encoded signals, no brain | 0.95 | 0.96 | 0.64 |
| DN candidates, on the 50 test episodes | 0.70 | 0.88 | 0.66 |

- **The brain adds noise, not useful nonlinearity, for this task.** The linear readout on the noise-free signals imitates better on every axis. dx loses the most (0.73 vs 0.95), which matches week 5, where x was the weakest decoded signal (0.64).
- **The week 5 worry about dz did not bite.** A linear readout (on the brain or on the signals) can't make the expert's "align first, then descend" cutoff, and the descent profile shows it: the brain drops below the expert's 13 cm travel height while still 10–100 mm from the cube. It still never knocked the cube. For the expert, travel takes about 6 steps and the descent about 9, so the gripper is over the cube before it gets low, and the brain's reach has the same timing.
- **Precision is where the brain falls short.** Its closest approach is 1.2 mm (median), against 0.1–0.2 mm for the expert and the signal readout. That is well inside the 15 mm tolerance for reaching. A grasp is less forgiving.

## Video
`week6_fly_reach.mp4`: test seeds 90000 and 90001, half speed. Left: the arm. Right: DN-candidate spikes and the brain's decoded dx, dy, dz, with the expert's actions dotted for comparison (not used). There is no failure episode, because none of the 50 test reaches failed.
The laptop could not render (no OpenGL backend in WSL), so the video was rendered elsewhere from `week6_episodes.pkl`.

## Decisions for week 7
- **Reach success rate can't compare networks: it is saturated.** The brain, the no-brain linear readout and the expert all score 50/50. In my sandbox test, even a random 3,000-neuron stand-in network reached 92–96%. Compare the connectome and the random network on precision (closest x-y distance, time to reach) and on the full pick-and-place.
- **Expand the encoder to 8 signals before pick-and-place** (as decided in week 5). The grasp and place phases need goal and height information the 4 signals don't carry.
- **Grasping needs better x precision than reaching.** The expert closes only within 15 mm, but a 1.2 mm median approach with dx R² 0.73 leaves less margin once the cube must sit between the fingers. If grasps fail, try the all-DN readout first (dx 0.81 vs 0.73, no extra brain time to refit).
- **Keep DAgger with brain-driven rounds.** Pick-and-place has more phases where the brain can drift into states the expert never visited.
- **Rendering on the laptop:** install OSMesa (`sudo apt install libosmesa6`), or render the saved episodes on Colab.
