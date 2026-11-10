# Week 4 results: Panda environment and expert baseline

Setup: Panda from mujoco_menagerie (commit 98a70c5), MuJoCo 3.17.0, 2 ms physics. Table, 5 cm 80 g cube, green target zone of 5 cm radius. Cube spawns in x 0.35–0.60 m, y −0.25–0.15 m, with yaw 0. Actions at 25 Hz (40 ms): gripper dx, dy, dz (max 2 cm per step, so 0.5 m/s) plus open/close. Run in a cloud container on 2026-11-04, CPU only.

## Expert success rate (deliverable)
Seeds 1000–1199 were used for tuning. The numbers below use fresh seeds, and the expert was frozen before they ran.

| test | success | 95% CI | median time | median placement error |
|---|---|---|---|---|
| random cube positions | **500 / 500** | 99.2–100% | 49 steps (1.96 s) | 3.5 mm (max 5.8 mm) |
| 11 × 9 grid over the spawn area, corners and edges included | 99 / 99 | 96.3–100% | 49 steps | 3.6 mm |

Success is measured from physics, not from the expert: cube in zone, on the table, speed under 1 cm/s, not touching the gripper, gripper at least 6 cm above it, all held for 10 steps (0.4 s).
The placement error has a small fixed offset of (−3.3, −1.2) mm. I did not trace its cause. It is well inside the zone.

## Where the expert breaks (100 fresh seeds each)
| condition | success | median time (steps) | median placement error |
|---|---|---|---|
| action noise 2 / 5 mm per step | 100 / 100 | 51 / 58 | 4.5 / 8.3 mm |
| action noise 10 mm per step | 94 | 104 | 17.1 mm |
| action noise 20 mm per step | 10 | 166 | 28.3 mm |
| cube sensing noise 2 / 5 mm | 100 / 99 | 48 / 53 | 3.9 / 5.3 mm |
| cube yaw 15° | 100 | 48 | 4.0 mm |
| cube yaw 30° | 99 | 158 (needs retries) | 4.3 mm |
| cube yaw 45° | 1 | n/a | n/a |
| cube mass 240 / 400 / 800 g | 100 / 99 / 93 | 48 | about 3 mm |
| friction 0.5 / 0.3 everywhere | 100 / 100 | 48 | about 3 mm |

Yaw is the hard limit: the action has no rotation, so a cube near 45° cannot be gripped. The cube stays at yaw 0 in this project unless a rotation action is added.

## Controller (15 moves at 0.5 m/s across the spawn corners and the zone)
- Damped least-squares IK with a null-space posture term at 500 Hz, driving the model's position servos with gravity and velocity feed-forward.
- Settled error: median 0.25 mm, worst 0.48 mm. Peak tilt 2.4°.
- Tracking error in motion: 4.9 mm RMS, 24.5 mm peak (about one action step of lag).
- Limits to know: a full-speed descent stopped dead overshoots by about 15 mm, and the arm briefly hits about 1.2 m/s while catching up early in a move. The expert slows down near its goals, so it avoids the first one. An imitator might not.

## Dataset: `pick_place_expert.npz`
- 1,000 episodes, 63,794 rows: train 800 episodes (51,121 rows), val 200 episodes (12,673 rows). Split by whole episode, with separate seeds. About 6.6 MB.
- Half the episodes are clean. Half add 5 mm per step of noise to the executed motion, while the label stays the clean expert action.
- Expert success in the data: 500 / 500 clean, 497 / 500 noisy. Failed episodes are kept and flagged (`ep_success`).
- Each row: 24 state values (`obs_names`), the 4-value action label, the executed action, phase label, episode number.
- 66% of steps have an axis at the 2 cm limit, so actions are often saturated.

## Checks
- Success detector gave the right answer on 5 hand-made cases. Do-nothing and random policies: 0 / 20 each.
- Same seed twice gave identical trajectories. Replaying the logged actions for all 200 val episodes gave the same final cube position (0.0 mm difference).
- Recomputing the expert on all 63,794 logged states reproduced every label, so the expert really is a function of the logged state. No NaNs, and no physics instability in evaluation or collection.

## Can the data be imitated? (closed loop, 100 fresh seeds per test)
| policy | training data | no noise | with 5 mm per step action noise |
|---|---|---|---|
| linear ridge on raw state | clean + noisy | 0 | not run |
| MLP 256×256, 3 training seeds | clean only | 100, 100, 98 | 60, 74, 50 |
| MLP 256×256, 3 training seeds | clean + noisy | 94, 99, 100 | 100, 100, 100 |

Model change: the Menagerie gripper servo is scaled 10× (about 2.5 N to about 25 N of tendon force), because the cube slipped and swung in the fingers. With an early version of the expert, 85% of dev episodes succeeded before this change, and the cube visibly shifted in the grip.

## Decisions for weeks 5–7
- **Train on clean + noisy.** Clean-only imitators fall apart under the noise a decoder will add.
- **The linear baseline matters.** A linear map of the raw state scores 0%, so if a linear readout of the brain works, the nonlinearity came from the brain. The week 7 comparison should include this baseline.
- **Time budget on the laptop:** a typical episode is about 60 steps, or 2.4 s of brain time, so about 11 s at 4.63 s per simulated second. The 10 s cap costs about 46 s. A policy trained at 40 ms per action should keep that rate, or the dataset needs re-collecting.
