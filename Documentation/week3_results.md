# Week 3 results: speed and subnetwork extraction

Setup: male-CNS v1.0, 176,422 neurons, 24.3M connections. JO-C/E stimulus at 150 Hz, w_scale 0.7. Runs on Colab (2026-10-31: 2 CPU cores and a T4 GPU) and on the laptop (2026-11-04: WSL capped at 4 GB, CPU only).

## Verification against Brian2 (identical input spikes)
- Colab, full network: numba (float64) and torch-gpu-f64 are identical to Brian2, every spike (738,786 of 738,786).
- Laptop, path L=4 subnetwork: numba is identical to Brian2 (425,623 of 425,623).
- float32 engines: individual spikes diverge from 28 ms, but rates agree (per-neuron correlation 0.9998, same DN readout). One rounding difference flips a spike and the network amplifies it, as happens between two seeds. This is not a bug. Use float64 when comparing against Brian2.
- Port detail: Brian2 discards synaptic input to refractory neurons, because of `(unless refractory)`. The engines replicate this.
- Reproducibility: the Step 2 and Step 3 numbers are identical on Colab and the laptop.

## Benchmark: seconds of compute per 1 s of brain time (single trial)
| network | neurons | connections | brian2 (Colab / laptop) | numba (Colab / laptop) | torch-cpu (Colab / laptop) | torch-gpu | torch-gpu-f64 | torch-gpu-spmv |
|---|---|---|---|---|---|---|---|---|
| full | 176,422 | 24,287,791 | 14.22 / – | 6.25 / **4.63** | 32.80 / 20.44 | 10.82 | 10.80 | 20.95 |
| path L=5 | 74,874 | 12,335,607 | 6.89 / 4.39 | 2.92 / 1.91 | 19.20 / 14.63 | 10.97 | 10.93 | 13.16 |
| path L=4 | 18,648 | 3,039,649 | 1.87 / 1.91 | 0.84 / 0.55 | 9.99 / 9.10 | 10.95 | 10.74 | 8.03 |
| active set | 10,677 | 926,228 | 1.56 / 1.63 | 0.72 / 0.51 | 9.08 / 8.18 | 10.89 | 10.78 | 7.68 |
| path L=3 | 4,530 | 496,997 | 0.83 / 0.73 | 0.27 / 0.19 | 6.43 / 5.49 | 10.91 | 11.13 | 7.85 |

GPU columns are from Colab only. Brian2 on the full network was not run on the laptop, which would exceed the 4 GB limit.
GPU batch throughput on the full network (Colab T4): 10.88 s per trial at batch 1, 1.71 s at batch 10, 1.21 s at batch 32.
Single-trial GPU time is flat at about 10.9 s for every network size. That is about 1.1 ms of fixed overhead per 0.1 ms step (Python loop, kernel launches, and a GPU-to-CPU sync each step), so it is not compute-bound.
Advancing the Numba engine in closed-loop-sized chunks (1–10 ms of brain time) costs no measurable extra time (tested on a full-size fake network).

## Subnetwork fidelity (all pass the pre-set rule; identical on both machines)
| network | DN corr A / B | DN mean ratio A / B | leg MNs firing A / B (full 9 / 10) |
|---|---|---|---|
| path L=3 | 0.974 / 0.962 | 0.944 / 0.987 | 8 / 8 |
| path L=4 | 0.996 / 0.997 | 1.018 / 0.980 | 9 / 9 |
| path L=5 | 1.000 / 1.000 | 1.005 / 1.009 | 9 / 9 |
| active set | 1.000 / 0.999 | 0.998 / 0.990 | 9 / 9 |

## Decision (final)
- **Closed loop: full network on the Numba CPU engine, on the laptop.** At 4.63 s per simulated second it passes the 10× checkpoint with about 2× headroom for MuJoCo and feedback-driven activity. It loses no fidelity, fits in 4 GB, and new inputs can be added without re-extracting anything.
- **Fallback: the active set** (0.51 s on the laptop, about 9× faster, DN correlation ≥ 0.999). It was built from JO-C/E runs only, so rebuild it from closed-loop runs before relying on it.
- **GPU (Colab): batched sweeps and many-episode runs** (1.21 s per trial at batch 32), not the single-trial loop.
