# Week 2 — Weight Matrix & First Spiking Simulation

Oct 28, 2026 · @Mak M

## Summary

The male-CNS wiring drives known pathways in a Brian2 LIF model, and the effect comes from the specific wiring. At the chosen w\_syn of 0.1925 mV (0.7 × Shiu's 0.275 mV), confirmed on the full 176,422-neuron network:

- Sugar GRNs (LB3b/LB3c) drive proboscis motor neuron MN9 L at 55 Hz, while 0.5% of the network fires.
- JO-C/E drives 40 of 68 descending candidates (mean 42 Hz) and 9 of 81 front-leg motor neurons, while 4.6% of the network fires.
- Each stimulus barely touches the other pathway: sugar gives the descending candidates 0.2 Hz, and JO-C/E gives MN9 L 4.5 Hz.
- Randomly rewired networks at matched activity send almost nothing to MN9 (0–4.6 Hz versus 96 Hz) and drive the descending candidates about 7× less.

**Readout decision:** descending neurons are the primary readout for Week 3, and front-leg motor neurons stay a secondary one.

## Weight matrix

The signed weight set has 25,862,574 neuron-to-neuron connections (125.0M synapses) and matches the Week 1 counts exactly. It came from the [Janelia bulk export](https://male-cns.janelia.org/download/) `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (151.9M segment rows), filtered to the 176,422 neurons in the Week 1 table. No synapse cutoff is applied: 6,287,789 connections have ≥5 synapses, and that cutoff is tested in Week 3. The 122 self-connections are kept.

| Presynaptic transmitter | Sign | Synapses |
| --- | --- | --- |
| acetylcholine | + | 72.8M (58.2%) |
| GABA, glutamate, histamine | − | 48.0M (38.4%) |
| dopamine, serotonin, octopamine | 0 (baseline), + in the `sign_modexc` variant | 4.2M (3.4%) |

**Guessed transmitters:** 25,159 neurons (14%) had `unclear` or no prediction. Each got the majority transmitter of its cell type (14,366 neurons), otherwise acetylcholine (10,793). These neurons carry only 2.4% of synapses. A histamine fill is allowed only in the optic lobe, where nearly all fills are photoreceptors (R1–R8) or T1. This rule rejected 65 wrong fills, which were leg motor neurons and nerve-cord sensory neurons.

**Saved in `data/weights_male-cns_v1.0/`:**

- `neurons_signs.parquet`: neuron index 0–176,421, bodyId, type, class, raw and used transmitter, `guessed` flag, `sign`, `sign_modexc`
- `edges.parquet`: all 25.9M connections as index pairs with synapse counts
- `W_signed_counts.npz`: sparse matrix (rows = postsynaptic), synapse count × sign, w\_syn not applied; 24.3M nonzero entries
- `meta.json`: the rules above, so later weeks can confirm they load identical weights

## Simulation model

The model in `sim/lif.py` is Shiu et al. 2024's leaky integrate-and-fire network with alpha synapses. The parameters are copied from their [`model.py`](https://github.com/rmorey/Drosophila_brain_model/blob/main/model.py), not retyped from the paper.

| Parameter | Value |
| --- | --- |
| resting and reset potential | −52 mV |
| spike threshold | −45 mV |
| membrane time constant | 20 ms |
| synaptic time constant | 5 ms |
| refractory period | 2.2 ms |
| synaptic delay | 1.8 ms |
| w\_syn (Shiu, tuned on FlyWire) | 0.275 mV per synapse |
| stimulus | Poisson, 150 Hz, strength fixed at 0.275 mV × 250 |

Changes from Shiu's code:

- The network is built once and reset between trials, instead of rebuilt each trial.
- w\_syn is a shared variable, so the sweep needs no rebuild.
- Stimulus strength stays fixed while w\_syn is swept.

**Compute:** WSL is capped at 4 GB, so checks 1–4 and the sweep ran on path subnetworks of 2.4k–9.5k neurons. These keep neurons on stimulus → target paths of at most L hops, counting connections with ≥5 synapses. The full network ran on Colab (CPU). It built in 20 s, peaked at 2.4 GB, and took about 0.5 min per 2 × 1 s trials, so it would also fit on the laptop.

## Sanity checks

All four checks pass. Unless noted, the results below are from subnetworks at Shiu's w\_syn (1×). With no stimulus, every network stayed silent (0 spikes).

| Check | Setup | Result | Verdict |
| --- | --- | --- | --- |
| 1. Direct targets | 335 JO-C/E + 2,087 direct targets, 220k connections | JO-C/E 147 Hz. Targets firing: 43% of all, 81% of those with ≥50 synapses. Spearman ρ = 0.63 between synapses from JO and rate | Pass |
| 2. Sugar → MN9 | 34 LB3b/LB3c, paths ≤5 hops, 6,004 neurons, 819k connections | MN9 L 42 Hz at 50 Hz stimulus, 96 Hz at 150 Hz. MN9 R 2–15 Hz. 12% of subnetwork firing | Pass |
| 3. JO → DN → leg MN | paths ≤4 hops, 9,488 neurons, 1.41M connections | 39 of 68 DN candidates fire, all 6 "bridge" DNs included. 11 of 81 front-leg MNs fire (mean 7.9 Hz). 4 MNs are unreachable within 7 hops | Pass |
| 4. Shuffled control | degree-preserving rewiring, 3 shuffles, w\_syn lowered until activity matches the real network | MN9 L 0–4.6 Hz versus 96 Hz. DN candidates 6–8 Hz versus 48 Hz. Leg MN mean 1.5–3.0 Hz versus 7.9 Hz | Pass |

Check notes:

- **Check 1:** strongly driven targets that stay silent are inhibitory-neighborhood cases. For example, WED203 receives 1,457 synapses from JO-C/E but fires 12 Hz.
- **Check 2:** MN9 L sits 2 hops from the sugar GRNs, and MN9 R sits 3. MN9 R has about a tenth of MN9 L's inputs (633 versus 6,358, as cited by [FLYCNS](https://github.com/IONOFIELD/FLYCNS/blob/main/RESULTS.md)), so MN9 L is the readout. The sugar types follow Tastekin, de Haan Vicente et al. 2026.
- **Check 4, first attempt:** at unchanged w\_syn, the shuffled networks ran away (58–67% of neurons firing, versus 12–25% with real wiring). Targets then fired as much as or more than in the real network, which made that comparison meaningless. The real wiring contains activity that random wiring of the same size does not.
- **Check 4, leg MNs:** shuffled networks make 14–18 MNs cross 1 Hz, compared with 11 for the real wiring, but each fires weakly. The real wiring drives a few MNs hard, such as femur reductor at 122 Hz.

Rasters: `check1_direct_targets.png`, `check2_sugar_MN9.png`, `check3_JO_DN_MN.png`. Tables: `check4_shuffle.csv`, `check4_shuffle_matched.csv`.

## w\_syn sweep

The chosen value is **0.7× Shiu's w\_syn (0.1925 mV)**, and the full network confirms it. Only w\_syn was varied, over 0.25× to 4×. The selection rule was fixed before the sweep ran: take the lowest scale at which MN9 L ≥ 20 Hz and the descending candidates average ≥ 20 Hz, with neither subnetwork above 30% of its neurons firing. Both 0.7× and 1.0× meet it.

&#91;embedded content: 03\_simulation.ipynb Cell 14 (subnetworks, 3 trials per scale) and 04\_fullnet\_colab.ipynb (full network, 2 trials) · stimulus 150 Hz\]

The full network matches the subnetworks closely. At 0.7×, MN9 L is 55 Hz (53 Hz in the subnetwork) and the descending candidates average 42 Hz (40 Hz). Whole-network firing is 0.5% under sugar and 4.6% under JO-C/E, rising to 2.1% and 7.0% at 1.0×. There is no runaway, unlike the [FLYCNS](https://github.com/IONOFIELD/FLYCNS/blob/main/RESULTS.md) male-CNS model, which ran away near its 0.5 scale.

**Caveat:** 0.7× sits just above MN9's switch-on point, so small changes to the weights could silence MN9. 1.0× is the fallback, and it also meets the rule. The front-leg MN mean plateaus near 10 Hz from 1× upward, so more weight barely gets more signal through the nerve cord.

## Descending neurons and readout decision

**38 of the 68 descending candidates fire at 0.7×** (subnetwork, 10 trials, ≥1 Hz mean). They are saved to `dn_pruned_week2.csv`, with rates at 0.7× and 1×. On the full network the count is 40.

| Week 1 role | Candidates | Fired | Mean rate (Hz) |
| --- | --- | --- | --- |
| bridge | 6 | 6 | 90.2 |
| output-side | 18 | 9 | 41.9 |
| input-side | 44 | 23 | 32.5 |

The strongest are DNp18 L (157 Hz), DNp73 L (152 Hz), DNb05 L/R (147/143 Hz), DNge012 L/R, DNpe017 L, DNge078 L/R, DNp10 L/R and DNge011 L. Most fire on both sides. Several descending neurons outside the candidate list fire just as hard: DNge027, DNbe001, DNg29, DNge044 and DNge039 (165–190 Hz). The Week 1 two-hop scoring missed them.

**Readout decision.** Activity from JO-C/E reaches the front-leg motor neurons, but weakly and in a small subset. At 0.7×, 9 of 81 fire, mean 5.5–5.8 Hz, led by femur reductor, tibia extensor and tarsus levator MNs. Their response plateaus near 10 Hz as w\_syn rises. In contrast, the descending candidates respond strongly (mean 40–42 Hz), scale smoothly with w\_syn, and drop about 7× under activity-matched rewiring. **Descending-neuron readout is primary for Week 3. Front-leg motor neurons stay a secondary readout**, to revisit once the nerve-cord circuit is simulated without subnetwork cuts.

## Deliverables and open questions

| Deliverable | Where |
| --- | --- |
| Weight set (matrix, sign table, guessed flags) | `data/weights_male-cns_v1.0/` |
| Brian2 simulation module | `sim/lif.py` |
| Weights notebook | `02_weights.ipynb` |
| Results notebook, checks 1–4, rasters, rate tables | `03_simulation.ipynb`, `check*.png`, `check4_*.csv` |
| w\_syn sweep plot and chosen value | `wsyn_sweep.png`, `wsyn_sweep_subnetworks.csv`, `wsyn_choice.json` |
| Full-network confirmation | `04_fullnet_colab.ipynb`, `fullnet_week2.csv` |
| Pruned descending list | `dn_pruned_week2.csv` (38 neurons) |
| Readout decision | this doc: descending neurons primary |

Open questions for Week 3:

- [ ] Test the ≥5-synapse cutoff: do MN9 L and the descending responses hold at 0.7× with 6.3M connections instead of 25.9M?
- [ ] Widen the descending readout to include strong non-candidates (DNge027, DNbe001, DNg29, DNge044, DNge039), or rank all 1,314 DNs by full-network rate.
- [ ] JO-C/E drives 175 of 708 VNC motor neurons on the full network, but only 9 of the 81 front-leg (ProLN) set. Which motor neurons are the other 166?
- [ ] MN9 R barely responds: is that sparse reconstruction (633 input synapses) or real asymmetry? The sugar GRNs have no side labels to test contralateral bias.
- [ ] Re-run the shuffled control on the full network, without the path-selection bias of subnetworks.
- [ ] The full network fits in 2.4 GB, so full-network runs can move back to the laptop. Is Colab still needed before the GPU port?

Sources: [MaleCNS download page](https://male-cns.janelia.org/download/) · [Shiu et al. model.py](https://github.com/rmorey/Drosophila_brain_model/blob/main/model.py) · [FLYCNS results](https://github.com/IONOFIELD/FLYCNS/blob/main/RESULTS.md)
