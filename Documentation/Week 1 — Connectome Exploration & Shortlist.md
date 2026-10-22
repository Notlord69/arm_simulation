# Week 1 — Connectome Exploration & Shortlist

Oct 22, 2026 · @Mak M

## Summary

Week 1 is done: the male-CNS v1.0 connectome is downloaded, explored, and narrowed to an antennal-grooming pathway of 335 input, 68 descending, and 81 front-leg motor neurons.

The headline finding: no descending neuron is strongly wired to both the antennal sensors and the front-leg motor neurons within two hops. The path on the nerve-cord side is likely three or more hops long. The Week 2 simulation will decide which descending neurons actually carry the signal.

All IDs are saved in `brain/data/shortlist_week1.csv`.

## Setup and data

The dataset is **male-CNS v1.0** on neuPrint: a complete male fly brain and nerve cord in one animal, so sensory, descending and motor neurons are all present.

| Item | Value |
| --- | --- |
| Server | neuprint.janelia.org |
| Dataset | `male-cns:v1.0` |
| Client | neuprint-python 0.6.4 |
| Environment | `sim-env` (shared with the MuJoCo arm code) |
| Token | `NEUPRINT_TOKEN` in `~/.bashrc`, not in code |
| Notebook | `brain/notebooks/01_exploration.ipynb` |

Files saved in `brain/data/`:

- `neurons_male-cns_v1.0.parquet`: the full neuron table (ID, type, class, subclass, neurotransmitter, side, synapse counts)
- `weight_histogram.parquet`: number of connections at each synapse count
- `dn_ranking_week1.csv`: every descending neuron's input, output and combined score
- `shortlist_week1.csv`: the three groups with IDs

## Dataset overview

The dataset holds 176,422 neurons and 25.9 million connections; about half the neurons are optic-lobe interneurons that a grooming subnetwork can mostly leave out.

### Neuron classes

| Class | Neurons | Role for this project |
| --- | --- | --- |
| ol\_intrinsic | 89,403 | Visual processing; mostly droppable |
| cb\_intrinsic | 32,164 | Central brain interneurons; part of the input path |
| vnc\_intrinsic | 13,161 | Nerve-cord interneurons; part of the output path |
| None (unclassified) | 9,722 | Kept in the weight matrix, kept off the shortlist |
| vnc\_sensory | 6,370 | Includes 1,030 leg proprioceptors (future arm feedback) |
| cb\_sensory | 4,868 | Includes 1,733 head mechanosensors, of which 335 are JO-C/E |
| ascending\_neuron | 1,846 | Nerve cord to brain |
| descending\_neuron | 1,314 | Brain to nerve cord; the middle group |
| vnc\_motor | 708 | Body motor neurons; 81 front-leg ones are the output |
| cb\_motor | 107 | Head, proboscis and neck muscles |

Smaller classes (efferent, endocrine, enteric, and "tbc" categories) add up to about 1,500 neurons.

### Neurotransmitters

| Predicted transmitter | Neurons | Sign in Week 2 |
| --- | --- | --- |
| Acetylcholine | 95,409 | Excitatory |
| Glutamate | 28,199 | Inhibitory |
| Unclear | 22,902 | To be filled (see open questions) |
| GABA | 20,357 | Inhibitory |
| Dopamine | 4,447 | Modulatory; rule to decide |
| None | 2,257 | To be filled |
| Histamine | 2,242 | Mostly photoreceptors; likely dropped |
| Serotonin | 483 | Modulatory; rule to decide |
| Octopamine | 126 | Modulatory; rule to decide |

The gaps fall where they matter least. Motor neurons are 95% unclear, but they are the readout, so their sign never feeds back into the network. Descending neurons are only 1% unclear. Sensory neurons are 13–24% unclear, and their confident predictions are 75–85% acetylcholine.

### Connection weights

Of 25,862,574 connections, only 6,287,789 (24%) have 5 or more synapses. Most connections are weak, so a ≥5-synapse cutoff would shrink the weight matrix about fourfold. Week 3 should test whether that cutoff changes the simulation's result.

## Shortlist

The pathway is antenna touch → descending command → front-leg movement, the antennal-grooming reflex that Shiu et al. 2024 validated in a model like this one. The front leg stands in for the robot arm.

| Group | Neurons | Selection rule |
| --- | --- | --- |
| Input: JO-C/E antennal sensors | 335 | `superclass = cb_sensory`, type starts `JO-C` or `JO-E` |
| Middle: descending neurons | 68 | Top 10 by input, output and combined score, plus all members of those types |
| Output: front-leg motor neurons | 81 (41 L, 40 R) | `superclass = vnc_motor`, `somaNeuromere = T1`, `exitNerve = ProLN` |

### Input: Johnston's organ C and E neurons

Johnston's organ is the antenna's touch and vibration sensor; its C and E subtypes trigger antennal grooming (Hampel et al. 2015). Types: JO-EV1 (50), JO-EV3 (45), JO-ED2\_a (38), JO-EV2 (36), JO-CM (30), JO-EV5 (23), JO-EV6 (21), JO-ED2\_b (21), JO-CL (19), JO-ED1 (18), JO-ED2\_c (14), JO-CA1 (10), JO-CA2 (9), JO-EV4 (1).

A first pass used all 1,733 head mechanosensors. It was dropped because 745 of them (`BM_InOm`) are eye bristles, which pulled the ranking toward eye grooming.

### Output: front-leg motor neurons

These are T1 motor neurons leaving through the prothoracic leg nerve (ProLN). Filtering on the nerve removes neck and flight motor neurons that also sit in T1. Types: Acc. ti flexor (19), Ti flexor (10), Ta depressor (9), ltm (8), Fe reductor (8), Ta levator (5), Acc. tr flexor (5), Ti extensor (4), ltm2-femur (4), ltm1-tibia (3), Tr flexor (2), Tr extensor (2).

Ti = tibia, Tr = trochanter, Fe = femur, Ta = tarsus, ltm = long tendon muscle.

### Middle: top 25 descending neurons

Role: input-side when in\_score is over 3× out\_score, output-side when the reverse, otherwise bridge.

| bodyId | Instance | in\_score | out\_score | score | Role |
| --- | --- | --- | --- | --- | --- |
| 10048 | DNp18\_L | 0.122 | 0.008 | 0.032 | input-side |
| 36541 | DNge078\_R | 0.007 | 0.124 | 0.030 | output-side |
| 10425 | DNp10\_L | 0.026 | 0.029 | 0.027 | bridge |
| 10433 | DNp10\_R | 0.021 | 0.034 | 0.027 | bridge |
| 10527 | DNpe017\_L | 0.041 | 0.016 | 0.026 | bridge |
| 12453 | DNg50\_R | 0.014 | 0.043 | 0.024 | output-side |
| 10030 | pIP1\_L | 0.043 | 0.014 | 0.024 | input-side |
| 13065 | DNge084\_L | 0.156 | 0.003 | 0.022 | input-side |
| 10118 | DNb05\_L | 0.014 | 0.033 | 0.022 | bridge |
| 10587 | DNge011\_L | 0.007 | 0.054 | 0.020 | output-side |
| 11458 | DNge043\_L | 0.023 | 0.015 | 0.019 | bridge |
| 10354 | DNp73\_R | 0.100 | 0.004 | 0.019 | input-side |
| 555871 | DNpe017\_R | 0.026 | 0.012 | 0.018 | bridge |
| 11600 | DNge084\_R | 0.116 | 0.003 | 0.017 | input-side |
| 15653 | DNge012\_L | 0.003 | 0.107 | 0.017 | output-side |
| 10026 | DNp18\_R | 0.067 | 0.004 | 0.017 | input-side |
| 10038 | pIP1\_R | 0.027 | 0.010 | 0.017 | bridge |
| 14537 | DNge078\_L | 0.003 | 0.099 | 0.016 | output-side |
| 10446 | DNp73\_L | 0.269 | 0.001 | 0.016 | input-side |
| 13130 | DNg105\_R | 0.001 | 0.176 | 0.015 | output-side |
| 11067 | DNb06\_R | 0.003 | 0.066 | 0.015 | output-side |
| 13178 | DNge039\_L | 0.004 | 0.055 | 0.014 | output-side |
| 10065 | DNb05\_R | 0.007 | 0.023 | 0.012 | output-side |
| 513052 | DNg15\_R | 0.009 | 0.017 | 0.012 | bridge |
| 10274 | DNg108\_L | 0.002 | 0.065 | 0.012 | output-side |

The other 43 of the 68 are partner sides of these types plus whole types pulled in by the input or output top 10, such as DNg07 and DNge091; several of those have an output score of zero. The full list is in `shortlist_week1.csv`.

## Connectivity findings

The antennal pathway exists but is narrow: the best descending neuron scores 0.032, about a fifth of what the eye-bristle inputs produced in the first pass.

&#91;embedded content: antennal grooming pathway · 3 groups\]

The ranking splits descending neurons into two kinds. Input-side neurons hear the antenna well but barely reach the leg within two hops: DNp73\_L (in 0.27), DNge084\_L (0.16), DNp18\_L (0.12). Output-side neurons drive the leg strongly but hear little from the antenna: DNg105\_R (out 0.18), DNge078\_R (0.12), DNge012\_L (0.11).

The likely reason is that the nerve-cord side runs DN → interneuron → premotor neuron → motor neuron, one hop beyond the window. That fits biology, since local leg circuits coordinate grooming. It also means the Week 3 subnetwork needs at least 3–4 hops on the nerve-cord side.

Left and right copies of the same types rank together (DNp10, DNp18, DNge078, DNge084, DNpe017, pIP1, DNb05). An artifact would not pick both copies so consistently, which supports the ranking.

831 of 1,314 descending neurons have some path on both sides at two hops, so path existence says little; the score is what separates them.

### How the scores work

Each hop is weighted by the fraction of the receiving neuron's total input it supplies, counting only connections of 5 or more synapses. A two-hop path's weight is the product of its two fractions.

- **in\_score:** direct sensor → DN fractions, plus all sensor → X → DN paths
- **out\_score:** direct DN → motor neuron fractions, plus all DN → Y → motor neuron paths
- **score:** the geometric mean of the two, √(in\_score × out\_score), so a neuron ranks high only if it is strong on both sides

## Open questions for Week 2

- [ ] **Unclear and missing transmitters (\~25,000 neurons):** fill each from the majority transmitter of its own cell type, fall back to excitatory, and flag which were guessed.
- [ ] **Modulatory transmitters (dopamine, serotonin, octopamine; \~5,000 neurons):** treat as excitatory, or zero their outgoing weights. Run both and compare.
- [ ] **Which descending neurons carry the signal:** stimulate JO-C/E and record which of the 68 fire, then whether the front-leg motor neurons follow.
- [ ] **Weight cutoff:** compare the full weight matrix against connections ≥5 synapses only.
- [ ] **One leg or two:** the arm is a single limb, so the output will eventually be one front leg (41 or 40 motor neurons). Keep both for the Week 2 sanity check, since grooming uses both legs.
