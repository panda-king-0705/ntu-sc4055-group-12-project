# Noise analysis (Person 4)

How the trained quantum models hold up under simulated hardware noise, for both the **VQC** (Person 2) and the **QSVC** (Person 3). Each model is run with and without entanglement, through:

- **depolarizing noise** at a range of CNOT error rates, and
- **FakeManilaV2**, a noise model of a real 5-qubit IBM device: gate errors, readout errors and T1/T2 relaxation, with its line topology 0–1–2–3–4.

Nothing is retrained on noisy data. We use the models exactly as Persons 2 and 3 trained them, noiseless, and only change how their circuits are executed. That matches what would happen on real hardware.

Reproduce with:

```
python scripts/noise_vqc.py     # VQC
python scripts/noise_qsvc.py    # QSVC, about 3 h (350k kernel circuits per noise level)
```

Both scripts take `--quick` for a smoke test in a few minutes. Outputs go to this folder.

## Setup

| | VQC | QSVC |
|---|---|---|
| Models | `results/vqc/best_entangled.npz` (8 CNOTs) and `best_no_entanglement.npz` (0 CNOTs) | Tuned `Z_r2`, α=0.3, C=0.1 (0 CNOTs); best entangling map by validation `ZZ_full_r1`, α=0.05 (24 CNOTs in the kernel circuit) |
| Readout | parity, P(fail) = (1 − ⟨ZZZZ⟩)/2 | kernel entry = P(0000) of the compute–uncompute circuit |
| Test rows | all 4,291 (620 failing) | every 5th row: 859 (126 failing), because every test row costs 319 kernel circuits |
| Classical reference | RBF SVM from `results/qsvc/e2_comparison.csv`: F1 0.773, 15 missed | the same RBF SVM refit on the 859 rows: F1 0.790, 2 missed |

**Compare each quantum model only with the classical row on the same test rows.**

**Depolarizing model**: CNOT error p, and one-qubit gates p/10 (real devices are roughly 10× better on one-qubit gates). Levels: 0, 0.1%, 0.5%, 1%, 2%, 5%, 10%, 20% for the VQC. The QSVC drops 0.5% and 20% because each level takes about 10 min.

**FakeManila**: circuits are transpiled to the device first (`optimization_level=1, seed_transpiler=0`), or the noise model never applies. On the line topology, the transpiler has to insert SWAPs:

| Circuit | CNOTs, logical | CNOTs on Manila |
|---|---|---|
| VQC entangled | 8 | **20** |
| VQC no entanglement | 0 | 0 |
| QSVC entangled kernel | 24 | **36** |
| QSVC no entanglement | 0 | 0 |

**Shots**: 1,000 per circuit unless stated. Shot-noise runs are repeated over 3–5 seeds and reported as mean ± std.

**Breaking point**, fixed before running anything: the smallest CNOT error at which, with 1,000 shots, the model's mean F1 drops below the classical reference, **or** its missed failures reach max(2 × noiseless, noiseless + 1).

**Extra VQC metrics**: F1 alone looks flat, because noise mostly pulls P(fail) towards 0.5 without pushing it across. So every VQC row also has:
- **loss**: BCE on the test set;
- **confidence**: mean |P(fail) − 0.5|; 0 means every prediction is a coin flip.

**Extra QSVC metrics**: `k_std`, the spread of the off-diagonal training kernel entries, and `kta_train`, the kernel-target alignment. These show the kernel itself degrading. C is re-tuned on the noisy validation kernel, as you would do on hardware. The `fixedC_` columns keep the noiseless C instead; the results barely differ.

Uncertainty for the FakeManila comparisons comes from the **engine-level bootstrap** in `qsvc_lib.engine_bootstrap`, the same method as `results/qsvc/`.

## VQC results

Figures: `figures/vqc_depolarizing.png`, `figures/vqc_shots.png`, `figures/vqc_fakemanila.png`.

### N1: depolarizing, exact expectation values (`vqc_n1_depolarizing.csv`)

| CNOT error | Entangled F1 / missed | Entangled confidence | Entangled loss | No-ent F1 / missed | No-ent confidence | No-ent loss |
|---|---|---|---|---|---|---|
| 0 | 0.791 / 27 | 0.253 | 0.324 | 0.796 / 41 | 0.208 | 0.375 |
| 1% | 0.791 / 27 | 0.229 | 0.352 | 0.796 / 41 | 0.203 | 0.381 |
| 5% | 0.791 / 27 | 0.153 | 0.448 | 0.796 / 41 | 0.185 | 0.404 |
| 10% | 0.791 / 27 | 0.091 | 0.538 | 0.796 / 41 | 0.164 | 0.432 |
| 20% | 0.790 / 27 | **0.029** | **0.639** | 0.796 / 41 | 0.128 | 0.481 |

Without shot noise, the predictions never flip. Depolarizing noise shrinks ⟨ZZZZ⟩ towards 0, so P(fail) moves towards 0.5 but stays on the same side. Confidence shows the real damage: the entangled model starts more confident but loses it about 4× faster. **The two curves cross at about 2% CNOT error.**

### N2: shot noise only, noiseless circuit (`vqc_n2_shots.csv`, 5 seeds)

| Shots | Entangled F1 / missed | No-ent F1 / missed |
|---|---|---|
| 100 | 0.767 / 30.2 | 0.745 / 59.4 |
| 1,000 | 0.788 / 27.4 | 0.789 / 45.0 |
| 10,000 | 0.791 / 27.2 | 0.796 / 42.2 |

At 1,000 shots, shot noise costs almost nothing. It is not why the VQC trails the classical model on missed failures.

### N3: depolarizing + 1,000 shots (`vqc_n3_depolarizing_shots.csv`, 3 seeds)

This is the realistic combination. Once noise has pulled predictions near 0.5, shot noise flips them.

| CNOT error | Entangled F1 / missed | No-ent F1 / missed |
|---|---|---|
| 0 | 0.786 / 26.3 | 0.786 / 46.3 |
| 2% | 0.785 / 25.0 | 0.789 / 45.3 |
| 10% | 0.775 / 28.7 | 0.786 / 48.0 |
| 20% | **0.678 / 60.0** | 0.782 / 49.3 |

**Breaking point:**
- Entangled: **20% CNOT error**, on both criteria (F1 0.678 < 0.773, and missed failures 60 ≥ 53).
- No entanglement: **never**, within the range tested.

### N4: FakeManila, 1,000 shots (`vqc_n4_fakemanila.csv`)

| Variant | Entangled F1 / missed | Entangled confidence | No-ent F1 / missed | No-ent confidence |
|---|---|---|---|---|
| Noiseless (N1) | 0.791 / 27 | 0.253 | 0.796 / 41 | 0.208 |
| **Full device noise** (5 seeds) | **0.784 / 28.2** | **0.134** | **0.785 / 47.4** | **0.142** |
| Readout errors only | 0.788 / 27.7 | 0.183 | 0.785 / 49.0 | 0.145 |
| Gate errors only | 0.786 / 27.7 | 0.186 | 0.788 / 45.3 | 0.203 |
| Gate + readout errors × 2 | 0.769 / 33.0 | 0.071 | 0.770 / 58.0 | 0.091 |
| Gate + readout errors × 4 | 0.443 / 156.3 | 0.016 | 0.595 / 122.0 | 0.025 |

Noiseless − FakeManila F1, engine bootstrap 95% CI:
- Entangled: [−0.002, +0.015], which includes 0.
- No entanglement: [+0.001, +0.026], a small but real loss.

- **On today's device error rates, both VQCs keep their F1.** They lose about half their confidence.
- **The no-entanglement model is hurt mostly by readout errors.** Its gate-only confidence is almost noiseless (0.203 vs 0.208). With a parity readout, a flip on any of the 4 bits flips the prediction, whether or not the circuit has CNOTs.
- **The entangled model is hurt equally by readout and by its 20 transpiled CNOTs.**
- **At 2× device error both models start to break, and at 4× they collapse.** The entangled one collapses harder: F1 0.44 vs 0.60.

The scaled variants (×0.5 to ×4) scale gate and readout errors but leave out T1/T2 relaxation. Scaling T1/T2 on the fake backend did not take effect. Leaving it out barely matters: ×1 without T1/T2 gives the same F1 as the full model (0.784) and similar confidence (0.138 vs 0.134).

## QSVC results

Figure: `figures/qsvc_noise.png`.

### Q1: depolarizing, exact kernel entries (`qsvc_q1_depolarizing.csv`)

| CNOT error | No-ent F1 / missed | No-ent k_std | No-ent KTA | Entangled F1 / missed | Entangled k_std | Entangled KTA |
|---|---|---|---|---|---|---|
| 0 | 0.803 / 2 | 0.174 | 0.497 | 0.807 / 7 | 0.123 | 0.476 |
| 1% | 0.803 / 2 | 0.171 | 0.497 | 0.812 / 7 | 0.099 | 0.468 |
| 2% | 0.803 / 2 | 0.167 | 0.497 | 0.810 / 7 | 0.079 | 0.450 |
| 5% | 0.805 / 2 | 0.158 | 0.497 | 0.799 / 5 | 0.041 | 0.322 |
| 10% | 0.803 / 2 | 0.144 | 0.496 | **0.777 / 9** | **0.014** | **0.138** |

**The entangled kernel concentrates.** As noise grows, every kernel entry is pulled towards the same value, so the spread shrinks almost 9× and alignment with the labels falls from 0.48 to 0.14. The SVM is left with less and less signal. This is the noise-induced concentration predicted by Wang, Du, Luo & Tao (*Quantum* 5, 531, 2021), and it grows with circuit depth: 24 noisy CNOTs compound.

**The kernel without entanglement barely changes.** Its kernel values shrink a little, but the SVM absorbs a uniform shrink, so predictions are identical at every level.

### Q2: depolarizing + 1,000 shots (`qsvc_q2_depolarizing_shots.csv`, 3 repetitions)

| CNOT error | No-ent F1 / missed | Entangled F1 / missed |
|---|---|---|
| 0 | 0.799 ± 0.003 / 3.0 | 0.793 ± 0.007 / 6.7 |
| 1% | 0.802 ± 0.003 / 2.3 | 0.791 ± 0.018 / 5.3 |
| 2% | 0.800 ± 0.006 / 2.3 | **0.782 ± 0.014** / 7.3 |
| 5% | 0.803 ± 0.003 / 2.0 | 0.787 ± 0.011 / 6.3 |
| 10% | 0.800 ± 0.005 / 2.0 | **0.755 ± 0.022 / 10.0** |

**Breaking point:**
- Entangled: **2% CNOT error** (F1 below 0.790). This threshold is soft. The run-to-run spread is ±0.014, the 5% level recovers to 0.787, and even at 0 noise the entangled kernel sits only 0.003 above classical. Describe it as "drops below classical from about 2%, and clearly by 10%". Missed failures never reach the doubling threshold.
- No entanglement: **never**.

### Q3: FakeManila, 1,000 shots (`qsvc_q3_fakemanila.csv`, one run)

| Kernel | CNOTs on Manila | F1 / missed | k_std | KTA | Noiseless − Manila F1, 95% CI |
|---|---|---|---|---|---|
| No entanglement | 0 | 0.800 / 2 | 0.142 | 0.494 | [0.000, +0.008] |
| Entangled | 36 | **0.778 / 5** | **0.071** | 0.419 | **[+0.004, +0.052]** |

On the simulated device, **only the entangled kernel loses F1 measurably**, and it drops below the classical model. Its kernel spread falls to about the level of 2–5% depolarizing noise, because the SWAPs make it 36 CNOTs deep.

## Key findings

1. **Entanglement is where noise hurts.** In both models, the entangled version degrades and the version without entanglement does not, except through readout errors. Every noisy CNOT adds damage, and mapping to a real device's topology adds more CNOTs: the VQC goes from 8 to 20, the QSVC kernel from 24 to 36.
2. **Noise shows up first as lost confidence, then as wrong answers.**
   - VQC: P(fail) is pulled towards 0.5. Accuracy holds until shot noise starts flipping the borderline predictions.
   - QSVC: the kernel concentrates, and kernel-target alignment collapses before F1 does.

   Plotting F1 alone hides both effects.
3. **Breaking points** (CNOT error at 1,000 shots):

   | Model | Breaking point |
   |---|---|
   | Entangled QSVC | about **2%** (soft) |
   | Entangled VQC | **20%** |
   | QSVC, no entanglement | none |
   | VQC, no entanglement | none |

   Today's IBM devices have CNOT errors of about 0.5–1%. The QSVC is more fragile than the VQC because its circuit is twice as deep (U followed by U†).
4. **At current device noise (FakeManila), every model keeps an F1 of about 0.78–0.80.** Only the entangled QSVC loses a statistically clear amount. Persons 2 and 3 found that the models without entanglement already score about as well noiseless. **On this dataset, entanglement adds noise sensitivity without adding accuracy.**
5. **Shot noise matters only at low shot counts.** At 1,000 shots it costs about 0.005 F1; at 100 shots, the no-entanglement VQC misses 45% more failures.

## Caveats

- **The QSVC was scored on every 5th test row**, for runtime. Its numbers are only comparable with the classical row refit on the same 859 rows. The VQC uses all 4,291 rows.
- **The classical reference is Person 3's RBF SVM.** Swap in Person 1's classical models once they are pushed.
- **The QSVC FakeManila result is a single run.** Its uncertainty comes from the engine bootstrap only, not from repeated shots.
- **Simulated noise is not real hardware.** FakeManila is a snapshot of one device's calibration, and depolarizing noise is a simplification: real errors are correlated, drift and include crosstalk.
- **The FakeManila sampler snippet in `docs/vqc_handoff.md` applies no noise.** It sets `sampler.options.backend_options` after construction, which Aer ignores, so the run is noiseless. Pass the noise model in the constructor, or use `SamplerV2.from_backend(backend)`, as `scripts/noise_vqc.py` does.

## Files

| File | Contents |
|---|---|
| `vqc_n1_depolarizing.csv` … `vqc_n4_fakemanila.csv` | VQC experiments N1–N4, one row per model × level × seed |
| `qsvc_q1_depolarizing.csv` … `qsvc_q3_fakemanila.csv` | QSVC experiments Q1–Q3 |
| `vqc_summary.json`, `qsvc_summary.json` | breaking points, transpiled circuit stats, bootstrap CIs |
| `vqc_run.log`, `qsvc_run.log` | console output of the full runs |
| `figures/` | `vqc_depolarizing.png`, `vqc_shots.png`, `vqc_fakemanila.png`, `qsvc_noise.png` |

## References

- Wang, Du, Luo & Tao, "Towards understanding the power of quantum kernels in the NISQ era", *Quantum* 5, 531 (2021).
- Agliardi et al., arXiv:2412.07915 (2024), on kernel concentration with real-world data.
