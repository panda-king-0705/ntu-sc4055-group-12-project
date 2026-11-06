# QSVC results (Person 3)

Quantum Support Vector Classifier on the shared 4-feature, angle-encoded C-MAPSS data
(`data/cmapss_qml_4f_500.npz`, 400 train / 100 test, balanced).
Reproduce with `python scripts/run_qsvc.py` (~20 min) or `--fast` (~3 min, same numbers).

## Method in one paragraph

A QSVC keeps the classical SVM optimiser and replaces the kernel by a state overlap,
$k(x,x') = |\langle 0|U^\dagger(x')U(x)|0\rangle|^2$, where $U$ is a feature-map circuit.
On hardware each entry is estimated with the *compute–uncompute* circuit $U^\dagger(x')U(x)$:
the probability of measuring $|0000\rangle$ equals the kernel value. We use Qiskit's
`ZZFeatureMap` (H layer, $P(2x_i)$ single-qubit phases, and $\text{CX}\,P(2(\pi-x_i)(\pi-x_j))\,\text{CX}$
entangling blocks) on 4 qubits. The headline kernel is computed with
`FidelityQuantumKernel` (sampler-based compute–uncompute); sweeps use
`FidelityStatevectorKernel`, which agrees to $4\times10^{-12}$ (E0). The SVM regulariser $C$
is always picked by 5-fold CV on the **training set only**; the feature map for the headline
model (Qiskit default: reps=2, full entanglement) was fixed before seeing any results.

## E1 — Headline comparison (test set, n=100)

| Model | CV acc | Test acc [95% bootstrap CI] | F1 | ROC-AUC | # support vectors |
|---|---|---|---|---|---|
| QSVC, ZZFeatureMap (reps=2, full) | 0.720 | **0.76** [0.68, 0.84] | 0.789 | 0.866 | 371 / 400 |
| Classical SVM, RBF | 0.960 | **0.97** [0.93, 1.00] | 0.971 | 0.994 | 81 |
| Classical SVM, linear | 0.957 | **0.96** [0.92, 0.99] | 0.961 | 0.994 | 91 |

Exact McNemar test, QSVC vs RBF-SVM: the QSVC is never right where the RBF-SVM is wrong; the reverse happens 21 times (p ≈ 1e-6).
**The default quantum kernel is significantly worse.** It uses 371/400 points as support vectors, so it is close to memorising the training data.

Figures: `figures/e1_kernel_heatmaps.png` and `figures/e1_confusion_qsvc.png`.

## E2 — Feature-map ablation: does entanglement help?

| Feature map | Entangling | CX gates | Depth | CV acc | Test acc | Kernel-target alignment | Mean off-diagonal k |
|---|---|---|---|---|---|---|---|
| Z, reps 1 | no | 0 | 2 | **0.950** | **0.97** | **0.371** | 0.138 |
| Z, reps 2 | no | 0 | 4 | 0.850 | 0.94 | 0.125 | 0.269 |
| ZZ linear, reps 1 | yes | 6 | 11 | 0.880 | 0.90 | 0.146 | 0.076 |
| ZZ linear, reps 2 | yes | 12 | 19 | 0.828 | 0.86 | 0.107 | 0.085 |
| ZZ full, reps 1 | yes | 12 | 17 | 0.852 | 0.95 | 0.119 | 0.084 |
| ZZ full, reps 2 | yes | 24 | 31 | 0.720 | 0.76 | 0.068 | 0.077 |
| ZZ full, reps 3 | yes | 36 | 45 | 0.633 | 0.63 | 0.059 | 0.078 |

On this dataset, **adding entanglement and repetitions consistently makes the kernel worse**.
The best map is the cheapest one, the product-state `Z` map, which needs no CX gates at all.
For the deep entangled maps, the mean off-diagonal kernel value (≈0.077) is close to 1/2⁴ = 0.0625,
which is the value expected for *Haar-random* states. The feature map scrambles the data so much
that every pair of engines looks about equally dissimilar, and the SVM has nothing left to generalise from.
This matches Person 2's VQC entanglement ablation and is a good point to cross-reference.

Figure: `figures/e2_feature_maps.png`.

## E3 — Kernel bandwidth: the fix

We scale every encoded angle by a factor α (Shaydulin & Wild, 2022, *"Importance of kernel bandwidth in quantum machine learning"*).

| ZZ full, reps 2 | α=1 (default) | α=0.5 | α=0.2 | **α=0.1** (best by CV) | α=0.05 |
|---|---|---|---|---|---|
| CV acc | 0.720 | 0.885 | 0.938 | **0.958** | 0.952 |
| Test acc | 0.76 | 0.93 | 0.95 | **0.97** | 0.97 |
| Kernel-target alignment | 0.068 | 0.118 | 0.207 | 0.332 | 0.428 |

The same circuit goes from 76% to 97% test accuracy. α was selected by training-set CV, not by test accuracy.
There is a sweet spot: if α is too large, the kernel looks random (≈ Haar); if α is too small, the kernel
approaches 1 everywhere and all points look the same (see the `Z_r2` curve at α=0.05, where the mean off-diagonal value is 0.99).

**Caveat:** at small α, every rotation angle is small, so the kernel is close to a smooth, low-order function of $x - x'$, much like a classical RBF kernel.
Reaching classical accuracy this way is evidence that the quantum kernel *can be tuned to behave like a good classical kernel*.
It is **not** evidence of a quantum advantage.

Figure: `figures/e3_bandwidth.png`.

## E4 — Learning curve (10 random balanced subsets per size)

| Train size | QSVC default | QSVC α=0.1 | Classical RBF SVM |
|---|---|---|---|
| 20 | 0.59 ± 0.05 | 0.88 ± 0.11 | 0.94 ± 0.04 |
| 40 | 0.61 ± 0.05 | 0.95 ± 0.02 | 0.94 ± 0.06 |
| 80 | 0.68 ± 0.02 | 0.96 ± 0.01 | 0.96 ± 0.01 |
| 160 | 0.74 ± 0.04 | 0.97 ± 0.01 | 0.97 ± 0.01 |
| 400 | 0.76 | 0.97 | 0.97 |

The tuned QSVC matches the classical SVM from about 40 samples onward. It shows no sample-efficiency advantage.
The default QSVC is still improving at 400 samples, which is consistent with a kernel that has to "see" almost every test point's neighbourhood.

Figure: `figures/e4_learning_curve.png`.

## E5 — Finite-shot kernel estimation (tuned kernel, α=0.1)

On real hardware each kernel entry is a sampled probability, so its error is ∝ 1/√shots.

| Shots per entry | 16 | 64 | 256 | 1024 | 4096 | exact |
|---|---|---|---|---|---|---|
| Test acc (mean ± std, 10 repeats) | 0.907 ± 0.027 | 0.954 ± 0.017 | 0.963 ± 0.009 | 0.968 ± 0.004 | 0.970 ± 0.005 | 0.970 |
| Kernel RMSE | 0.081 | 0.041 | 0.021 | 0.011 | 0.005 | 0 |

About 1k shots per entry is enough here. But the full model needs $\binom{400}{2} + 100\cdot400 = 119{,}800$
kernel circuits, so 1k shots each is about 1.2×10⁸ circuit executions. This cost grows as $O(n^2)$ in the dataset size,
and it is the main practical bottleneck for QSVC compared with a VQC, whose inference cost is $O(1)$ circuits per test point.
Shot noise is only one source of error; Person 4's hardware-noise study covers gate and readout noise.

Figure: `figures/e5_shots.png`.

## Takeaways for the report / video

1. **The QSVC is correct but not better.** With a well-chosen bandwidth it reaches classical accuracy (97%); with Qiskit's default settings it is significantly worse (76%).
2. **More quantum is not better here.** More entanglement, more repetitions and larger angles all push the kernel toward "random-state" behaviour and hurt generalisation.
3. **Bandwidth is the most important hyperparameter**, more important than the circuit architecture.
4. **The cost scales as O(n²) kernel circuits**, each needing roughly 10³ shots. That is the main obstacle to running a QSVC at scale on hardware.

## Limitations

- There is one fixed train/test split (Person 1's seed 42) with 100 test points, so the CIs are about ±8%. We report bootstrap CIs and McNemar tests rather than relying on point estimates.
- The 4 PCA features make this problem easy for classical models, which leaves little room for any quantum kernel to beat them.
- Shot noise is simulated as binomial sampling of the exact kernel. Gate and readout noise are in Person 4's experiments.
