# QSVC results (Person 3)

Quantum Support Vector Classifier on the shared data, following the group protocol in
[`docs/vqc_handoff.md`](../../docs/vqc_handoff.md):
- `load_data(quantum=True)`: the 4 PCA features scaled to [0, π].
- `train_val_split`: 319 train / 81 validation rows, split by engine.
- Every choice (C, feature map, angle scale α) is made on **validation**.
- The test set (4,291 rows from 20 held-out engines, 14.4% failing) is only used for reporting, scored with `evaluate()`.

Reproduce with `python scripts/run_qsvc.py` (about 10 min). It writes everything in this folder.

## Method

A QSVC keeps the classical SVM optimiser and replaces its kernel with a quantum state overlap:

$$k(x,x') = |\langle 0|U^\dagger(x')\,U(x)|0\rangle|^2$$

where $U$ is a feature-map circuit. On hardware, each entry is estimated with the *compute–uncompute* circuit $U^\dagger(x')U(x)$: the probability of measuring $|0000\rangle$ is the kernel value. The SVM is then trained on the precomputed Gram matrix. This is exactly what `qiskit_machine_learning.algorithms.QSVC` does, and we checked the two give identical predictions (E0).

Kernels are computed with `FidelityStatevectorKernel` (exact, noiseless). It agrees with the circuit-based `FidelityQuantumKernel` to within 4×10⁻¹², but it takes seconds rather than about 1.5 h on the 4,291×319 test kernel. Finite-shot sampling is studied separately in E4.

**Feature maps** (all 4 qubits, one feature per qubit):

| Name | Circuit | CX gates |
|---|---|---|
| RY (VQC encoding) | RY(x_i) on each qubit, the same encoding as Person 2's VQC | 0 |
| Z_r1, Z_r2 | Qiskit `z_feature_map`: [H, P(2x_i)] × reps | 0 |
| ZZ_linear_r1/r2 | Qiskit `zz_feature_map`: Z layer + CX·P(2(π−x_i)(π−x_j))·CX on neighbours | 6 / 12 |
| ZZ_full_r1/r2/r3 | the same, on all pairs (**Qiskit's default is ZZ_full_r2**) | 12 / 24 / 36 |

**Angle scale α**: every feature map gets α·x instead of x. A smaller α makes the kernel smoother (a wider bandwidth). See Shaydulin & Wild (2022), *Importance of kernel bandwidth in quantum machine learning*.

The models without CX gates have product-state kernels, for example $\prod_i \cos^2((x_i-x'_i)/2)$ for RY. Those can be computed classically in closed form. Two independent consistency checks fall out of the grid: RY at α equals Z_r1 at α/2, and their rows in `e1_grid.csv` are identical.

## E1: Feature-map × bandwidth grid (64 configurations)

Full table: `e1_grid.csv`. Figures: `figures/e1_feature_maps.png` and `figures/e1_bandwidth.png`.

**At the default angles (α = 1), entanglement hurts:**

| Feature map | CX | Val acc | Test recall | Missed failures (of 620) | False alarms | Test F1 | Mean off-diagonal k |
|---|---|---|---|---|---|---|---|
| RY (VQC encoding) | 0 | 0.951 | 0.973 | 17 | 336 | **0.774** | 0.496 |
| Z_r1 | 0 | 0.951 | 0.969 | 19 | 334 | **0.773** | 0.139 |
| Z_r2 | 0 | 0.877 | 0.816 | 114 | 527 | 0.612 | 0.263 |
| ZZ_linear_r1 | 6 | 0.864 | 0.852 | 92 | 425 | 0.671 | 0.075 |
| ZZ_linear_r2 | 12 | 0.864 | 0.852 | 92 | 640 | 0.591 | 0.084 |
| ZZ_full_r1 | 12 | 0.877 | 0.887 | 70 | 519 | 0.651 | 0.083 |
| **ZZ_full_r2 (Qiskit default)** | 24 | 0.765 | 0.718 | **175** | 935 | **0.445** | 0.077 |
| ZZ_full_r3 | 36 | 0.704 | 0.637 | 225 | 1390 | 0.328 | 0.080 |

For the entangling maps, the mean similarity between two *different* engines' states is about 0.08, close to 1/2⁴ = 0.0625, the value expected for **Haar-random states**. The circuit scrambles the data so thoroughly that every pair of points looks almost equally unrelated, and the SVM has nothing to generalise from. On this data, more entangling layers make it worse every time.

**Shrinking the angles fixes it.** With α = 0.05–0.1, every feature map, including ZZ_full_r3, recovers a test F1 of 0.78–0.82 (right panel of `e1_bandwidth.png`). If α is too small instead, the kernel tends towards 1 for every pair, so the useful region is a band in the middle.

**Caveat on α → 0:** with small angles, every map's kernel is close to a smooth, low-order function of x − x', much like a classical RBF kernel. Recovering classical performance this way shows the quantum kernel can be *tuned to behave like a good classical kernel*. It is **not** a quantum advantage.

## E2: Final comparison on the test set

Rules: the "default" was fixed before seeing results. "Tuned" is the grid point with the best **validation** accuracy, with ties broken by validation recall and then by kernel-target alignment on train. Uncertainty comes from an **engine-level bootstrap**: the 4,291 test rows come from only 20 engines, and rows from one engine are strongly correlated, so we resample engines rather than rows.

| Model | Test recall [95% CI] | Missed failures (of 620) | False alarms (of 3,671) | Test F1 [95% CI] | Misses with ≤10 cycles left |
|---|---|---|---|---|---|
| QSVC default (ZZ_full_r2, α=1) | 0.718 [0.68, 0.76] | 175 | 935 | 0.445 [0.42, 0.47] | **58** |
| QSVC tuned (Z_r2, α=0.3) | 0.966 [0.94, 0.99] | 21 | 316 | 0.780 [0.72, 0.84] | 0 |
| QSVC, VQC encoding (RY, α=1) | 0.973 [0.95, 0.99] | 17 | 336 | 0.774 [0.71, 0.84] | 0 |
| Classical SVM (RBF, same features) | 0.976 [0.96, 0.99] | 15 | 340 | 0.773 [0.71, 0.84] | 0 |
| VQC entangled (Person 2, best seed) | 0.956 [0.92, 0.99] | 27 | 286 | 0.791 [0.74, 0.85] | 0 |
| VQC no entanglement (Person 2, best seed) | 0.934 [0.88, 0.98] | 41 | 256 | 0.796 [0.74, 0.86] | 5 |

Paired engine-bootstrap 95% CIs for the difference in F1:
- Tuned QSVC − RBF SVM: [−0.003, +0.018]
- Tuned QSVC − VQC entangled: [−0.022, +0.001]
- Tuned QSVC − VQC no entanglement: [−0.036, +0.013]
- Default QSVC − tuned QSVC: [−0.376, −0.295]

**So:**
1. The tuned QSVC, the RBF SVM and both VQCs are **statistically indistinguishable**. They also trade recall against false alarms in slightly different ways: the kernel models catch a few more failures, while the VQCs raise fewer false alarms.
2. The default QSVC is clearly worse. It also misses 58 rows with ≤10 cycles left, which are the most dangerous errors. Every tuned kernel model's misses are 22–30 cycles from failure, right at the label boundary (see `figures/e2_predicted_vs_rul.png`).
3. Kernel heatmaps: `figures/e2_kernel_heatmaps.png`. The default kernel is almost diagonal, while the tuned and RBF kernels show clear class blocks.

**The validation set is too small to pick a feature map.** 28 of the 64 configurations tie at the maximum validation accuracy (77/81 rows, recall 38/39), even though their test F1 ranges from 0.76 to 0.82. The protocol therefore can't separate the good configurations; it can only reject the bad ones. In particular, we **did not** report the best-on-test grid point (ZZ_linear_r2 at α=0.05, F1 0.820), because choosing it would mean selecting on the test set.

## E3: Learning curve (`figures/e3_learning_curve.png`)

Balanced random subsets of the 319 training rows (10 subsets per size). C and γ are re-tuned on validation each time, and results are scored on the full test set.

| Train rows | QSVC default F1 | QSVC tuned F1 | RBF SVM F1 |
|---|---|---|---|
| 20 | 0.28 ± 0.02 | 0.77 ± 0.04 | 0.78 ± 0.04 |
| 40 | 0.32 ± 0.03 | 0.77 ± 0.02 | 0.79 ± 0.03 |
| 80 | 0.35 ± 0.03 | 0.77 ± 0.02 | 0.77 ± 0.03 |
| 160 | 0.40 ± 0.03 | 0.79 ± 0.02 | 0.80 ± 0.02 |
| 319 | 0.45 | 0.78 | 0.77 |

The tuned QSVC and the RBF SVM are already near their ceiling with 20 rows, so there's **no sample-efficiency advantage**. The default QSVC is still improving at 319 rows, which is consistent with a near-diagonal kernel that has to see almost every neighbourhood to make a prediction.

## E4: Finite measurement shots (`figures/e4_shots.png`)

On hardware, each kernel entry is a sampled probability, Binomial(shots, k)/shots, so its error is ∝ 1/√shots.

| Shots per entry | 16 | 64 | 256 | 1024 | 4096 | exact |
|---|---|---|---|---|---|---|
| Test F1 (10 repeats) | 0.771 ± 0.007 | 0.782 ± 0.004 | 0.783 ± 0.003 | 0.782 ± 0.002 | 0.782 ± 0.001 | 0.780 |
| Missed failures | 18.8 ± 4.4 | 18.5 ± 2.1 | 19.6 ± 1.0 | 20.6 ± 0.5 | 20.5 ± 1.0 | 21 |
| Kernel RMSE | 0.079 | 0.040 | 0.020 | 0.010 | 0.005 | 0 |

The tuned kernel is very robust to shot noise: even 16–64 shots barely change the result. The real cost is the **number of circuits**. Training needs 319·318/2 = 50,721 kernel circuits, and scoring the test set needs 4,291 × 319 = 1,368,829 more: one per (test row, training row) pair. So prediction costs O(n_train) circuits per new engine reading. For comparison, the VQC needs one circuit per prediction. That is the main practical drawback of QSVC on hardware. Gate and readout noise are covered by Person 4.

## Takeaways for the report and video

1. **A QSVC can match the classical and VQC models, but it doesn't beat them.** Tuned, it reaches F1 ≈ 0.78 with 21 missed failures, statistically tied with an RBF SVM and both VQCs.
2. **Qiskit's default ZZFeatureMap is a bad choice for this data:** F1 0.45, with 175 missed failures, 58 of them within 10 cycles of failure.
3. **More quantum is not better here.** Entanglement, extra repetitions and large angles push the kernel towards "random-state" behaviour (mean overlap ≈ 1/2ⁿ). The kernel bandwidth matters more than the circuit's architecture.
4. **The cost is in the number of circuits, not in shots:** about 1.4 million kernel circuits for this test set, compared with one circuit per prediction for the VQC.

## Limitations

- Validation has only 81 rows (4 errors at best), so it can't rank the good configurations (see E2).
- The test set has only 20 engines. CIs use an engine-level bootstrap, and they are wide (F1 roughly ±0.06).
- The standardisation, PCA and [0, π] scaling are fitted on all 400 training rows, including the 81 validation rows. This doesn't touch the test set, but it makes validation slightly optimistic. It's a group-wide choice and isn't changed here.
- The kernels are exact or shot-sampled only. Gate and readout noise are in Person 4's study.
- The features are 4 PCA components of an easy problem, which leaves little room for any kernel to beat a classical RBF SVM.
