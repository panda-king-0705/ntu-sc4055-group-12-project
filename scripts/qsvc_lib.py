"""
Quantum Support Vector Classifier (QSVC) helpers for the C-MAPSS project.

A QSVC is a classical SVM whose kernel is a quantum state overlap:

    k(x, x') = |<phi(x) | phi(x')>|^2,   |phi(x)> = U_phi(x) |0...0>

where U_phi is a data-encoding circuit (the "feature map"). The quantum
computer only evaluates the Gram matrix; the SVM dual problem is solved
classically. We therefore compute the kernel once and feed it to
sklearn's SVC(kernel="precomputed"), which is exactly what
qiskit_machine_learning.algorithms.QSVC does internally.

Two kernel back-ends are used:
  * FidelityQuantumKernel   - the "real" circuit-based kernel (ComputeUncompute
                               fidelity via a Sampler). Used for the headline model.
  * FidelityStatevectorKernel - computes the same overlap directly from
                               statevectors. Mathematically identical in the noiseless
                               case (verified numerically in run_qsvc.py) and ~500x
                               faster, so it is used for the ablation sweeps. It also
                               supports finite-shot sampling via `shots=`.
"""
from __future__ import annotations

import warnings

import numpy as np
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    FidelityStatevectorKernel,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=DeprecationWarning)

N_QUBITS = 4
C_GRID = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path: str = "data/cmapss_qml_4f_500.npz"):
    """Load the angle-encoded ([0, pi]) PCA features shared by all quantum models."""
    d = np.load(path)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------
# name -> (builder, kwargs). "Z" maps have no entangling gates (product states),
# so their kernel factorises into a product of per-qubit kernels; this is the
# QSVC analogue of Person 2's "VQC without CNOTs" ablation.
FEATURE_MAPS = {
    "Z_r1": (z_feature_map, dict(reps=1)),
    "Z_r2": (z_feature_map, dict(reps=2)),
    "ZZ_linear_r1": (zz_feature_map, dict(reps=1, entanglement="linear")),
    "ZZ_linear_r2": (zz_feature_map, dict(reps=2, entanglement="linear")),
    "ZZ_full_r1": (zz_feature_map, dict(reps=1, entanglement="full")),
    "ZZ_full_r2": (zz_feature_map, dict(reps=2, entanglement="full")),
    "ZZ_full_r3": (zz_feature_map, dict(reps=3, entanglement="full")),
}


def make_feature_map(name: str, n_qubits: int = N_QUBITS):
    builder, kw = FEATURE_MAPS[name]
    return builder(n_qubits, **kw)


def circuit_stats(fm) -> dict:
    """Depth and two-qubit gate count after decomposing to basic gates."""
    from qiskit import transpile

    t = transpile(fm, basis_gates=["rz", "sx", "x", "cx", "h", "p"], optimization_level=0)
    ops = t.count_ops()
    return {"depth": t.depth(), "cx": int(ops.get("cx", 0)), "n_params": fm.num_parameters}


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def statevector_kernel(fm, shots: int | None = None, seed: int | None = None):
    k = FidelityStatevectorKernel(feature_map=fm, shots=shots, enforce_psd=True)
    if seed is not None:
        # shot sampling draws from qiskit-machine-learning's global RNG
        from qiskit_machine_learning.utils import algorithm_globals

        algorithm_globals.random_seed = seed
    return k


def circuit_kernel(fm):
    return FidelityQuantumKernel(feature_map=fm, enforce_psd=True)


def gram_matrices(kernel, X_train, X_test):
    """Return (K_train [n_tr x n_tr], K_test [n_te x n_tr])."""
    K_tr = kernel.evaluate(X_train)
    K_te = kernel.evaluate(X_test, X_train)
    return K_tr, K_te


def sample_kernel(K_exact, shots: int, rng, symmetric: bool):
    """Finite-shot estimate of a fidelity kernel.

    With the compute-uncompute circuit, each entry is estimated as the fraction of
    `shots` that return |0...0>, i.e. Binomial(shots, K_ij) / shots. This is the
    same noise model as FidelityStatevectorKernel(shots=...), but vectorised
    (qiskit's version loops over entries in Python and is ~1000x slower).
    Symmetric (train) matrices: sample i<j only, mirror, set diag=1 (no circuit
    needed for k(x,x)), then project to the PSD cone as qiskit does.
    """
    S = rng.binomial(shots, np.clip(K_exact, 0, 1)) / shots
    if symmetric:
        S = np.triu(S, 1)
        S = S + S.T
        np.fill_diagonal(S, 1.0)
        w, V = np.linalg.eigh(S)
        S = (V * np.clip(w, 0, None)) @ V.T
    return S


def rbf_gram(X_a, X_b, gamma):
    d2 = (X_a**2).sum(1)[:, None] + (X_b**2).sum(1)[None, :] - 2 * X_a @ X_b.T
    return np.exp(-gamma * np.maximum(d2, 0))


# ---------------------------------------------------------------------------
# Kernel diagnostics
# ---------------------------------------------------------------------------
def kernel_target_alignment(K, y) -> float:
    """Centred kernel-target alignment (Cortes et al., 2012). 1 = perfect, 0 = none."""
    y_pm = np.where(y == 1, 1.0, -1.0)
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Y = np.outer(y_pm, y_pm)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def offdiag_stats(K) -> dict:
    """Mean / std of off-diagonal kernel entries. Std -> 0 signals exponential
    concentration: every pair of points looks equally (dis)similar."""
    mask = ~np.eye(K.shape[0], dtype=bool)
    v = K[mask]
    return {"k_mean": float(v.mean()), "k_std": float(v.std())}


# ---------------------------------------------------------------------------
# Training / evaluation on a precomputed kernel
# ---------------------------------------------------------------------------
def select_C(K_tr, y_tr, c_grid=C_GRID, n_splits=5, seed=0):
    """Pick SVM regularisation C by stratified k-fold CV on the training kernel only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c, best_acc = None, -1.0
    for c in c_grid:
        accs = []
        for tr, va in skf.split(K_tr, y_tr):
            m = SVC(kernel="precomputed", C=c).fit(K_tr[np.ix_(tr, tr)], y_tr[tr])
            accs.append(accuracy_score(y_tr[va], m.predict(K_tr[np.ix_(va, tr)])))
        if np.mean(accs) > best_acc:
            best_c, best_acc = c, float(np.mean(accs))
    return best_c, best_acc


def fit_eval(K_tr, K_te, y_tr, y_te, C=None, seed=0) -> dict:
    """CV-select C (if not given), fit on full train kernel, score on test."""
    cv_acc = None
    if C is None:
        C, cv_acc = select_C(K_tr, y_tr, seed=seed)
    m = SVC(kernel="precomputed", C=C).fit(K_tr, y_tr)
    pred = m.predict(K_te)
    score = m.decision_function(K_te)
    return {
        "C": C,
        "cv_acc": cv_acc,
        "test_acc": accuracy_score(y_te, pred),
        "test_f1": f1_score(y_te, pred),
        "test_auc": roc_auc_score(y_te, score),
        "n_support": int(m.n_support_.sum()),
        "pred": pred,
        "score": score,
    }


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for test accuracy (test set is only 100 samples)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = np.mean(y_true[idx] == y_pred[idx])
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: do two classifiers' error patterns differ on the same test set?"""
    from scipy.stats import binomtest

    a_right = pred_a == y_true
    b_right = pred_b == y_true
    n01 = int(np.sum(a_right & ~b_right))  # A right, B wrong
    n10 = int(np.sum(~a_right & b_right))  # A wrong, B right
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return {"a_only_right": n01, "b_only_right": n10, "p_value": float(p)}


def balanced_subset(y, n, rng):
    """Indices of a class-balanced random subset of size n."""
    pos = rng.choice(np.where(y == 1)[0], n // 2, replace=False)
    neg = rng.choice(np.where(y == 0)[0], n // 2, replace=False)
    return np.concatenate([pos, neg])
