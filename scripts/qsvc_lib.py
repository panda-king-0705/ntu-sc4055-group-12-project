"""
Quantum Support Vector Classifier (QSVC) helpers (Person 3).

A QSVC is a classical SVM whose kernel is a quantum state overlap:

    k(x, x') = |<phi(x) | phi(x')>|^2,   |phi(x)> = U_phi(x) |0...0>

where U_phi is a data-encoding circuit (the "feature map"). The quantum computer
only estimates the Gram matrix; the SVM dual problem is solved classically. We
compute the kernel once and feed it to sklearn's SVC(kernel="precomputed"),
which is exactly what qiskit_machine_learning.algorithms.QSVC does internally
(checked in run_qsvc.py, E0).

Kernel back-ends:
  * FidelityStatevectorKernel - overlap computed from statevectors. Used for all
    experiments (group decision, docs/vqc_handoff.md): the 4,291 x 319 test kernel
    takes seconds instead of ~1.5 h.
  * FidelityQuantumKernel     - the circuit-based compute-uncompute kernel that would
    run on hardware. Only used in E0 to show both give the same values.

Data, split and metrics come from scripts/common.py so all models are comparable.
"""
from __future__ import annotations

import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from sklearn.svm import SVC

from common import evaluate

warnings.filterwarnings("ignore", category=DeprecationWarning)

N_QUBITS = 4
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------
def ry_feature_map(n_qubits: int = N_QUBITS) -> QuantumCircuit:
    """The VQC's encoding (one RY(x_i) per qubit, no entanglement).
    Its kernel is prod_i cos^2((x_i - x'_i) / 2): fully classical to compute."""
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="RY")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc


# name -> (builder, kwargs). Maps without entangling gates give product-state
# kernels (a product of per-qubit kernels); this is the QSVC analogue of the
# VQC "no CNOT" ablation.
FEATURE_MAPS = {
    "RY (VQC encoding)": (ry_feature_map, {}),
    "Z_r1": (z_feature_map, dict(reps=1)),
    "Z_r2": (z_feature_map, dict(reps=2)),
    "ZZ_linear_r1": (zz_feature_map, dict(reps=1, entanglement="linear")),
    "ZZ_linear_r2": (zz_feature_map, dict(reps=2, entanglement="linear")),
    "ZZ_full_r1": (zz_feature_map, dict(reps=1, entanglement="full")),
    "ZZ_full_r2": (zz_feature_map, dict(reps=2, entanglement="full")),
    "ZZ_full_r3": (zz_feature_map, dict(reps=3, entanglement="full")),
}
ENTANGLING = {k: k.startswith("ZZ") for k in FEATURE_MAPS}


def make_feature_map(name: str, n_qubits: int = N_QUBITS) -> QuantumCircuit:
    builder, kw = FEATURE_MAPS[name]
    return builder(n_qubits, **kw)


def circuit_stats(fm) -> dict:
    """Depth and CX count after decomposing to basic gates."""
    from qiskit import transpile

    t = transpile(fm, basis_gates=["rz", "sx", "x", "cx", "h", "p", "ry"], optimization_level=0)
    return {"depth": t.depth(), "cx": int(t.count_ops().get("cx", 0))}


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def statevector_kernel(fm):
    return FidelityStatevectorKernel(feature_map=fm, enforce_psd=True)


def circuit_kernel(fm):
    return FidelityQuantumKernel(feature_map=fm, enforce_psd=True)


def kernel_blocks(fm, X_tr, X_va, X_te, alpha: float = 1.0) -> dict:
    """All Gram blocks against the training rows, for one feature map and angle scale.
    Returns K_tr (n_tr x n_tr), K_va (n_va x n_tr), K_te (n_te x n_tr)."""
    k = statevector_kernel(fm)
    a = lambda X: alpha * X
    K_tr = k.evaluate(a(X_tr))
    K_rest = k.evaluate(np.vstack([a(X_va), a(X_te)]), a(X_tr))
    return {"tr": K_tr, "va": K_rest[: len(X_va)], "te": K_rest[len(X_va):]}


def sample_kernel(K_exact, shots: int, rng, symmetric: bool):
    """Finite-shot estimate of a fidelity kernel.

    With the compute-uncompute circuit, each entry is estimated as the fraction of
    `shots` returning |0...0>, i.e. Binomial(shots, K_ij) / shots. This is the noise
    model of FidelityStatevectorKernel(shots=...), vectorised (qiskit loops over
    entries in Python, ~1000x slower). Symmetric (train) matrices: sample i<j only,
    mirror, set diag=1 (k(x,x)=1 needs no circuit), then project onto the PSD cone
    as qiskit does.
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
    """Centred kernel-target alignment (Cortes et al., 2012). Higher = kernel similarity
    agrees better with the labels."""
    y_pm = np.where(y == 1, 1.0, -1.0)
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Y = np.outer(y_pm, y_pm)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def offdiag_stats(K) -> dict:
    """Mean / std of off-diagonal train kernel entries. Mean near 1/2^n (Haar-random
    states) = every pair looks equally dissimilar; mean near 1 = every pair looks the same."""
    v = K[~np.eye(K.shape[0], dtype=bool)]
    return {"k_mean": float(v.mean()), "k_std": float(v.std())}


# ---------------------------------------------------------------------------
# Training / model selection on a precomputed kernel (validation set only)
# ---------------------------------------------------------------------------
def val_key(m: dict):
    """Selection rule shared with the VQC: validation accuracy, ties -> validation recall."""
    return (round(m["accuracy"], 6), round(m["recall"], 6))


def fit_precomputed(K_tr, y_tr, C):
    return SVC(kernel="precomputed", C=C).fit(K_tr, y_tr)


def tune_C(K, y_tr, y_va, c_grid=C_GRID):
    """Pick C on the validation rows. Ties -> smaller C (simpler model)."""
    best = None
    for c in c_grid:
        m = fit_precomputed(K["tr"], y_tr, c)
        v = evaluate(y_va, m.predict(K["va"]))
        if best is None or val_key(v) > val_key(best[1]):
            best = (c, v, m)
    return best  # (C, val_metrics, fitted model)


def score_test(model, K_te, y_te) -> tuple[dict, np.ndarray, np.ndarray]:
    pred = model.predict(K_te)
    return evaluate(y_te, pred), pred, model.decision_function(K_te)


# ---------------------------------------------------------------------------
# Uncertainty: the 4,291 test rows come from only 20 engines, and rows of one
# engine are strongly correlated, so resample *engines*, not rows.
# ---------------------------------------------------------------------------
def engine_bootstrap(y, preds: dict, units, metric="f1", n_boot=2000, seed=0):
    """Engine-level (cluster) bootstrap. Returns per-model 95% CI and, for every
    pair (a, b), the CI of metric(a) - metric(b) on the same resamples."""
    rng = np.random.default_rng(seed)
    engines = np.unique(units)
    rows_of = {e: np.where(units == e)[0] for e in engines}
    names = list(preds)
    vals = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        idx = np.concatenate([rows_of[e] for e in rng.choice(engines, len(engines), replace=True)])
        for n in names:
            vals[n][b] = evaluate(y[idx], preds[n][idx])[metric]
    q = lambda a: (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
    ci = {n: q(vals[n]) for n in names}
    diff = {f"{a} - {b}": q(vals[a] - vals[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return ci, diff


def balanced_subset(y, n, rng):
    """Indices of a class-balanced random subset of size n."""
    pos = rng.choice(np.where(y == 1)[0], n // 2, replace=False)
    neg = rng.choice(np.where(y == 0)[0], n - n // 2, replace=False)
    return np.concatenate([pos, neg])
