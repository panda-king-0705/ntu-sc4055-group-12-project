"""Noise analysis of the QSVC (Person 4).

    python scripts/noise_qsvc.py           # full run
    python scripts/noise_qsvc.py --quick   # smoke test, writes to results/noise/quick/

A QSVC needs one circuit per *pair* of rows: U(x) then U(x')^dagger, and the kernel value is the
probability of measuring |0000>. On hardware every kernel entry is noisy, the training kernel too,
so the SVM is refit on the noisy kernel each time. C is chosen on the noisy validation kernel (what
you would do on hardware); the fixedC_ columns keep the noiseless C instead.

Kernels: the feature map + angle scale with the best validation score in Person 3's grid
(results/qsvc/e1_grid.csv, same rule as run_qsvc.py E2), once without and once with entangling gates.
That gives Z_r2, alpha=0.3 (Person 3's tuned QSVC, no CNOTs) and ZZ_full_r1, alpha=0.05 (24 CNOTs).

Test rows: every 5th row of the 20 test engines (859 rows), because the full test kernel is 1.4M
circuits, about 2.3 h per FakeManila run. The noiseless baseline is scored on the same rows.

    Q1  depolarizing noise, exact kernel values (no shots, no readout error)
    Q2  depolarizing noise + 1,000 shots, sampled from the Q1 kernels (qsvc_lib.sample_kernel)
    Q3  simulated IBM hardware (FakeManilaV2, transpiled), 1,000 shots, one run per kernel

Depolarizing noise and the breaking point are defined as in noise_vqc.py (breaking point on Q2), but
with fewer levels: every level costs ~5 min (no entanglement) and ~10 min (entangled) of simulation.
Outputs -> results/noise/ (qsvc_*.csv, qsvc_summary.json) and results/noise/figures/.
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from common import RESULTS_DIR, evaluate, load_data, load_units, save_json, train_val_split
from noise_vqc import COLORS, LABELS, OUT_DIR, SHOTS, breaking_points, depolarizing, style
from qsvc_lib import (N_QUBITS, engine_bootstrap, fit_precomputed, kernel_blocks, kernel_target_alignment,
                      make_feature_map, offdiag_stats, sample_kernel, score_test, tune_C)
from run_qsvc import fmt_table, log, prefixed, rbf_svm, write_csv

P2_LEVELS = [0, 0.001, 0.01, 0.02, 0.05, 0.1]

# |0000><0000| = prod_q (I + Z_q) / 2, written out as 16 Pauli terms
ZERO = SparsePauliOp.from_sparse_list(
    [("Z" * len(qs), qs, 1 / 2**N_QUBITS)
     for qs in ([q for q in range(N_QUBITS) if m >> q & 1] for m in range(2**N_QUBITS))], N_QUBITS)


def pick_kernels():
    """Best (feature map, alpha) by validation in Person 3's grid, without and with entangling gates."""
    rows = list(csv.DictReader(open(RESULTS_DIR / "qsvc" / "e1_grid.csv")))
    key = lambda r: ((round(float(r["val_accuracy"]), 6), round(float(r["val_recall"]), 6)), float(r["kta_train"]))
    picks = {}
    for name, flag in [("no_entanglement", "False"), ("entangled", "True")]:
        best = max((r for r in rows if r["entangling"] == flag), key=key)
        picks[name] = (best["feature_map"], float(best["alpha"]))
    return picks


def kernel_circuit(fm):
    """Compute-uncompute circuit U(x) U(x')^dagger: P(|0000>) is the kernel value."""
    a, b = ParameterVector("a", N_QUBITS), ParameterVector("b", N_QUBITS)
    return fm.assign_parameters(a).compose(fm.assign_parameters(b).inverse()), a, b


def pairs(X_tr, X_va, X_te):
    """(x, x') for every kernel entry needed: train upper triangle, then val x train, then test x train."""
    iu = np.triu_indices(len(X_tr), 1)
    A = np.vstack([X_tr[iu[0]], np.repeat(X_va, len(X_tr), 0), np.repeat(X_te, len(X_tr), 0)])
    B = np.vstack([X_tr[iu[1]], np.tile(X_tr, (len(X_va), 1)), np.tile(X_tr, (len(X_te), 1))])
    return A, B


def blocks(v, n_tr, n_va):
    """Kernel values in pairs() order -> {"tr", "va", "te"}, like qsvc_lib.kernel_blocks."""
    iu = np.triu_indices(n_tr, 1)
    K = np.zeros((n_tr, n_tr))
    K[iu] = v[: len(iu[0])]
    K = K + K.T
    np.fill_diagonal(K, 1.0)  # k(x, x) = 1 needs no circuit, as in sample_kernel
    w, V = np.linalg.eigh(K)
    K = (V * np.clip(w, 0, None)) @ V.T  # PSD projection, as qiskit does
    rest = v[len(iu[0]):].reshape(-1, n_tr)
    return {"tr": K, "va": rest[:n_va], "te": rest[n_va:]}


def exact_values(qc, a, b, A, B, noise, chunk=5_000):  # Aer slows down superlinearly on big noisy batches
    est = EstimatorV2(options={"default_precision": 0.0, "backend_options": {"noise_model": noise}})
    return np.concatenate([est.run([(qc, ZERO, {tuple(a): A[s:s + chunk], tuple(b): B[s:s + chunk]})]).result()[0].data.evs
                           for s in range(0, len(A), chunk)])


def manila_values(tq, a, b, A, B, chunk=20_000):
    backend = AerSimulator.from_backend(FakeManilaV2())
    out = []
    for s in range(0, len(A), chunk):
        sampler = SamplerV2.from_backend(backend, default_shots=SHOTS, seed=s)
        bits = sampler.run([(tq, {tuple(a): A[s:s + chunk], tuple(b): B[s:s + chunk]})]).result()[0].data.meas.array
        out.append((bits[..., 0] == 0).mean(axis=1))  # share of shots that read 0000
        log(f"   FakeManila {min(s + chunk, len(A)):,} / {len(A):,} kernel entries")
    return np.concatenate(out)


def fit_score(K, C_fixed, exact, y):
    """Refit the SVM on kernel K. Returns metrics, kernel diagnostics and the test predictions."""
    y_tr, y_va, y_te = y
    C, _, model = tune_C(K, y_tr, y_va)
    t, pred, _ = score_test(model, K["te"], y_te)
    fixed = evaluate(y_te, fit_precomputed(K["tr"], y_tr, C_fixed).predict(K["te"]))
    row = {"C": C, **prefixed("test", t), "fixedC_test_f1": fixed["f1"], "fixedC_test_fn": fixed["fn"],
           **offdiag_stats(K["tr"]), "kta_train": kernel_target_alignment(K["tr"], y_tr),
           "kernel_rmse_test": float(np.sqrt(np.mean((K["te"] - exact["te"]) ** 2)))}
    return row, pred


def mean_std(rows, key, col):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r[col])
    return list(groups), [np.mean(v) for v in groups.values()], [np.std(v, ddof=1) if len(v) > 1 else 0 for v in groups.values()]


def plot(q1, q2, q3, ref, n_fail, path):
    panels = [("test_f1", "Test F1"), ("test_fn", f"Missed failures (of {n_fail})"),
              ("k_std", "Spread of training kernel values (std)"), ("kta_train", "Kernel-target alignment (train)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for name in COLORS:
        r1 = [r for r in q1 if r["kernel"] == name]
        r2 = [r for r in q2 if r["kernel"] == name]
        r3 = next(r for r in q3 if r["kernel"] == name)
        for ax, (col, _) in zip(axes.flat, panels):
            ax.plot([r["p_cx"] for r in r1], [r[col] for r in r1], "o-", lw=2, color=COLORS[name],
                    label=f"{LABELS[name]} ({r1[0]['feature_map']}), exact")
            x, m, sd = mean_std(r2, "p_cx", col)
            ax.errorbar(x, m, yerr=sd, fmt="s--", ms=4, capsize=3, color=COLORS[name], alpha=0.6,
                        label=f"{LABELS[name]}, {SHOTS} shots")
            ax.axhline(r3[col], color=COLORS[name], ls=":", lw=1.2, label=f"{LABELS[name]}, FakeManila")
        for ax, col in zip(axes[0], ["fixedC_test_f1", "fixedC_test_fn"]):
            ax.plot([r["p_cx"] for r in r1], [r[col] for r in r1], "x", color=COLORS[name], ms=7)
    axes[0, 0].axhline(ref["f1"], color="#52514e", ls="-.", lw=1, label="Classical RBF SVM")
    axes[0, 1].axhline(ref["fn"], color="#52514e", ls="-.", lw=1)
    for ax, (_, title) in zip(axes.flat, panels):
        style(ax, title)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(left=-1e-4)
    for ax in axes[1]:
        ax.set_xlabel("CNOT depolarizing error (1-qubit gates: a tenth of it)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("QSVC under noise (× = noiseless C kept; dotted = FakeManila, 1,000 shots)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small train/test subsets, two noise levels")
    parser.add_argument("--out", default=None, help="output folder (default results/noise/)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else OUT_DIR / "quick" if args.quick else OUT_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)
    X_full, X_te, y_full, y_te = load_data(quantum=True)
    X_tr, X_va, y_tr, y_va = train_val_split(X_full, y_full)
    units_te = load_units()[1]
    step, levels, reps = (25, [0, 0.05], 1) if args.quick else (5, P2_LEVELS, 3)
    if args.quick:
        X_tr, y_tr = X_tr[::4], y_tr[::4]
    X_te, y_te, units_te = X_te[::step], y_te[::step], units_te[::step]
    y = (y_tr, y_va, y_te)

    params, _, svm = rbf_svm(X_tr, X_va, y_tr, y_va)
    ref = evaluate(y_te, svm.predict(X_te))
    log(f"train {len(X_tr)} | val {len(X_va)} | test {len(X_te)} rows ({int(y_te.sum())} failing, every {step}th) | "
        f"classical RBF SVM {params}: F1 {ref['f1']:.3f}, missed {ref['fn']}")

    rng = np.random.default_rng(0)
    backend = AerSimulator.from_backend(FakeManilaV2())
    q1, q2, q3 = [], [], []
    summary = {"test_rows": len(X_te), "test_every_nth_row": step, "shots": SHOTS,
               "classical_reference": {"params": params, "f1": ref["f1"], "fn": ref["fn"]}}
    for name, (fm_name, alpha) in pick_kernels().items():
        fm = make_feature_map(fm_name)
        qc, a, b = kernel_circuit(fm)
        A, B = (alpha * M for M in pairs(X_tr, X_va, X_te))
        exact = kernel_blocks(fm, X_tr, X_va, X_te, alpha)
        C0, _, _ = tune_C(exact, y_tr, y_va)
        base = {"kernel": name, "feature_map": fm_name, "alpha": alpha}
        log(f"{name}: {fm_name}, alpha={alpha}, noiseless C={C0}, {len(A):,} kernel entries, circuit {dict(qc.count_ops())}")

        for p2 in levels:
            start = time.time()
            K = exact if p2 == 0 else blocks(exact_values(qc, a, b, A, B, depolarizing(p2)), len(X_tr), len(X_va))
            row, pred = fit_score(K, C0, exact, y)
            q1.append({**base, "p_cx": p2, **row})
            if p2 == 0:
                pred0 = pred
            for rep in range(reps):
                Ks = {"tr": sample_kernel(K["tr"], SHOTS, rng, symmetric=True),
                      "va": sample_kernel(K["va"], SHOTS, rng, symmetric=False),
                      "te": sample_kernel(K["te"], SHOTS, rng, symmetric=False)}
                q2.append({**base, "p_cx": p2, "rep": rep, **fit_score(Ks, C0, exact, y)[0]})
            log(f"   p_cx {p2}: F1 {row['test_f1']:.3f}, missed {row['test_fn']}, k_std {row['k_std']:.3f} "
                f"({time.time() - start:.0f}s)")

        tq = transpile(qc.measure_all(inplace=False), backend, optimization_level=1, seed_transpiler=0)
        K = blocks(manila_values(tq, a, b, A, B), len(X_tr), len(X_va))
        row, pred = fit_score(K, C0, exact, y)
        ops = dict(tq.count_ops())
        q3.append({**base, "cx_after_transpile": ops.get("cx", 0), **row})
        ci, diff = engine_bootstrap(y_te, {"noiseless": pred0, "FakeManila": pred}, units_te, "f1")
        summary[name] = {"feature_map": fm_name, "alpha": alpha, "noiseless_C": C0,
                         "transpiled_for_manila": {"ops": ops, "physical_qubits": tq.layout.final_index_layout()},
                         "fakemanila_f1_ci95": ci["FakeManila"],
                         "noiseless_minus_fakemanila_f1_ci95": diff["noiseless - FakeManila"]}
        log(f"   FakeManila: F1 {row['test_f1']:.3f}, missed {row['test_fn']}, k_std {row['k_std']:.3f}")

    for rows, file in [(q1, "qsvc_q1_depolarizing.csv"), (q2, "qsvc_q2_depolarizing_shots.csv"), (q3, "qsvc_q3_fakemanila.csv")]:
        write_csv(rows, out / file)
    for name in COLORS:
        x, f1, _ = mean_std([r for r in q2 if r["kernel"] == name], "p_cx", "test_f1")
        _, fn, _ = mean_std([r for r in q2 if r["kernel"] == name], "p_cx", "test_fn")
        fn0 = next(r["test_fn"] for r in q1 if r["kernel"] == name and r["p_cx"] == 0)
        summary[name]["breaking_point_cx_error"] = breaking_points(x, f1, fn, ref["f1"], fn0)
    save_json(summary, out / "qsvc_summary.json")

    cols = ["kernel", "p_cx", "C", "test_f1", "test_fn", "test_fp", "fixedC_test_f1", "k_mean", "k_std", "kta_train"]
    log("\nQ1 depolarizing, exact kernels:\n" + fmt_table(q1, cols))
    log("\nQ3 FakeManila:\n" + fmt_table(q3, ["kernel", "cx_after_transpile"] + cols[2:]))
    log("Breaking points and bootstrap: " + str({k: summary[k] for k in COLORS}))
    plot(q1, q2, q3, ref, int(y_te.sum()), out / "figures" / "qsvc_noise.png")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
