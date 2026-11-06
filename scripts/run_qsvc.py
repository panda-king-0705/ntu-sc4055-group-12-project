"""
Person 3 - Quantum Support Vector Classifier (QSVC) experiments.

Run from the repo root:

    python scripts/run_qsvc.py          # full run, headline kernel via FidelityQuantumKernel (~20 min)
    python scripts/run_qsvc.py --fast   # statevector kernel everywhere (~2 min, identical numbers)

Experiments
    E0  Sanity: FidelityQuantumKernel == FidelityStatevectorKernel, and qiskit's QSVC
        class == SVC on the precomputed kernel.
    E1  Headline model: ZZFeatureMap (reps=2, full entanglement; Qiskit's default,
        fixed *before* looking at results) vs classical SVMs on the same 4 features.
        Bootstrap 95% CIs + exact McNemar test.
    E2  Feature-map ablation: no entanglement (Z) vs linear / full ZZ entanglement,
        1-3 repetitions. C chosen by 5-fold CV on train only.
    E3  Kernel bandwidth: scale the encoded angles by alpha. Diagnoses kernel
        concentration (off-diagonal kernel values collapsing).
    E4  Learning curve: train size 20..400, 10 random balanced subsets each.
    E5  Finite-shot kernel estimation: 16..4096 shots vs exact (10 repeats each).

Outputs -> results/qsvc/ (CSV + JSON) and results/qsvc/figures/ (PNG).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from qsvc_lib import (  # noqa: E402
    FEATURE_MAPS,
    balanced_subset,
    bootstrap_ci,
    circuit_kernel,
    circuit_stats,
    fit_eval,
    gram_matrices,
    kernel_target_alignment,
    load_data,
    make_feature_map,
    mcnemar_exact,
    offdiag_stats,
    rbf_gram,
    sample_kernel,
    statevector_kernel,
)

OUT = "results/qsvc"
FIG = os.path.join(OUT, "figures")
MAIN_FM = "ZZ_full_r2"
SEED = 0



# ---------------------------------------------------------------------------
# tiny table helpers (pandas is avoided: its DLLs are blocked on some locked-down Windows machines)
def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt_table(rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else (f"({v[0]:.2f}, {v[1]:.2f})" if isinstance(v, tuple) else str(v))
    cells = [[f(r.get(c, "")) for c in cols] for r in rows]
    w = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    line = lambda xs: "  ".join(x.rjust(w[i]) for i, x in enumerate(xs))
    return "\n".join([line(cols)] + [line(x) for x in cells])


def group_stats(rows, keys, vals):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for gk, rs in groups.items():
        d = dict(zip(keys, gk))
        for v in vals:
            a = np.array([r[v] for r in rs], float)
            d[f"{v}_mean"], d[f"{v}_std"] = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out.append(d)
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classical_svm(X_tr, y_tr, kernel):
    grid = {"C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]}
    if kernel == "rbf":
        grid["gamma"] = [0.01, 0.03, 0.1, 0.3, 1, 3]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(SVC(kernel=kernel), grid, cv=cv).fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ---------------------------------------------------------------------------
def e0_sanity(X_tr, y_tr):
    log("E0 sanity checks")
    from qiskit_machine_learning.algorithms import QSVC
    from sklearn.svm import SVC as _SVC

    fm = make_feature_map(MAIN_FM)
    sub = X_tr[:30]
    K_circ = circuit_kernel(fm).evaluate(sub)
    K_sv = statevector_kernel(fm).evaluate(sub)
    kdiff = float(np.abs(K_circ - K_sv).max())

    ksv = statevector_kernel(fm)
    q = QSVC(quantum_kernel=ksv, C=1.0).fit(X_tr[:80], y_tr[:80])
    s = _SVC(kernel="precomputed", C=1.0).fit(ksv.evaluate(X_tr[:80]), y_tr[:80])
    agree = float(np.mean(q.predict(X_tr[80:160]) == s.predict(ksv.evaluate(X_tr[80:160], X_tr[:80]))))
    res = {"max_abs_kernel_diff_circuit_vs_statevector": kdiff, "QSVC_vs_precomputed_SVC_agreement": agree}
    log(f"   {res}")
    return res


def e1_headline(X_tr, X_te, y_tr, y_te, fast):
    log(f"E1 headline QSVC ({MAIN_FM}) using {'statevector' if fast else 'FidelityQuantumKernel (sampler)'} kernel")
    fm = make_feature_map(MAIN_FM)
    kern = statevector_kernel(fm) if fast else circuit_kernel(fm)
    t0 = time.time()
    K_tr, K_te = gram_matrices(kern, X_tr, X_te)
    kernel_time = time.time() - t0
    log(f"   kernel computed in {kernel_time:.1f}s")
    np.savez_compressed(os.path.join(OUT, "kernel_main.npz"), K_train=K_tr, K_test=K_te)

    q = fit_eval(K_tr, K_te, y_tr, y_te, seed=SEED)
    rows = [{"model": f"QSVC ({MAIN_FM})", "cv_acc": q["cv_acc"], "params": f"C={q['C']}",
             "test_acc": q["test_acc"], "test_f1": q["test_f1"], "test_auc": q["test_auc"],
             "n_support": q["n_support"], "ci95": bootstrap_ci(y_te, q["pred"])}]
    preds = {"QSVC": q["pred"]}

    for kname in ["rbf", "linear"]:
        m, p, cv = classical_svm(X_tr, y_tr, kname)
        pred = m.predict(X_te)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        rows.append({"model": f"Classical SVM ({kname})", "cv_acc": cv, "params": str(p),
                     "test_acc": accuracy_score(y_te, pred), "test_f1": f1_score(y_te, pred),
                     "test_auc": roc_auc_score(y_te, m.decision_function(X_te)),
                     "n_support": int(m.n_support_.sum()), "ci95": bootstrap_ci(y_te, pred)})
        preds[f"SVM_{kname}"] = pred
        if kname == "rbf":
            rbf_gamma = p["gamma"]

    write_csv(rows, os.path.join(OUT, "e1_headline.csv"))
    log("\n" + fmt_table(rows, ["model", "cv_acc", "test_acc", "test_f1", "test_auc", "n_support", "ci95"]))

    mc = {k: mcnemar_exact(y_te, preds["QSVC"], v) for k, v in preds.items() if k != "QSVC"}
    log(f"   McNemar QSVC vs classical: {mc}")

    # --- figures: kernel heatmaps (quantum vs RBF), confusion matrix
    order = np.argsort(y_tr, kind="stable")
    K_rbf = rbf_gram(X_tr, X_tr, rbf_gamma)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, K, title in [(ax[0], K_tr, f"Quantum kernel ({MAIN_FM})"), (ax[1], K_rbf, f"RBF kernel (gamma={rbf_gamma})")]:
        im = a.imshow(K[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
        n0 = int((y_tr == 0).sum())
        a.axhline(n0 - 0.5, color="w", lw=0.8); a.axvline(n0 - 0.5, color="w", lw=0.8)
        a.set_title(f"{title}\nKTA={kernel_target_alignment(K, y_tr):.3f}")
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Training Gram matrices, sorted by label (healthy | near-failure)")
    fig.savefig(os.path.join(FIG, "e1_kernel_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_te, preds["QSVC"])
    fig, a = plt.subplots(figsize=(3.6, 3.2))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    a.set_xticks([0, 1], ["healthy", "fail<=30"]); a.set_yticks([0, 1], ["healthy", "fail<=30"])
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"QSVC test confusion matrix")
    fig.savefig(os.path.join(FIG, "e1_confusion_qsvc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"kernel_seconds": kernel_time, "kernel_backend": "statevector" if fast else "FidelityQuantumKernel",
            "table": rows, "mcnemar": mc,
            "kta": kernel_target_alignment(K_tr, y_tr), **offdiag_stats(K_tr)}, K_tr, K_te


def e2_feature_maps(X_tr, X_te, y_tr, y_te):
    log("E2 feature-map ablation")
    rows = []
    for name in FEATURE_MAPS:
        fm = make_feature_map(name)
        K_tr, K_te = gram_matrices(statevector_kernel(fm), X_tr, X_te)
        r = fit_eval(K_tr, K_te, y_tr, y_te, seed=SEED)
        rows.append({"feature_map": name, "entangling": not name.startswith("Z_"), **circuit_stats(fm),
                     "C": r["C"], "cv_acc": r["cv_acc"], "test_acc": r["test_acc"], "test_auc": r["test_auc"],
                     "n_support": r["n_support"], "kta": kernel_target_alignment(K_tr, y_tr), **offdiag_stats(K_tr)})
    write_csv(rows, os.path.join(OUT, "e2_feature_maps.csv"))
    log("\n" + fmt_table(rows))
    df = {k: [r[k] for r in rows] for k in rows[0]}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(rows))
    colors = ["tab:gray" if not e else "tab:blue" for e in df["entangling"]]
    ax[0].bar(x - 0.2, df["cv_acc"], 0.4, color=colors, alpha=0.5, label="5-fold CV (train)")
    ax[0].bar(x + 0.2, df["test_acc"], 0.4, color=colors, label="test")
    ax[0].set_xticks(x, df["feature_map"], rotation=30, ha="right"); ax[0].set_ylim(0.5, 1)
    ax[0].set_ylabel("accuracy"); ax[0].set_title("Accuracy (grey = no entanglement)"); ax[0].legend()
    ax[1].bar(x, df["kta"], color=colors)
    ax2 = ax[1].twinx(); ax2.plot(x, df["k_std"], "ro-", label="std of off-diag K")
    ax[1].set_xticks(x, df["feature_map"], rotation=30, ha="right")
    ax[1].set_ylabel("kernel-target alignment"); ax2.set_ylabel("off-diagonal kernel std", color="r")
    ax[1].set_title("Kernel quality")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e2_feature_maps.png"), dpi=150); plt.close(fig)
    return rows


def e3_bandwidth(X_tr, X_te, y_tr, y_te):
    log("E3 bandwidth (angle scaling) sweep")
    rows = []
    for fm_name in [MAIN_FM, "ZZ_full_r3", "Z_r2"]:
        fm = make_feature_map(fm_name)
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
            K_tr, K_te = gram_matrices(statevector_kernel(fm), alpha * X_tr, alpha * X_te)
            r = fit_eval(K_tr, K_te, y_tr, y_te, seed=SEED)
            rows.append({"feature_map": fm_name, "alpha": alpha, "C": r["C"], "cv_acc": r["cv_acc"],
                         "test_acc": r["test_acc"], "test_auc": r["test_auc"],
                         "kta": kernel_target_alignment(K_tr, y_tr), **offdiag_stats(K_tr)})
    write_csv(rows, os.path.join(OUT, "e3_bandwidth.csv"))
    log("\n" + fmt_table(rows))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, fm_name in enumerate(dict.fromkeys(r["feature_map"] for r in rows)):
        g = {k: [r[k] for r in rows if r["feature_map"] == fm_name] for k in rows[0]}
        c = f"C{i}"
        ax[0].plot(g["alpha"], g["test_acc"], "o-", color=c, label=f"{fm_name} test")
        ax[0].plot(g["alpha"], g["cv_acc"], "x--", color=c, alpha=0.5, label=f"{fm_name} CV")
        ax[1].plot(g["alpha"], g["k_std"], "o-", color=c, label=f"{fm_name} std")
        ax[1].plot(g["alpha"], g["k_mean"], "s:", color=c, alpha=0.5, label=f"{fm_name} mean")
    ax[1].axhline(1 / 2**4, color="k", lw=0.8, ls="--")
    ax[1].text(0.06, 1 / 2**4 + 0.01, "Haar-random mean 1/2^n", fontsize=7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("angle scale alpha"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("Accuracy vs kernel bandwidth"); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("angle scale alpha")
    ax[1].set_ylabel("off-diagonal kernel value")
    ax[1].set_title("Kernel statistics (too smooth <-> too random)"); ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_bandwidth.png"), dpi=150); plt.close(fig)
    return rows


def e4_learning_curve(X_tr, X_te, y_tr, y_te, K_main, best_alpha):
    log(f"E4 learning curve (QSVC default, QSVC alpha={best_alpha}, RBF SVM)")
    fm = make_feature_map(MAIN_FM)
    K_bw = gram_matrices(statevector_kernel(fm), best_alpha * X_tr, best_alpha * X_te)
    kernels = {f"QSVC {MAIN_FM}": K_main, f"QSVC {MAIN_FM} alpha={best_alpha}": K_bw}
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [20, 40, 80, 160, 400]:
        for rep in range(10 if n < 400 else 1):
            idx = balanced_subset(y_tr, n, rng) if n < 400 else np.arange(400)
            for name, (Ktr, Kte) in kernels.items():
                r = fit_eval(Ktr[np.ix_(idx, idx)], Kte[:, idx], y_tr[idx], y_te, seed=SEED)
                rows.append({"model": name, "n_train": n, "rep": rep, "test_acc": r["test_acc"]})
            m, _, _ = classical_svm(X_tr[idx], y_tr[idx], "rbf")
            rows.append({"model": "Classical SVM (rbf)", "n_train": n, "rep": rep,
                         "test_acc": float(np.mean(m.predict(X_te) == y_te))})
    write_csv(rows, os.path.join(OUT, "e4_learning_curve.csv"))
    agg = group_stats(rows, ["model", "n_train"], ["test_acc"])
    log("\n" + fmt_table(agg))

    fig, a = plt.subplots(figsize=(6, 4))
    for name in dict.fromkeys(r["model"] for r in agg):
        g = [r for r in agg if r["model"] == name]
        a.errorbar([r["n_train"] for r in g], [r["test_acc_mean"] for r in g], yerr=[r["test_acc_std"] for r in g],
                   marker="o", capsize=3, label=name)
    a.set_xscale("log"); a.set_xticks([20, 40, 80, 160, 400], ["20", "40", "80", "160", "400"]); a.minorticks_off()
    a.set_xlabel("training samples"); a.set_ylabel("test accuracy (mean ± std, 10 subsets)")
    a.set_title("Learning curve"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e4_learning_curve.png"), dpi=150); plt.close(fig)
    return agg


def e5_shots(X_tr, X_te, y_tr, y_te, alpha):
    log(f"E5 finite-shot kernel estimation ({MAIN_FM}, alpha={alpha})")
    fm = make_feature_map(MAIN_FM)
    Xa, Xb = alpha * X_tr, alpha * X_te
    K_exact = gram_matrices(statevector_kernel(fm), Xa, Xb)
    r_exact = fit_eval(*K_exact, y_tr, y_te, seed=SEED)
    rows = [{"shots": "exact", "rep": 0, "test_acc": r_exact["test_acc"], "kernel_rmse": 0.0}]
    rng = np.random.default_rng(SEED)
    for shots in [16, 64, 256, 1024, 4096]:
        for rep in range(10):
            K_tr = sample_kernel(K_exact[0], shots, rng, symmetric=True)
            K_te = sample_kernel(K_exact[1], shots, rng, symmetric=False)
            r = fit_eval(K_tr, K_te, y_tr, y_te, C=r_exact["C"], seed=SEED)
            rmse = float(np.sqrt(np.mean((K_tr - K_exact[0]) ** 2)))
            rows.append({"shots": shots, "rep": rep, "test_acc": r["test_acc"], "kernel_rmse": rmse})
    write_csv(rows, os.path.join(OUT, "e5_shots.csv"))
    agg = group_stats([r for r in rows if r["shots"] != "exact"], ["shots"], ["test_acc", "kernel_rmse"])
    log("\n" + fmt_table(agg) + f"\n exact test_acc = {r_exact['test_acc']:.3f}")

    s = np.array([r["shots"] for r in agg])
    acc_m = np.array([r["test_acc_mean"] for r in agg]); acc_s = np.array([r["test_acc_std"] for r in agg])
    rmse = np.array([r["kernel_rmse_mean"] for r in agg])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].errorbar(s, acc_m, yerr=acc_s, marker="o", capsize=3, label="finite shots")
    ax[0].axhline(r_exact["test_acc"], color="k", ls="--", label="exact kernel")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("shots per kernel entry"); ax[0].set_ylabel("test accuracy"); ax[0].legend()
    ax[0].set_title("Accuracy vs shot budget")
    ax[1].loglog(s, rmse, "o-", label="measured")
    ax[1].loglog(s, rmse[0] * np.sqrt(s[0] / s), "k--", label=r"$\propto 1/\sqrt{shots}$")
    ax[1].set_xlabel("shots per kernel entry"); ax[1].set_ylabel("kernel RMSE vs exact"); ax[1].legend()
    ax[1].set_title("Kernel estimation error")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e5_shots.png"), dpi=150); plt.close(fig)
    return {"exact_test_acc": r_exact["test_acc"], "C": r_exact["C"],
            "by_shots": agg}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="use the statevector kernel for E1 too")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    X_tr, X_te, y_tr, y_te = load_data()
    log(f"data: train {X_tr.shape} (pos={y_tr.sum()}), test {X_te.shape} (pos={y_te.sum()})")

    summary = {"main_feature_map": MAIN_FM}
    summary["e0_sanity"] = e0_sanity(X_tr, y_tr)
    summary["e1_headline"], K_tr, K_te = e1_headline(X_tr, X_te, y_tr, y_te, args.fast)
    summary["e2_feature_maps"] = e2_feature_maps(X_tr, X_te, y_tr, y_te)
    bw = e3_bandwidth(X_tr, X_te, y_tr, y_te)
    summary["e3_bandwidth"] = bw
    # choose alpha by *CV* accuracy on train (not test) to avoid test-set selection
    main_bw = [r for r in bw if r["feature_map"] == MAIN_FM]
    best_alpha = max(main_bw, key=lambda r: (r["cv_acc"], -abs(np.log(r["alpha"]))))["alpha"]
    summary["e3_best_alpha_by_cv"] = best_alpha
    log(f"   best alpha by CV: {best_alpha}")
    summary["e4_learning_curve"] = e4_learning_curve(X_tr, X_te, y_tr, y_te, (K_tr, K_te), best_alpha)
    summary["e5_shots"] = e5_shots(X_tr, X_te, y_tr, y_te, best_alpha)

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"done -> {OUT}/")


if __name__ == "__main__":
    main()
