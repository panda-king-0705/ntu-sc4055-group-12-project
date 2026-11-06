"""
Person 3 - Quantum Support Vector Classifier (QSVC) experiments.

Run from the repo root (about 15 min, no GPU needed):

    python scripts/run_qsvc.py

Follows the group protocol in docs/vqc_handoff.md: data, the 319/81 engine-based
train/validation split and the metrics all come from scripts/common.py; every choice
(C, feature map, angle scale) is made on the validation rows; the 4,291-row test set
is only used for reporting.

Experiments
    E0  Sanity: FidelityQuantumKernel == FidelityStatevectorKernel, and qiskit's QSVC
        class == SVC on the precomputed kernel.
    E1  Grid over 8 feature maps x 8 angle scales (alpha). For each: C tuned on
        validation, kernel diagnostics on train, test metrics recorded.
        -> feature-map ablation (does entanglement help?) and bandwidth sweep.
    E2  Final comparison on the test set: QSVC default (Qiskit's ZZFeatureMap, fixed
        before any results), QSVC tuned (best grid point by validation), QSVC with the
        VQC's encoding, classical RBF SVM on the same features, and the two VQCs.
        Engine-level bootstrap CIs and error analysis (how close to failure the misses were).
    E3  Learning curve: 20..319 training rows, 10 random balanced subsets each.
    E4  Finite-shot kernel estimation: 16..4096 shots per kernel entry vs exact.

Outputs -> results/qsvc/ (CSV + JSON) and results/qsvc/figures/ (PNG).
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, RESULTS_DIR, evaluate, load_data, load_units, train_val_split  # noqa: E402
from qsvc_lib import (  # noqa: E402
    C_GRID,
    ENTANGLING,
    FEATURE_MAPS,
    balanced_subset,
    circuit_kernel,
    circuit_stats,
    engine_bootstrap,
    fit_precomputed,
    kernel_blocks,
    kernel_target_alignment,
    make_feature_map,
    offdiag_stats,
    rbf_gram,
    sample_kernel,
    score_test,
    statevector_kernel,
    tune_C,
    val_key,
)

OUT = RESULTS_DIR / "qsvc"
FIG = OUT / "figures"
DEFAULT = ("ZZ_full_r2", 1.0)  # Qiskit's default ZZFeatureMap, fixed a priori
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
GAMMA_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SEED = 0
METRICS = ["accuracy", "precision", "recall", "f1", "false_negative_rate", "tp", "fp", "tn", "fn"]


# ---------------------------------------------------------------------------
# small helpers (no pandas: its DLLs are blocked on some locked-down Windows machines)
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt_table(rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    cells = [[f(r.get(c, "")) for c in cols] for r in rows]
    w = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    line = lambda xs: "  ".join(x.rjust(w[i]) for i, x in enumerate(xs))
    return "\n".join([line(cols)] + [line(x) for x in cells])


def prefixed(prefix, m):
    return {f"{prefix}_{k}": m[k] for k in METRICS}


def label(fm, alpha):
    return fm if alpha == 1.0 else f"{fm}, alpha={alpha}"


def rbf_svm(X_tr, X_va, y_tr, y_va):
    """Classical RBF SVM on the same [0, pi] features, C and gamma tuned on validation."""
    best = None
    for c, g in itertools.product(C_GRID, GAMMA_GRID):
        m = SVC(kernel="rbf", C=c, gamma=g).fit(X_tr, y_tr)
        v = evaluate(y_va, m.predict(X_va))
        if best is None or val_key(v) > val_key(best[1]):
            best = (dict(C=c, gamma=g), v, m)
    return best


class LazyKernels(dict):
    """Recompute a (feature map, alpha) kernel only when a later experiment needs it
    (keeping all 64 grid kernels would take ~0.8 GB)."""

    def __init__(self, X_tr, X_va, X_te):
        super().__init__()
        self.X = (X_tr, X_va, X_te)

    def __missing__(self, key):
        self[key] = kernel_blocks(make_feature_map(key[0]), *self.X, alpha=key[1])
        return self[key]


# ---------------------------------------------------------------------------
def e0_sanity(X_tr, y_tr):
    log("E0 sanity checks")
    from qiskit_machine_learning.algorithms import QSVC

    fm = make_feature_map(DEFAULT[0])
    sub = X_tr[:30]
    kdiff = float(np.abs(circuit_kernel(fm).evaluate(sub) - statevector_kernel(fm).evaluate(sub)).max())
    k = statevector_kernel(fm)
    q = QSVC(quantum_kernel=k, C=1.0).fit(X_tr[:80], y_tr[:80])
    s = SVC(kernel="precomputed", C=1.0).fit(k.evaluate(X_tr[:80]), y_tr[:80])
    agree = float(np.mean(q.predict(X_tr[80:160]) == s.predict(k.evaluate(X_tr[80:160], X_tr[:80]))))
    res = {"max_abs_kernel_diff_circuit_vs_statevector": kdiff, "QSVC_class_vs_precomputed_SVC_agreement": agree}
    log(f"   {res}")
    return res


def e1_grid(X_tr, X_va, X_te, y_tr, y_va, y_te):
    log(f"E1 grid: {len(FEATURE_MAPS)} feature maps x {len(ALPHAS)} angle scales")
    rows = []
    for fm_name in FEATURE_MAPS:
        fm = make_feature_map(fm_name)
        stats = circuit_stats(fm)
        for alpha in ALPHAS:
            K = kernel_blocks(fm, X_tr, X_va, X_te, alpha)
            C, v, model = tune_C(K, y_tr, y_va)
            t, _, _ = score_test(model, K["te"], y_te)
            rows.append({"feature_map": fm_name, "alpha": alpha, "entangling": ENTANGLING[fm_name], **stats,
                         "C": C, "n_support": int(model.n_support_.sum()),
                         "kta_train": kernel_target_alignment(K["tr"], y_tr), **offdiag_stats(K["tr"]),
                         **prefixed("val", v), **prefixed("test", t)})
        log(f"   {fm_name} done")
    write_csv(rows, OUT / "e1_grid.csv")
    return rows


def plot_grid(rows):
    # --- feature-map ablation at alpha = 1 (the circuits as Qiskit defines them)
    a1 = [r for r in rows if r["alpha"] == 1.0]
    x = np.arange(len(a1))
    col = ["tab:blue" if r["entangling"] else "tab:gray" for r in a1]
    names = [r["feature_map"] for r in a1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(x - 0.2, [r["val_accuracy"] for r in a1], 0.4, color=col, alpha=0.45, label="validation accuracy (selection)")
    ax[0].bar(x + 0.2, [r["test_f1"] for r in a1], 0.4, color=col, label="test F1")
    ax[0].set_xticks(x, names, rotation=30, ha="right"); ax[0].set_ylim(0.3, 1)
    ax[0].set_title("Feature maps at alpha = 1 (grey = no entanglement)"); ax[0].legend(fontsize=8)
    ax[1].bar(x, [r["kta_train"] for r in a1], color=col)
    ax2 = ax[1].twinx()
    ax2.plot(x, [r["k_mean"] for r in a1], "ro-")
    ax2.axhline(1 / 16, color="r", ls=":", lw=0.8)
    ax2.text(-0.4, 1 / 16, "Haar-random 1/2^n", color="r", fontsize=7, ha="left", va="bottom",
             bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax[1].set_xticks(x, names, rotation=30, ha="right")
    ax[1].set_ylabel("kernel-target alignment (train)"); ax2.set_ylabel("mean off-diagonal kernel value", color="r")
    ax[1].set_title("Kernel quality")
    fig.tight_layout(); fig.savefig(FIG / "e1_feature_maps.png", dpi=150); plt.close(fig)

    # --- bandwidth sweep
    show = ["RY (VQC encoding)", "Z_r2", "ZZ_linear_r2", "ZZ_full_r2", "ZZ_full_r3"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, fm in enumerate(show):
        g = [r for r in rows if r["feature_map"] == fm]
        a = [r["alpha"] for r in g]
        c = f"C{i}"
        ax[0].plot(a, [r["val_accuracy"] for r in g], "o-", color=c, label=fm)
        ax[1].plot(a, [r["test_f1"] for r in g], "o-", color=c, label=fm)
        ax[2].plot(a, [r["k_mean"] for r in g], "o-", color=c, label=fm)
    ax[2].axhline(1 / 16, color="k", ls="--", lw=0.8); ax[2].text(0.06, 0.07, "Haar-random 1/2^n", fontsize=7)
    for a_, t in zip(ax, ["validation accuracy (used for selection)", "test F1", "mean off-diagonal kernel value (train)"]):
        a_.set_xscale("log"); a_.set_xlabel("angle scale alpha"); a_.set_title(t)
    ax[2].set_yscale("log"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "e1_bandwidth.png", dpi=150); plt.close(fig)


def e2_compare(rows, kernels, X_tr, X_va, X_te, y_tr, y_va, y_te, units_te, rul_te):
    log("E2 final comparison on the test set")
    # tuned = best grid point on validation; ties -> higher train kernel-target alignment
    best = max(rows, key=lambda r: ((round(r["val_accuracy"], 6), round(r["val_recall"], 6)), r["kta_train"]))
    picks = {
        "QSVC default (ZZ_full_r2)": DEFAULT,
        f"QSVC tuned ({label(best['feature_map'], best['alpha'])})": (best["feature_map"], best["alpha"]),
    }
    if (best["feature_map"], best["alpha"]) != ("RY (VQC encoding)", 1.0):  # avoid a duplicate row
        picks["QSVC, VQC encoding (RY)"] = ("RY (VQC encoding)", 1.0)
    table, preds, scores = [], {}, {}
    for name, key in picks.items():
        K = kernels[key]
        C, v, model = tune_C(K, y_tr, y_va)
        t, p, s = score_test(model, K["te"], y_te)
        preds[name], scores[name] = p, s
        table.append({"model": name, "params": f"C={C}", "n_support": int(model.n_support_.sum()),
                      **prefixed("val", v), **prefixed("test", t)})

    params, v, m = rbf_svm(X_tr, X_va, y_tr, y_va)
    p = m.predict(X_te)
    preds["Classical SVM (RBF, same features)"] = p
    table.append({"model": "Classical SVM (RBF, same features)", "params": str(params),
                  "n_support": int(m.n_support_.sum()), **prefixed("val", v), **prefixed("test", evaluate(y_te, p))})

    for vqc in ["entangled", "no_entanglement"]:
        d = np.load(RESULTS_DIR / "vqc" / f"best_{vqc}.npz")
        p = (d["test_proba"] >= 0.5).astype(int)
        name = f"VQC {vqc} (Person 2, best seed)"
        preds[name] = p
        table.append({"model": name, "params": f"seed={int(d['seed'])}", "n_support": "",
                      **prefixed("test", evaluate(y_te, p))})

    # engine-level bootstrap
    ci_f1, diff_f1 = engine_bootstrap(y_te, preds, units_te, "f1")
    ci_rec, _ = engine_bootstrap(y_te, preds, units_te, "recall")
    for r in table:
        r["test_f1_ci95"] = "[{:.3f}, {:.3f}]".format(*ci_f1[r["model"]])
        r["test_recall_ci95"] = "[{:.3f}, {:.3f}]".format(*ci_rec[r["model"]])
        miss = rul_te[(y_te == 1) & (preds[r["model"]] == 0)]
        r["missed_fail_min_rul"] = int(miss.min()) if len(miss) else ""
        r["missed_fail_median_rul"] = float(np.median(miss)) if len(miss) else ""
        r["missed_fail_rul_le_10"] = int((miss <= 10).sum())
    write_csv(table, OUT / "e2_comparison.csv")
    log("\n" + fmt_table(table, ["model", "val_accuracy", "test_recall", "test_fn", "test_fp", "test_f1",
                                 "test_f1_ci95", "missed_fail_min_rul", "missed_fail_rul_le_10"]))

    # --- figures: kernel heatmaps and P(miss) vs remaining cycles
    tuned_key = picks[[k for k in picks if k.startswith("QSVC tuned")][0]]
    order = np.argsort(y_tr, kind="stable")
    n0 = int((y_tr == 0).sum())
    K_rbf = rbf_gram(X_tr, X_tr, params["gamma"])
    mats = [(kernels[DEFAULT]["tr"], "QSVC default (ZZ_full_r2)"),
            (kernels[tuned_key]["tr"], f"QSVC tuned ({label(*tuned_key)})"),
            (K_rbf, f"RBF kernel (gamma={params['gamma']})")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for a, (K, t) in zip(ax, mats):
        im = a.imshow(K[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
        a.axhline(n0 - 0.5, color="w", lw=0.8); a.axvline(n0 - 0.5, color="w", lw=0.8)
        a.set_title(f"{t}\nKTA={kernel_target_alignment(K, y_tr):.3f}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Training Gram matrices sorted by label (healthy | fails within 30 cycles)")
    fig.savefig(FIG / "e2_kernel_heatmaps.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    bins = np.arange(0, 61, 5)
    fig, a = plt.subplots(figsize=(7, 4))
    for i, (name, p) in enumerate(preds.items()):
        flagged = [p[(rul_te >= lo) & (rul_te < lo + 5)].mean() for lo in bins[:-1]]
        a.plot(bins[:-1] + 2.5, flagged, "o-", ms=3, label=name, color=f"C{i}")
    a.axvline(30.5, color="k", ls="--", lw=0.8); a.text(31.5, 0.95, "label boundary (RUL = 30)", fontsize=7)
    a.set_xlabel("remaining useful life (cycles)"); a.set_ylabel("fraction predicted 'fails'")
    a.set_title("Test predictions vs true remaining life"); a.legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "e2_predicted_vs_rul.png", dpi=150); plt.close(fig)

    return {"tuned_choice": {"feature_map": tuned_key[0], "alpha": tuned_key[1]}, "table": table,
            "f1_ci95": ci_f1, "recall_ci95": ci_rec, "f1_diff_ci95": diff_f1}, tuned_key


def e3_learning_curve(kernels, tuned_key, X_tr, X_va, X_te, y_tr, y_va, y_te):
    log("E3 learning curve")
    models = {"QSVC default": kernels[DEFAULT], f"QSVC tuned ({label(*tuned_key)})": kernels[tuned_key]}
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [20, 40, 80, 160, len(y_tr)]:
        for rep in range(10 if n < len(y_tr) else 1):
            idx = balanced_subset(y_tr, n, rng) if n < len(y_tr) else np.arange(len(y_tr))
            for name, K in models.items():
                Ks = {"tr": K["tr"][np.ix_(idx, idx)], "va": K["va"][:, idx]}
                _, _, m = tune_C(Ks, y_tr[idx], y_va)
                t = evaluate(y_te, m.predict(K["te"][:, idx]))
                rows.append({"model": name, "n_train": n, "rep": rep, **prefixed("test", t)})
            _, _, m = rbf_svm(X_tr[idx], X_va, y_tr[idx], y_va)
            rows.append({"model": "Classical SVM (RBF)", "n_train": n, "rep": rep,
                         **prefixed("test", evaluate(y_te, m.predict(X_te)))})
    write_csv(rows, OUT / "e3_learning_curve.csv")

    agg = []
    for name in dict.fromkeys(r["model"] for r in rows):
        for n in dict.fromkeys(r["n_train"] for r in rows):
            g = [r for r in rows if r["model"] == name and r["n_train"] == n]
            d = {"model": name, "n_train": n}
            for k in ["test_f1", "test_recall"]:
                a = np.array([r[k] for r in g])
                d[f"{k}_mean"], d[f"{k}_std"] = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
            agg.append(d)
    log("\n" + fmt_table(agg))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, name in enumerate(dict.fromkeys(r["model"] for r in agg)):
        g = [r for r in agg if r["model"] == name]
        for a, k in zip(ax, ["test_f1", "test_recall"]):
            a.errorbar([r["n_train"] for r in g], [r[f"{k}_mean"] for r in g], yerr=[r[f"{k}_std"] for r in g],
                       marker="o", capsize=3, label=name, color=f"C{i}")
    ticks = sorted({r["n_train"] for r in agg})
    for a, t in zip(ax, ["test F1", "test recall"]):
        a.set_xscale("log"); a.set_xticks(ticks, [str(t_) for t_ in ticks]); a.minorticks_off()
        a.set_xlabel("training rows"); a.set_title(f"{t} (mean ± std over 10 subsets)")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "e3_learning_curve.png", dpi=150); plt.close(fig)
    return agg


def e4_shots(kernels, tuned_key, y_tr, y_va, y_te):
    log(f"E4 finite-shot kernel estimation ({label(*tuned_key)})")
    K = kernels[tuned_key]
    C, _, m = tune_C(K, y_tr, y_va)
    exact = evaluate(y_te, m.predict(K["te"]))
    rng = np.random.default_rng(SEED)
    rows = []
    for shots in [16, 64, 256, 1024, 4096]:
        for rep in range(10):
            Ktr = sample_kernel(K["tr"], shots, rng, symmetric=True)
            Kte = sample_kernel(K["te"], shots, rng, symmetric=False)
            t = evaluate(y_te, fit_precomputed(Ktr, y_tr, C).predict(Kte))
            rows.append({"shots": shots, "rep": rep, "kernel_rmse_train": float(np.sqrt(np.mean((Ktr - K["tr"]) ** 2))),
                         **prefixed("test", t)})
    write_csv(rows, OUT / "e4_shots.csv")
    agg = []
    for s in dict.fromkeys(r["shots"] for r in rows):
        g = [r for r in rows if r["shots"] == s]
        d = {"shots": s}
        for k in ["test_f1", "test_recall", "test_fn", "kernel_rmse_train"]:
            a = np.array([r[k] for r in g], float)
            d[f"{k}_mean"], d[f"{k}_std"] = float(a.mean()), float(a.std(ddof=1))
        agg.append(d)
    log("\n" + fmt_table(agg) + f"\n exact: f1={exact['f1']:.3f} recall={exact['recall']:.3f} fn={exact['fn']}")

    s = np.array([r["shots"] for r in agg])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, c, lab in [("test_f1", "C0", "F1"), ("test_recall", "C1", "recall")]:
        ax[0].errorbar(s, [r[f"{k}_mean"] for r in agg], yerr=[r[f"{k}_std"] for r in agg], marker="o",
                       capsize=3, color=c, label=f"{lab}, finite shots")
        ax[0].axhline(exact[k.split("_", 1)[1]], color=c, ls="--", lw=0.8, label=f"{lab}, exact kernel")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("shots per kernel entry"); ax[0].legend(fontsize=7)
    ax[0].set_title("Test metrics vs shot budget")
    rmse = np.array([r["kernel_rmse_train_mean"] for r in agg])
    ax[1].loglog(s, rmse, "o-", label="measured")
    ax[1].loglog(s, rmse[0] * np.sqrt(s[0] / s), "k--", label=r"$\propto 1/\sqrt{shots}$")
    ax[1].set_xlabel("shots per kernel entry"); ax[1].set_ylabel("kernel RMSE vs exact"); ax[1].legend()
    ax[1].set_title("Kernel estimation error")
    fig.tight_layout(); fig.savefig(FIG / "e4_shots.png", dpi=150); plt.close(fig)
    n_tr, n_te = len(y_tr), len(y_te)
    return {"C": C, "exact": exact, "by_shots": agg,
            "circuits_needed": {"train": n_tr * (n_tr - 1) // 2, "test": n_te * n_tr}}


# ---------------------------------------------------------------------------
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    X_full, X_te, y_full, y_te = load_data(quantum=True)
    X_tr, X_va, y_tr, y_va = train_val_split(X_full, y_full)
    _, units_te = load_units()
    rul_te = np.load(DATA_DIR / "cmapss_qml_4f_500.npz")["rul_test"]
    log(f"train {X_tr.shape} ({y_tr.mean():.0%} failing) | val {X_va.shape} ({y_va.mean():.0%}) | "
        f"test {X_te.shape} ({y_te.mean():.1%}, {len(np.unique(units_te))} engines)")

    summary = {"default": {"feature_map": DEFAULT[0], "alpha": DEFAULT[1]}}
    summary["e0_sanity"] = e0_sanity(X_tr, y_tr)
    rows = e1_grid(X_tr, X_va, X_te, y_tr, y_va, y_te)
    kernels = LazyKernels(X_tr, X_va, X_te)
    plot_grid(rows)
    summary["e1_grid"] = rows
    summary["e2"], tuned_key = e2_compare(rows, kernels, X_tr, X_va, X_te, y_tr, y_va, y_te, units_te, rul_te)
    summary["e3_learning_curve"] = e3_learning_curve(kernels, tuned_key, X_tr, X_va, X_te, y_tr, y_va, y_te)
    summary["e4_shots"] = e4_shots(kernels, tuned_key, y_tr, y_va, y_te)
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"done -> {OUT.relative_to(RESULTS_DIR.parent).as_posix()}/")


if __name__ == "__main__":
    main()
