"""Noise analysis of the two trained VQCs (Person 4). Nothing is retrained.

    python scripts/noise_vqc.py            # full run
    python scripts/noise_vqc.py --quick    # smoke test, writes to results/noise/quick/

Loads results/vqc/best_entangled.npz and best_no_entanglement.npz and predicts the full test set
(4,291 rows) under noise:
    N1  depolarizing noise, exact expectation values (no shots, no readout error)
    N2  shots only, no gate noise: does sampling the measurement hurt by itself?
    N3  depolarizing noise + 1,000 shots
    N4  simulated IBM hardware (FakeManilaV2, transpiled), 1,000 shots: the full noise model,
        readout errors only, gate errors only, and every error rate scaled x0.5 / x1 / x2 / x4

Depolarizing levels are the CNOT error; 1-qubit gates get a tenth of it, roughly as on IBM devices.
Besides the evaluate() metrics every row has the loss (BCE) and the confidence, mean |P(fail) - 0.5|,
because noise first pulls P(fail) towards 0.5 and only flips predictions once shots are added.

Breaking point, fixed before running: the smallest CNOT error at which, with 1,000 shots (N3),
(a) mean test F1 drops below the classical RBF SVM, or (b) missed failures reach twice the noiseless count.

Outputs -> results/noise/ (vqc_*.csv, vqc_summary.json) and results/noise/figures/.
"""
import argparse
import copy
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from common import RESULTS_DIR, evaluate, load_data, load_units, save_json
from qsvc_lib import engine_bootstrap
from vqc import VQCClassifier, bce_loss, proba_from_counts

OUT_DIR = RESULTS_DIR / "noise"
MODELS = ["entangled", "no_entanglement"]
COLORS = {"entangled": "#2a78d6", "no_entanglement": "#eb6834"}
LABELS = {"entangled": "Entangled", "no_entanglement": "No entanglement"}
P2_LEVELS = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SHOTS = 1000
SHOT_LIST = [100, 1000, 10000]

# name -> (error scale, NoiseModel.from_backend options, number of sampler seeds)
# T1/T2 can't be scaled on the fake backend, so the scaled runs leave out thermal relaxation
# (from_backend then models a gate's whole error as depolarizing); "full" vs "x1" shows the difference.
MANILA_VARIANTS = {
    "full": (1, {}, 5),
    "readout only": (1, dict(gate_error=False, thermal_relaxation=False), 3),
    "gates only": (1, dict(readout_error=False), 3),
    "x0.5": (0.5, dict(thermal_relaxation=False), 3),
    "x1": (1, dict(thermal_relaxation=False), 3),
    "x2": (2, dict(thermal_relaxation=False), 3),
    "x4": (4, dict(thermal_relaxation=False), 3),
}
SCALED = ["x0.5", "x1", "x2", "x4"]


def depolarizing(p2):
    """CNOT error p2, 1-qubit gates p2 / 10. Lists the 1-qubit gates of both the VQC and the QSVC kernels."""
    noise = NoiseModel()
    if p2 > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        noise.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["ry", "rz", "h", "p"])
    return noise


def manila_noise(scale=1.0, **options):
    """FakeManilaV2's noise model with every gate and readout error multiplied by `scale`."""
    dev = copy.deepcopy(FakeManilaV2())
    for op in dev.target.operation_names:
        for props in dev.target[op].values():
            if props is not None and props.error is not None:
                props.error *= scale
    return NoiseModel.from_backend(dev, **options)


def noisy_sampler(noise, shots, seed):
    # the noise model has to go in here: setting sampler.options.backend_options afterwards is ignored
    return SamplerV2(default_shots=shots, seed=seed, options={"backend_options": {"noise_model": noise}})


def sample_proba(qc, model, X, sampler):
    res = sampler.run([(qc, model.bind(X))]).result()[0]
    return np.array([proba_from_counts(res.data.meas.get_counts(i)) for i in range(len(X))])


def scores(y, p):
    m = evaluate(y, (p >= 0.5).astype(int))
    m.update(loss=bce_loss(y, p), confidence=float(np.abs(p - 0.5).mean()))
    return {f"test_{k}": v for k, v in m.items()}


def breaking_points(levels, f1, fn, ref_f1, fn0):
    """Smallest noise level where mean F1 < the classical reference, and where missed failures >= 2x noiseless
    (and at least one more than noiseless, so a model with 0 misses doesn't trigger at once)."""
    first = lambda hits: next((float(lv) for lv, hit in zip(levels, hits) if hit), None)
    return {"f1_below_classical": first(np.asarray(f1) < ref_f1),
            "missed_failures_doubled": first(np.asarray(fn) >= max(2 * fn0, fn0 + 1))}


def classical_reference():
    """Person 3's classical RBF SVM on the same features (full test set), until Person 1's models are in."""
    rows = csv.DictReader(open(RESULTS_DIR / "qsvc" / "e2_comparison.csv"))
    r = next(r for r in rows if r["model"].startswith("Classical SVM"))
    return {"f1": float(r["test_f1"]), "fn": int(r["test_fn"])}


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_depolarizing(n1, n3, ref, n_fail, path):
    panels = [("test_f1", "Test F1"), ("test_fn", f"Missed failures (of {n_fail})"),
              ("test_confidence", "Confidence, mean |P(fail) − 0.5|"), ("test_loss", "Test loss (BCE)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for name in MODELS:
        e = n1[n1.model == name]
        s = n3[n3.model == name].groupby("p_cx", sort=False)
        for ax, (col, _) in zip(axes.flat, panels):
            ax.plot(e.p_cx, e[col], "o-", color=COLORS[name], lw=2, label=f"{LABELS[name]}, exact")
            ax.errorbar(s[col].mean().index, s[col].mean(), yerr=s[col].std(), fmt="s--", ms=4, capsize=3,
                        color=COLORS[name], alpha=0.6, label=f"{LABELS[name]}, {SHOTS} shots")
    axes[0, 0].axhline(ref["f1"], color="#52514e", ls=":", label="Classical RBF SVM")
    axes[0, 1].axhline(ref["fn"], color="#52514e", ls=":")
    for ax, (_, title) in zip(axes.flat, panels):
        style(ax, title)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(left=-1e-4)
    for ax in axes[1]:
        ax.set_xlabel("CNOT depolarizing error (1-qubit gates: a tenth of it)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("VQC under depolarizing noise", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shots(n1, n2, n_fail, path):
    panels = [("test_f1", "Test F1"), ("test_fn", f"Missed failures (of {n_fail})"),
              ("test_confidence", "Confidence, mean |P(fail) − 0.5|")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for name in MODELS:
        exact = n1[(n1.model == name) & (n1.p_cx == 0)].iloc[0]
        s = n2[n2.model == name].groupby("shots")
        for ax, (col, _) in zip(axes, panels):
            ax.errorbar(s[col].mean().index, s[col].mean(), yerr=s[col].std(), fmt="o-", capsize=3,
                        color=COLORS[name], label=f"{LABELS[name]}, finite shots")
            ax.axhline(exact[col], color=COLORS[name], ls="--", lw=1, label=f"{LABELS[name]}, exact")
    for ax, (_, title) in zip(axes, panels):
        style(ax, title)
        ax.set_xscale("log")
        ax.set_xlabel("Shots per prediction (no gate noise)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_manila(n1, n4, ref, n_fail, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    scale = {"x0.5": 0.5, "x1": 1, "x2": 2, "x4": 4}
    panels = [("test_f1", "Test F1"), ("test_fn", f"Missed failures (of {n_fail})"),
              ("test_confidence", "Confidence, mean |P(fail) − 0.5|")]
    for name in MODELS:
        g = n4[n4.model == name].groupby("variant", sort=False)
        exact = n1[(n1.model == name) & (n1.p_cx == 0)].iloc[0]
        for ax, (col, _) in zip(axes.flat, panels):
            m, sd = g[col].mean(), g[col].std()
            ax.errorbar([0] + [scale[v] for v in SCALED], [exact[col]] + [m[v] for v in SCALED],
                        yerr=[0] + [sd[v] for v in SCALED], fmt="o-", capsize=3, color=COLORS[name], label=LABELS[name])
            ax.errorbar([1], [m["full"]], yerr=[sd["full"]], fmt="*", ms=12, color=COLORS[name], alpha=0.8)
    axes[0, 0].axhline(ref["f1"], color="#52514e", ls=":", label="Classical RBF SVM")
    axes[0, 1].axhline(ref["fn"], color="#52514e", ls=":")
    for ax, (_, title) in zip(axes.flat, panels):
        style(ax, title)
        ax.set_xlabel("FakeManila error rates × (0 = noiseless, ★ = full model with T1/T2)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    # which part of the hardware noise does the damage?
    ax = axes[1, 1]
    bars = ["noiseless", "readout only", "gates only", "full"]
    x = np.arange(len(bars))
    for k, name in enumerate(MODELS):
        g = n4[n4.model == name].groupby("variant")["test_confidence"]
        exact = n1[(n1.model == name) & (n1.p_cx == 0)].iloc[0]["test_confidence"]
        vals = [exact] + [g.mean()[b] for b in bars[1:]]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=COLORS[name], label=LABELS[name])
    ax.set_xticks(x, bars)
    style(ax, "Confidence by noise source (FakeManila)")
    fig.suptitle(f"VQC on simulated IBM hardware (FakeManilaV2, {SHOTS} shots)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="every 10th test row, few levels and seeds")
    args = parser.parse_args()

    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _, X, _, y = load_data(quantum=True)
    units = load_units()[1]
    levels, shot_list, variants = P2_LEVELS, SHOT_LIST, MANILA_VARIANTS
    if args.quick:
        X, y, units = X[::10], y[::10], units[::10]
        levels, shot_list = [0, 0.05], [100]
        variants = {v: (s, kw, 1) for v, (s, kw, _) in MANILA_VARIANTS.items()}
    n_seeds = lambda n: 1 if args.quick else n

    models = {name: VQCClassifier.load(RESULTS_DIR / "vqc" / f"best_{name}.npz") for name in MODELS}
    ref = classical_reference()
    print(f"test rows {len(X)} ({int(y.sum())} failing) | classical RBF SVM: F1 {ref['f1']:.3f}, missed {ref['fn']}",
          flush=True)

    n1, n2, n3, n4 = [], [], [], []
    for p2 in levels:
        est = EstimatorV2(options={"default_precision": 0.0, "backend_options": {"noise_model": depolarizing(p2)}})
        for name, model in models.items():
            n1.append({"model": name, "p_cx": p2, **scores(y, model.predict_proba(X, estimator=est))})
    print("N1 depolarizing, exact: done", flush=True)

    for name, model in models.items():
        qc = model.circuit.measure_all(inplace=False)
        for shots in shot_list:
            for seed in range(n_seeds(5)):
                p = sample_proba(qc, model, X, noisy_sampler(NoiseModel(), shots, seed))
                n2.append({"model": name, "shots": shots, "seed": seed, **scores(y, p)})
        print(f"N2 shots only, {name}: done", flush=True)
        for p2 in levels:
            for seed in range(n_seeds(3)):
                p = sample_proba(qc, model, X, noisy_sampler(depolarizing(p2), SHOTS, seed))
                n3.append({"model": name, "p_cx": p2, "seed": seed, **scores(y, p)})
            print(f"N3 depolarizing + {SHOTS} shots, {name}, p_cx {p2}: done", flush=True)

    backend = AerSimulator.from_backend(FakeManilaV2())
    transpiled, manila_preds = {}, {}
    for name, model in models.items():
        qc = transpile(model.circuit.measure_all(inplace=False), backend, optimization_level=1, seed_transpiler=0)
        transpiled[name] = {"ops": dict(qc.count_ops()), "physical_qubits": qc.layout.final_index_layout()}
        for variant, (scale, options, seeds) in variants.items():
            noise = manila_noise(scale, **options)
            for seed in range(seeds):
                p = sample_proba(qc, model, X, noisy_sampler(noise, SHOTS, seed))
                n4.append({"model": name, "variant": variant, "seed": seed, **scores(y, p)})
                if variant == "full" and seed == 0:
                    manila_preds[name] = (p >= 0.5).astype(int)
            print(f"N4 FakeManila {variant}, {name}: done", flush=True)

    n1, n2, n3, n4 = (pd.DataFrame(r) for r in [n1, n2, n3, n4])
    for df, file in [(n1, "vqc_n1_depolarizing.csv"), (n2, "vqc_n2_shots.csv"),
                     (n3, "vqc_n3_depolarizing_shots.csv"), (n4, "vqc_n4_fakemanila.csv")]:
        df.to_csv(out / file, index=False)

    summary = {"test_rows": len(X), "shots": SHOTS, "classical_reference": ref, "transpiled_for_manila": transpiled}
    for name, model in models.items():
        exact = n1[(n1.model == name) & (n1.p_cx == 0)].iloc[0]
        m3 = n3[n3.model == name].groupby("p_cx")[["test_f1", "test_fn"]].mean()
        pred0 = model.predict(X)
        ci, diff = engine_bootstrap(y, {"noiseless": pred0, "FakeManila": manila_preds[name]}, units, "f1")
        summary[name] = {
            "breaking_point_cx_error": breaking_points(m3.index, m3.test_f1, m3.test_fn, ref["f1"], exact.test_fn),
            "fakemanila_f1_ci95": ci["FakeManila"], "noiseless_minus_fakemanila_f1_ci95": diff["noiseless - FakeManila"],
        }
    save_json(summary, out / "vqc_summary.json")

    cols = ["test_f1", "test_fn", "test_fp", "test_confidence", "test_loss"]
    print("\nN1 depolarizing, exact:\n" + n1.set_index(["model", "p_cx"])[cols].round(3).to_string())
    for df, keys, title in [(n2, ["model", "shots"], "N2 shots only"), (n3, ["model", "p_cx"], f"N3 depolarizing + {SHOTS} shots"),
                            (n4, ["model", "variant"], "N4 FakeManila")]:
        print(f"\n{title} (mean over seeds):\n" + df.groupby(keys, sort=False)[cols].mean().round(3).to_string())
    print("\nTranspiled for FakeManila:", transpiled)
    print("Breaking points and bootstrap:", {k: summary[k] for k in MODELS})

    n_fail = int(y.sum())
    plot_depolarizing(n1, n3, ref, n_fail, out / "figures" / "vqc_depolarizing.png")
    plot_shots(n1, n2, n_fail, out / "figures" / "vqc_shots.png")
    plot_manila(n1, n4, ref, n_fail, out / "figures" / "vqc_fakemanila.png")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
