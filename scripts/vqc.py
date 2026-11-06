"""VQC (variational quantum classifier) for "does the engine fail within 30 cycles?"

Circuit on 4 qubits (one per PCA feature):
    RY(x_i) on every qubit                         angle encoding, x is already in [0, pi]
    n_layers x [RY(θ) RZ(θ) on every qubit, then a ring of CNOTs]
    RY(θ) on every qubit
    measure all qubits -> odd parity = class 1 (failure), even parity = class 0

entangle=False is the ablation: same circuit, same parameters, just no CNOTs.
Training is noiseless with exact expectation values (Aer, no shot noise), optimiser is COBYLA.

    python scripts/vqc.py                        # full run: both circuits, 5 seeds each
    python scripts/vqc.py --seeds 1 --maxiter 50 # quick test

Everything is written to results/vqc/. For the noise experiments, load a trained model with
    model = VQCClassifier.load("results/vqc/best_entangled.npz")
and pass a noisy Aer estimator to model.predict(X, estimator=...), or build your own sampler
run from model.circuit / model.bind(X) and decode the counts with proba_from_counts().
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

from common import RESULTS_DIR, evaluate, load_data, train_val_split

OUT_DIR = RESULTS_DIR / "vqc"


def build_circuit(n_qubits=4, n_layers=2, entangle=True):
    x = ParameterVector("x", n_qubits)
    theta = ParameterVector("θ", n_qubits * (2 * n_layers + 1))
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.ry(x[i], i)
    qc.barrier()

    k = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.ry(theta[k], i)
            qc.rz(theta[k + 1], i)
            k += 2
        if entangle:
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)
        qc.barrier()

    # only RY at the end, an RZ right before measuring Z wouldn't change anything
    for i in range(n_qubits):
        qc.ry(theta[k], i)
        k += 1

    return qc, x, theta


def bce_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def proba_from_counts(counts):
    """P(class 1) from measurement counts = share of shots with odd parity."""
    shots = sum(counts.values())
    odd = sum(c for bits, c in counts.items() if bits.count("1") % 2 == 1)
    return odd / shots


class VQCClassifier:
    def __init__(self, n_qubits=4, n_layers=2, entangle=True, maxiter=500, seed=0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.maxiter = maxiter
        self.seed = seed

        self.circuit, self.x_params, self.theta_params = build_circuit(n_qubits, n_layers, entangle)
        # <Z...Z> = P(even parity) - P(odd parity)
        self.observable = SparsePauliOp("Z" * n_qubits)
        self.estimator = EstimatorV2(options={"default_precision": 0.0})  # exact

        self.weights = None
        self.history = {}

    def bind(self, X, weights=None):
        """Parameter values for a whole batch: one row of features + the same weights per sample."""
        weights = self.weights if weights is None else weights
        return {
            tuple(self.x_params): X,
            tuple(self.theta_params): np.tile(weights, (len(X), 1)),
        }

    def _proba(self, X, weights, estimator=None):
        estimator = estimator or self.estimator
        job = estimator.run([(self.circuit, self.observable, self.bind(X, weights))])
        parity = job.result()[0].data.evs
        return (1 - parity) / 2

    def predict_proba(self, X, estimator=None):
        return self._proba(X, self.weights, estimator)

    def predict(self, X, threshold=0.5, estimator=None):
        return (self.predict_proba(X, estimator) >= threshold).astype(int)

    def fit(self, X, y, X_val=None, y_val=None, val_every=10):
        rng = np.random.default_rng(self.seed)
        w0 = rng.uniform(-np.pi, np.pi, len(self.theta_params))
        self.history = {"train_loss": [], "val_loss": [], "val_step": []}

        def objective(w):
            loss = bce_loss(y, self._proba(X, w))
            self.history["train_loss"].append(loss)
            step = len(self.history["train_loss"])
            if X_val is not None and step % val_every == 0:
                self.history["val_loss"].append(bce_loss(y_val, self._proba(X_val, w)))
                self.history["val_step"].append(step)
            return loss

        res = minimize(objective, w0, method="COBYLA", options={"maxiter": self.maxiter})
        self.weights = res.x
        return self

    def save(self, path, **extra):
        np.savez(
            path,
            weights=self.weights,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            entangle=self.entangle,
            seed=self.seed,
            train_loss=np.array(self.history.get("train_loss", [])),
            val_loss=np.array(self.history.get("val_loss", [])),
            val_step=np.array(self.history.get("val_step", [])),
            **extra,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(int(d["n_qubits"]), int(d["n_layers"]), bool(d["entangle"]), seed=int(d["seed"]))
        model.weights = d["weights"]
        return model


def plot_curves(curves, path):
    colors = {"entangled": "#2a78d6", "no_entanglement": "#eb6834"}
    labels = {"entangled": "Entangled (CNOT ring)", "no_entanglement": "No entanglement"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for name, runs in curves.items():
        for ax, key in zip(axes, ["train_loss", "val_loss"]):
            # COBYLA can stop early, so pad shorter runs with their last value
            n = max(len(r[key]) for r in runs)
            losses = np.array([np.pad(r[key], (0, n - len(r[key])), mode="edge") for r in runs])
            if key == "train_loss":
                steps = np.arange(1, n + 1)
            else:
                steps = max((r["val_step"] for r in runs), key=len)
            ax.plot(steps, losses.mean(axis=0), color=colors[name], lw=2, label=labels[name])
            ax.fill_between(steps, losses.min(axis=0), losses.max(axis=0), color=colors[name], alpha=0.12, lw=0)

    axes[0].set_title("Training loss", loc="left")
    axes[1].set_title("Validation loss", loc="left")
    axes[0].set_ylabel("Binary cross-entropy")
    for ax in axes:
        ax.set_xlabel("COBYLA function evaluations")
        ax.grid(color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[1].legend(frameon=False)
    fig.suptitle("VQC training, mean over seeds (shaded = min to max)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=500)
    args = parser.parse_args()

    (OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)

    X_train_full, X_test, y_train_full, y_test = load_data(quantum=True)
    X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full)
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    print(f"train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    rows, curves, best = [], {}, {}
    for entangle in [True, False]:
        name = "entangled" if entangle else "no_entanglement"
        curves[name] = []

        for seed in range(args.seeds):
            start = time.time()
            model = VQCClassifier(n_layers=args.layers, entangle=entangle, maxiter=args.maxiter, seed=seed)
            model.fit(X_train, y_train, X_val, y_val)

            row = {"model": name, "seed": seed, "n_params": len(model.theta_params),
                   "evals": len(model.history["train_loss"]), "train_time_s": round(time.time() - start, 1)}
            for split, (X, y) in splits.items():
                for metric, value in evaluate(y, model.predict(X)).items():
                    row[f"{split}_{metric}"] = value
            rows.append(row)
            curves[name].append(model.history)

            run_path = OUT_DIR / "runs" / f"{name}_seed{seed}.npz"
            model.save(run_path, test_proba=model.predict_proba(X_test))
            print(f"{name:16s} seed {seed}  val acc {row['val_accuracy']:.3f}  test F1 {row['test_f1']:.3f}"
                  f"  test FN {row['test_fn']:3d}  ({row['train_time_s']}s)")

            # pick the model for the noise study on validation accuracy, ties -> higher val recall
            score = (row["val_accuracy"], row["val_recall"])
            if name not in best or score > best[name][0]:
                best[name] = (score, model)

    for name, (_, model) in best.items():
        model.save(OUT_DIR / f"best_{name}.npz", test_proba=model.predict_proba(X_test))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "runs.csv", index=False)

    cols = ["test_accuracy", "test_recall", "test_precision", "test_f1", "test_fn", "test_fp"]
    summary = df.groupby("model")[cols].agg(["mean", "std"])
    summary.to_csv(OUT_DIR / "summary.csv")
    print("\nTest set, mean and std over seeds:")
    print(summary.round(3).to_string())
    print("\nModels picked for noise analysis (by val accuracy):",
          {name: int(m.seed) for name, (_, m) in best.items()})

    plot_curves(curves, OUT_DIR / "loss_curves.png")
    for entangle in [True, False]:
        qc, _, _ = build_circuit(n_layers=args.layers, entangle=entangle)
        name = "entangled" if entangle else "no_entanglement"
        qc.draw("mpl", fold=-1).savefig(OUT_DIR / f"circuit_{name}.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
