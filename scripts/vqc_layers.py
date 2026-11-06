"""How many layers should the VQC have? Decided on the validation set only, test set is not touched.

    python scripts/vqc_layers.py

Writes results/vqc/layer_search.csv
"""
import pandas as pd

from common import RESULTS_DIR, evaluate, load_data, train_val_split
from vqc import VQCClassifier, bce_loss


def main():
    X_train_full, _, y_train_full, _ = load_data(quantum=True)
    X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full)

    rows = []
    for n_layers in [1, 2, 3]:
        for entangle in [True, False]:
            for seed in range(3):
                model = VQCClassifier(n_layers=n_layers, entangle=entangle, seed=seed).fit(X_train, y_train)
                rows.append({
                    "model": "entangled" if entangle else "no_entanglement",
                    "n_layers": n_layers,
                    "seed": seed,
                    "n_params": len(model.theta_params),
                    "train_loss": bce_loss(y_train, model.predict_proba(X_train)),
                    "val_loss": bce_loss(y_val, model.predict_proba(X_val)),
                    "val_accuracy": evaluate(y_val, model.predict(X_val))["accuracy"],
                })
                print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    (RESULTS_DIR / "vqc").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "vqc" / "layer_search.csv", index=False)
    summary = df.groupby(["model", "n_layers"])[["n_params", "train_loss", "val_loss", "val_accuracy"]].mean()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
