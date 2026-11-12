import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path so common.py is importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from common import load_data, train_val_split, evaluate, save_json, RESULTS_DIR

LOGREG_RESULT_DIR = RESULTS_DIR / "log_reg"
FIGURES_DIR = LOGREG_RESULT_DIR / "figures"


def format_metrics(metrics, prefix):
    """Prefixes keys from common.evaluate standard output (e.g., val_accuracy)."""
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Logistic Regression with custom hyperparameters.")
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum iterations / epochs for solver (max_iter).")
    parser.add_argument("--solver", type=str, default="sgd", choices=["lbfgs", "saga", "liblinear", "sgd"],
                        help="Optimization algorithm.")
    parser.add_argument("--seed", type=int, default=12, help="Seed for training model.")
    return parser.parse_args()


def main():
    args = parse_args()
    LOGREG_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data & split validation set
    X_train_full, X_test, y_train_full, y_test = load_data(quantum=False)
    X_tr, X_val, y_tr, y_val = train_val_split(X_train_full, y_train_full, val_size=0.2, seed=args.seed)

    e1_grid_results = []
    best_entry = None
    best_val_f1 = -1.0

    c_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 3.0, 10.0]
    learning_rates = [0.001, 0.01, 0.1]

    # Grid search over parameters
    for c in c_values:
        for lr in learning_rates:
            if args.solver == "sgd":
                from sklearn.linear_model import SGDClassifier
                clf = SGDClassifier(loss="log_loss", alpha=1.0 / c, learning_rate="constant", eta0=lr, max_iter=args.epochs, random_state=args.seed)
            else:
                clf = LogisticRegression(solver=args.solver, C=c, max_iter=args.epochs, random_state=args.seed)

            clf.fit(X_tr, y_tr)

            # Evaluate split metrics using untouched common.evaluate
            val_raw = evaluate(y_val, clf.predict(X_val))
            test_raw = evaluate(y_test, clf.predict(X_test))

            entry = {
                "feature_map": f"Raw PCA ({args.solver.upper()})",
                "solver": args.solver,
                "C": c,
                "epochs": args.epochs,
                "learning_rate": lr,
                "seed": args.seed,
                "n_support": int(X_tr.shape[1]),
                **format_metrics(val_raw, "val"),
                **format_metrics(test_raw, "test")
            }
            
            if val_raw["f1"] > best_val_f1:
                best_val_f1 = val_raw["f1"]
                best_entry = entry

            e1_grid_results.append(entry)

    # Build detailed JSON object matching target schema
    detailed_json = {
        "default": {
            "model": "Logistic Regression",
            "solver": args.solver,
            "max_iter": args.epochs,
            "seed": args.seed,
            "grid_size": len(e1_grid_results)
        },
        "e0_sanity": {
            "train_samples": len(X_tr),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        },
        "e1_grid": e1_grid_results
    }

    # Save detailed JSON output
    out_path = LOGREG_RESULT_DIR / f"{args.solver}_summary.json"
    save_json(detailed_json, out_path)
    print(f"Detailed grid evaluation saved to '{out_path}'")

    # Generate visual analytics plot
    plot_results(e1_grid_results, args.solver)

    # Print best model table and hyperparameters at the end
    print_test_results(best_entry)


def plot_results(grid_results, solver):
    """Plots metrics against Regularization C on a log scale."""
    c_vals = [r["C"] for r in grid_results]
    val_f1 = [r["val_f1"] for r in grid_results]
    test_f1 = [r["test_f1"] for r in grid_results]
    test_fnr = [r["test_false_negative_rate"] for r in grid_results]
    test_acc = [r["test_accuracy"] for r in grid_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(c_vals, val_f1, label="Val F1", marker="o", color="blue")
    ax1.plot(c_vals, test_f1, label="Test F1", marker="s", color="green")
    ax1.plot(c_vals, test_acc, label="Test Accuracy", marker="^", color="purple")
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Regularization C (Log Scale)")
    ax1.set_ylabel("Score Ratio", color="black")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(c_vals, test_fnr, label="Test False Negative Rate", marker="x", color="red", linestyle="--")
    ax2.set_ylabel("False Negative Rate", color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title(f"Hyperparameter Grid Evaluation Metrics vs Regularization C ({solver.upper()})")
    plt.tight_layout()
    plot_file = FIGURES_DIR / f"{solver}_lr_hyperparameter_grid.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Plots saved to '{plot_file}'\n")


def print_test_results(best_entry):
    """Prints the hyperparameter configuration and markdown table for the best model."""
    if not best_entry:
        return

    total_failing = best_entry["test_tp"] + best_entry["test_fn"]

    print("\n=========================================================================")
    print("                   BEST MODEL TEST SET SUMMARY                           ")
    print("=========================================================================")
    print("Hyperparameters Used:")
    print(f"  • Solver          : {best_entry['solver']}")
    print(f"  • Regularization C: {best_entry['C']}")
    print(f"  • Epochs/Max Iter : {best_entry['epochs']}")
    print(f"  • Learning Rate   : {best_entry['learning_rate']}")
    print(f"  • Seed            : {best_entry['seed']}")
    print("-" * 73)
    print(f"| Test set ({total_failing} failing rows) | Recall | Missed failures | False alarms | F1 |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    print(
        f"| C = {best_entry['C']} | "
        f"{best_entry['test_recall'] * 100:.2f}% | "
        f"{best_entry['test_fn']} | "
        f"{best_entry['test_fp']} | "
        f"{best_entry['test_f1']:.4f} |"
    )
    print("=========================================================================\n")


if __name__ == "__main__":
    main()