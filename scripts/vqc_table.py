"""Turns results/vqc/runs.csv into a table image for the slides.

    python scripts/vqc_table.py

Writes results/vqc/results_table.png
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "vqc"


def main():
    runs = pd.read_csv(OUT_DIR / "runs.csv")
    stats = runs.groupby("model").agg(["mean", "std"])
    n_fail = int(runs["test_tp"].iloc[0] + runs["test_fn"].iloc[0])
    n_healthy = int(runs["test_tn"].iloc[0] + runs["test_fp"].iloc[0])

    rows = [
        ("Validation accuracy", "val_accuracy", "{:.3f}"),
        ("Test recall (failures caught)", "test_recall", "{:.3f}"),
        ("Test precision", "test_precision", "{:.3f}"),
        ("Test F1", "test_f1", "{:.3f}"),
        ("Test accuracy", "test_accuracy", "{:.3f}"),
        (f"False negatives (of {n_fail} failing)", "test_fn", "{:.1f}"),
        (f"False positives (of {n_healthy} healthy)", "test_fp", "{:.1f}"),
    ]

    cells = []
    for label, col, fmt in rows:
        row = [label]
        for model in ["entangled", "no_entanglement"]:
            mean, std = stats.loc[model, (col, "mean")], stats.loc[model, (col, "std")]
            row.append(f"{fmt.format(mean)} ± {fmt.format(std)}")
        cells.append(row)

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["", "Entangled", "No entanglement"],
                     colWidths=[0.46, 0.27, 0.27], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d8d4")
        cell.visible_edges = "B"
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "B"
            cell.set_edgecolor("#52514e")
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"

    n_seeds = runs["seed"].nunique()
    ax.set_title(f"VQC results, mean ± std over {n_seeds} seeds", loc="left", fontsize=13, pad=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results_table.png", dpi=250, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'results_table.png'}")


if __name__ == "__main__":
    main()
