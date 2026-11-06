# Notes for Claude

SC4055 group project: predict whether a turbofan engine (NASA C-MAPSS FD001) fails within 30 cycles. We compare classical models (LogReg, SVM), a VQC and a QSVC, plus a noise study of the quantum models.

Read `docs/vqc_handoff.md` before writing code. The short version:

- Load data, split and score with `scripts/common.py`: `load_data(quantum=True|False)`, `train_val_split` (319 train / 81 val, split by engine, seed 42), `evaluate`. Never make a different split.
- Tune on the validation set only. The test set is for final numbers.
- Label 1 = fails within 30 cycles. False negatives (missed failures) are the costly error.
- The test set is every row of 20 held-out engines (4,291 rows, 14.4% failing), so compare models on recall, false negatives and F1, not accuracy alone.
- Don't modify `scripts/common.py`, `data/*.npz` or `build_circuit` in `scripts/vqc.py` unless the user confirms the group agreed.
- Put new work in new files under `scripts/` and outputs in `results/<name>/`. Don't overwrite `results/vqc/`.
- Versions: Qiskit 2.6, qiskit-aer 0.18, qiskit-machine-learning 0.9.1 (see `requirements.txt`). Old-tutorial APIs (`execute`, `Aer.get_backend`, `qiskit.algorithms`, V1 `Sampler`/`Estimator`) don't exist.
- Run scripts from the repo root: `python scripts/<name>.py`.
