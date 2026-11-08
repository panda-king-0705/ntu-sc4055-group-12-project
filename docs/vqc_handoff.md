# VQC handoff (Person 2)

What the VQC part added to the repo, what everyone should reuse, and what to watch out for.
If you're an AI assistant working in this repo, read **Rules** before writing any code.

## Data update (already done, nothing to do)

Two changes to the preprocessing output (`scripts/preprocess.py`, data files regenerated):

- **Test set = every row of the 20 test engines**: 4,291 rows, 620 failing (14.4%), instead of 100 balanced rows. With 100 rows all models scored 96–98% and one row was 1%, so models couldn't be compared. Predicting is cheap, so only training needs to stay small.
- **Engine ID and remaining cycles per row** are now saved (`unit_train`, `unit_test`, `rul_train`, `rul_test`). `train_val_split` uses the IDs to split by engine, so no engine is in both training and validation (before, validation rows shared engines with training rows).

The 400 training rows are exactly the same as before. File names and the existing keys are unchanged, so code that loads the data still works. Raw data for rerunning the preprocessing: put `train_FD001.txt` in `data/CMaps/` (gitignored).

## Setup

```
conda create -n sc4055 python=3.12
conda activate sc4055
pip install -r requirements.txt
```

A plain venv works too (`python -m venv .venv`, then the same `pip install`). Versions are pinned. Qiskit 2.x is very different from most online tutorials: `execute`, `Aer.get_backend`, `qiskit.algorithms` and the old V1 `Sampler`/`Estimator` don't exist anymore.

Run scripts from the repo root, e.g. `python scripts/vqc.py`.

## Files

| File | What it is |
|---|---|
| `scripts/common.py` | Shared data loading, train/val split and metrics. **Everyone uses this.** |
| `scripts/vqc.py` | VQC model + experiment (entangled vs no entanglement, 5 seeds each). About 4 min. |
| `scripts/vqc_layers.py` | Picks the number of layers using validation data only. About 6 min. |
| `scripts/vqc_table.py` | Makes the results table image for the slides. |
| `results/vqc/` | Outputs, listed below. |

`results/vqc/`:
- `summary.csv`: test metrics per model, mean and std over seeds
- `runs.csv`: one row per (model, seed) with train/val/test metrics
- `runs/<model>_seed<k>.npz`: trained weights, loss history, and `test_proba` (P(fail) for each test row)
- `best_entangled.npz`, `best_no_entanglement.npz`: the best model of each type by validation accuracy (ties go to higher validation recall). **Use these for the noise analysis.**
- `layer_search.csv`: validation results for 1, 2 and 3 layers
- `loss_curves.png`, `circuit_entangled.png`, `circuit_no_entanglement.png`, `results_table.png`: figures for the report and slides

## Rules (so all our numbers are comparable)

1. Load data with `load_data(quantum=True)` (features scaled to [0, π], for VQC and QSVC) or `load_data(quantum=False)` (raw PCA features, for LogReg and SVM). Both have the same rows, labels and order.
2. Split the 400 training rows with `train_val_split(X_train, y_train)`, which gives 319 train / 81 validation, split by engine. Don't make your own split or change the seed (42). Pass the full `X_train` from `load_data()`; the function picks the same rows for both data files.
3. Tune everything (SVM `C`/`gamma`, kernel or feature map choice, layers, iterations, thresholds) on the **validation** set. Use the test set only for the final numbers.
4. Score with `evaluate(y_true, y_pred)`. It returns accuracy, precision, recall, f1, false_negative_rate and tp/fp/tn/fn. Label 1 = fails within 30 cycles, so `fn` is a missed failure, the costly error. **The test set is imbalanced (14.4% failing), so compare models on recall, false negatives and F1, not accuracy alone.** A model that always says "healthy" would get 85.6% accuracy.
5. Train final models on the 319 training rows, not all 400, so classical and quantum models see the same data.
6. Where results depend on randomness (VQC starting angles, etc.), run several seeds and report mean ± std. LogReg and SVM are deterministic, so one run is enough.
7. Save outputs in your own folder (`results/classical/`, `results/qsvc/`, `results/noise/`), using the metric names from `evaluate()` so everything merges into one table. Don't overwrite `results/vqc/`.
8. Don't change `common.py`, the data files, or `build_circuit` in `vqc.py` without telling the group. The saved VQC weights only work with the circuit exactly as it is.

Example for a classical model:

```python
from sklearn.svm import SVC
from common import load_data, train_val_split, evaluate

X_train_full, X_test, y_train_full, y_test = load_data(quantum=False)
X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full)

model = SVC(C=1.0).fit(X_train, y_train)   # choose C by looking at X_val / y_val
print(evaluate(y_test, model.predict(X_test)))
```

For error analysis, the remaining cycles of each test row are in the data file: `np.load("data/cmapss_qml_4f_500.npz")["rul_test"]`. For example, check how close to failure the missed failures were.

## The VQC in short

- 4 qubits, one per feature. Angle encoding RY(x_i), then 2 layers of [RY(θ) and RZ(θ) on every qubit, then a CNOT ring 0→1→2→3→0], then a final RY layer. 20 trainable angles.
- Readout: measure all qubits, odd parity = class 1. P(fail) = (1 − ⟨ZZZZ⟩) / 2.
- Training: binary cross-entropy, COBYLA (scipy), at most 500 evaluations, exact noiseless expectation values (Aer `EstimatorV2` with precision 0).
- Ablation: `entangle=False` removes only the CNOTs. Everything else is identical, including the starting angles per seed.

```python
from vqc import VQCClassifier, proba_from_counts

model = VQCClassifier(n_layers=2, entangle=True, seed=0).fit(X_train, y_train, X_val, y_val)
model.predict_proba(X)            # P(fail), 1-D array
model.predict(X, threshold=0.5)   # 0/1 labels
model.save(path); VQCClassifier.load(path)
model.circuit                     # QuantumCircuit without measurements
model.bind(X)                     # parameter values for a whole batch, for Qiskit primitives
```

Results on the test set (5 seeds, mean ± std):

| | Recall | False negatives (of 620) | False positives (of 3,671) | F1 |
|---|---|---|---|---|
| Entangled | 0.956 ± 0.005 | 27.0 ± 3.2 | 289.0 ± 13.5 | 0.790 ± 0.009 |
| No entanglement | 0.934 ± 0.000 | 41.0 ± 0.0 | 256.2 ± 0.4 | 0.796 ± 0.000 |

Entanglement catches more failures but raises more false alarms, so F1 is about equal. The entangled model's misses were all 22–30 cycles from failure. The non-entangled model also missed 5 rows with 10 or fewer cycles left.

## For the noise analysis (Person 4)

Both snippets below have been run and work.

Depolarizing noise through an estimator (exact expectation values, no shots):

```python
import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2
from vqc import VQCClassifier
from common import load_data, evaluate, RESULTS_DIR

_, X_test, _, y_test = load_data(quantum=True)
model = VQCClassifier.load(RESULTS_DIR / "vqc" / "best_entangled.npz")

noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
noise.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["ry", "rz"])
est = EstimatorV2(options={"default_precision": 0.0, "backend_options": {"noise_model": noise}})
print(evaluate(y_test, model.predict(X_test, estimator=est)))
```

Simulated IBM hardware through a sampler with shots:

```python
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from vqc import proba_from_counts

backend = AerSimulator.from_backend(FakeManilaV2())
qc = transpile(model.circuit.measure_all(inplace=False), backend, optimization_level=1, seed_transpiler=0)
# the noise model has to go in the constructor, setting sampler.options afterwards is silently ignored
sampler = SamplerV2(default_shots=2000, seed=0,
                    options={"backend_options": {"noise_model": backend.options.noise_model}})

result = sampler.run([(qc, model.bind(X_test))]).result()[0]
p_fail = np.array([proba_from_counts(result.data.meas.get_counts(i)) for i in range(len(X_test))])
print(evaluate(y_test, (p_fail >= 0.5).astype(int)))
```

Things to know:
- **Transpile for the fake backend.** Its noise model is defined on the device's native gates (ecr, sx, rz, ...). An untranspiled circuit with cx/ry gates would run without noise.
- **The estimator route has no readout errors**, because it computes expectation values straight from the state. Use the sampler route for the hardware-like experiment.
- **The metrics alone will look flat.** Noise mostly pulls predictions toward 0.5 rather than flipping them. With 20% depolarizing noise on every CNOT, the mean |P(fail) − 0.5| on the test set dropped from 0.25 to 0.04 and the loss rose from 0.32 to 0.62, but recall stayed at 0.956 and F1 at 0.79. So also plot loss or confidence, and use finite shots, because shot noise is what flips the predictions sitting near 0.5.
- **Run both models with the same noise.** CNOT noise only affects the entangled model, because it has 8 CNOTs and the other version has none. That is a real trade-off worth discussing, not a bug.

## For the QSVC (Person 3)

- Use `load_data(quantum=True)`, the same split, and pick `C` on validation.
- **Use `FidelityStatevectorKernel`, not `FidelityQuantumKernel`.** Both are in `qiskit_machine_learning.kernels` and give the same values (checked, difference ~1e-11), but on the 4,291 × 319 test kernel the statevector one took 8 seconds and the other would take about 86 minutes (ZZFeatureMap, 4 qubits).
- The VQC uses plain angle encoding (one RY(x_i) per qubit). With that same encoding, the fidelity kernel works out to the product of cos²((x_i − x'_i)/2) over the features, which is easy to compute classically. An entangling feature map like `zz_feature_map` is the "more quantum" choice. Either is fine, but say in the report which one you used and why. A different feature map is a different encoding, not just a different model.

## For the classical baselines (Person 1)

- The classical models aren't in the repo yet. Please push them, trained on the 319-row split (rule 5) and tuned on validation.
- A quick check with default settings on the new data gave: logistic regression recall 0.945, 34 false negatives, F1 0.813; SVM recall 0.960, 25 false negatives, F1 0.816. Your tuned numbers will the official ones though.
