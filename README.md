# ntu-sc4055-group-12-project

## Data and Preprocessing

Using [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (train_FD001.txt). To rerun the preprocessing, put `train_FD001.txt` in `data/CMaps/` (that folder is gitignored) and run `python scripts/preprocess.py` from the repo root.

The preprocessing script performs the following steps:

1. Binary Classification: Calculates Remaining Useful Life (RUL) and assigns label 1 if RUL ≤ 30 cycles, otherwise 0.
2. Sensor Selection: Removes seven constant or near-constant sensors, retaining 14 sensor features.
3. Train-Test Split: Splits data by engine unit ID (80% training, 20% testing) to prevent engine-level data leakage.
4. Sampling: Randomly selects 400 training samples with equal representation of both classes. The test set keeps every row of the 20 test engines (4,291 rows, 14.4% failing): only training has to be small because of the quantum training cost, and predicting is cheap.
5. Standardization: Standardizes sensor features using training-set mean and standard deviation.
6. PCA: Reduces the 14 sensor features to 4 principal components using NumPy SVD.
7. Quantum Angle Encoding: Rescales the PCA features to [0, π] for quantum rotation gates, using the training-set min and max. Test values outside that range are clipped (89 of 4,291 rows). This transformation is only required for quantum models.
### Preprocessing Configuration
|Parameter|Value|
|---|---|
|Dataset|C-MAPSS FD001|
|Random seed|42|
|Training samples|400 (200 per class, 78 engines)|
|Testing samples|4,291 (all rows of 20 engines, 620 failing)|
|Selected sensor features|14|
|PCA components|4|
|Classical encoding range|[-7.0314, 6.2959]|
|Quantum encoding range|[0, π]|

### Output

The processed datasets are saved as (the "500" in the names is from the first version and was kept so nothing breaks):
```
data/cmapss_ml_4f_500.npz    # raw PCA features, for classical models
data/cmapss_qml_4f_500.npz   # PCA features scaled to [0, π], for quantum models
```
Both contain the same rows in the same order:

- X_train: Training features (400 × 4)
- X_test: Testing features (4,291 × 4)
- y_train, y_test: Labels (1 = fails within 30 cycles)
- unit_train, unit_test: Engine ID of each row
- rul_train, rul_test: Remaining cycles of each row (for error analysis)

### Loading the Dataset
Use the shared helpers in `scripts/common.py`, so every model uses the same split and metrics (see `docs/vqc_handoff.md`):
```python
from common import load_data, train_val_split, evaluate

X_train_full, X_test, y_train_full, y_test = load_data(quantum=True)   # or quantum=False
X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full)   # by engine, 319 / 81
```

### Usage
|Model|Data file|
|---|---|
|Logistic Regression|`load_data(quantum=False)`|
|Classical SVM|`load_data(quantum=False)`|
|VQC|`load_data(quantum=True)`|
|QSVC|`load_data(quantum=True)`|

All models share the same training, validation and testing samples and labels to ensure a consistent performance comparison. The test set is imbalanced, so compare models on recall, false negatives and F1 rather than accuracy alone.

## Quantum Support Vector Classifier (QSVC)

Code: `scripts/qsvc_lib.py` (feature maps, quantum kernels, validation-based tuning, engine-level bootstrap) and `scripts/run_qsvc.py` (all experiments, about 10 min: `python scripts/run_qsvc.py`). Results, figures and discussion: [`results/qsvc/README.md`](results/qsvc/README.md).

It uses the shared protocol: `common.py` data, the 319/81 split, tuning on validation only, and `evaluate()` metrics. Kernels come from `FidelityStatevectorKernel`, which is checked against `FidelityQuantumKernel` (they agree to within 4e-12).

| Test set (620 failing rows) | Recall | Missed failures | False alarms | F1 |
|---|---|---|---|---|
| QSVC, Qiskit default ZZFeatureMap | 0.718 | 175 | 935 | 0.445 |
| QSVC, tuned on validation (Z_r2, α=0.3) | 0.966 | 21 | 316 | 0.780 |
| Classical RBF SVM, same features | 0.976 | 15 | 340 | 0.773 |
|Logistic Regression (solver=sgd, c=0.05, lr=0.001, epoch=1000)| 0.974 | 16 | 353 | 0.766 |

The tuned QSVC is statistically tied with the RBF SVM and both VQCs (engine-level bootstrap). The default entangling feature map is much worse, because its kernel behaves like one built from random states.
