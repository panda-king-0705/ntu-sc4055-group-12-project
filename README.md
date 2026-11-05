# ntu-sc4055-group-12-project

## Data and Preprocessing

Using [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (train_FD001.txt)

The preprocessing script performs the following steps:

1. Binary Classification: Calculates Remaining Useful Life (RUL) and assigns label 1 if RUL ≤ 30 cycles, otherwise 0.
2. Sensor Selection: Removes seven constant or near-constant sensors, retaining 14 sensor features.
3. Train-Test Split: Splits data by engine unit ID (80% training, 20% testing) to prevent engine-level data leakage.
4. Balanced Sampling: Randomly selects 400 training samples and 100 testing samples, with equal representation of both classes.
5. Standardization: Standardizes sensor features using training-set mean and standard deviation.
6. PCA: Reduces the 14 sensor features to 4 principal components using NumPy SVD.
7. Quantum Angle Encoding: Rescales the PCA features to [0, π] for quantum rotation gates. This transformation is only required for quantum models.
### Preprocessing Configuration
|Parameter|Value|
|---|---|
|Dataset|C-MAPSS FD001|
|Random seed|42|
|Total samples|500|
|Training samples|400|
|Testing samples|100|
|Selected sensor features|14|
|PCA components|4|
|Classical encoding range|[-7.0314, 6.2959]|
|Quantum encoding range|[0, π]|

### Output

The processed dataset is saved as:
```
data/CMaps/cmapss_quantum_4f_500.npz
```
It contains:

- X_train: Training PCA features (400 × 4)
- X_test: Testing PCA features (100 × 4)
- X_train_quantum: Angle-encoded training features (400 × 4)
- X_test_quantum: Angle-encoded testing features (100 × 4)
- y_train: Training labels
- y_test: Testing labels

### Loading the Dataset
Use NumPy to load the preprocessed dataset:
```python
data = np.load("data/cmapss_qml_4f_500.npz") 
# or data = np.load("data/cmapss_ml_4f_500.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
```

### Usage
|Model|Input features|
|---|---|
|Logistic Regression|X_train, X_test|
|Classical SVM|X_train, X_test|
|VQC|X_train_quantum, X_test_quantum|
|QSVC|X_train_quantum, X_test_quantum|

All models share the same training and testing samples and labels to ensure a consistent performance comparison.