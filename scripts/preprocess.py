import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Step 1: Load Data & Assign Column Names
# ---------------------------------------------------------
data_path = 'data/CMaps/train_FD001.txt'

col_names = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv(data_path, sep=r'\s+', header=None, names=col_names)

# ---------------------------------------------------------
# Step 2: Calculate RUL & Binary Target
# ---------------------------------------------------------
max_cycles = df.groupby('unit')['cycle'].transform('max')
df['RUL'] = max_cycles - df['cycle']
df['target'] = (df['RUL'] <= 30).astype(int)

# ---------------------------------------------------------
# Step 3: Remove Flatline Sensors
# ---------------------------------------------------------
flat_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
feature_cols = [c for c in df.columns if c.startswith('s') and c not in flat_sensors]

# ---------------------------------------------------------
# Step 4: Unit-based Train/Test Split 
# ---------------------------------------------------------
np.random.seed(42)
unique_units = df['unit'].unique()
np.random.shuffle(unique_units)

n_train = int(0.8 * len(unique_units))
train_units = unique_units[:n_train]
test_units = unique_units[n_train:]

df_train_full = df[df['unit'].isin(train_units)]
df_test_full = df[df['unit'].isin(test_units)]

# ---------------------------------------------------------
# Step 5: Balanced Downsampling
# ---------------------------------------------------------
def sample_balanced(df_source, total_samples, seed=42):
    n_per_class = total_samples // 2
    df_pos = df_source[df_source['target'] == 1].sample(n=n_per_class, random_state=seed)
    df_neg = df_source[df_source['target'] == 0].sample(n=n_per_class, random_state=seed)
    return pd.concat([df_pos, df_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

df_train_sub = sample_balanced(df_train_full, total_samples=400, seed=42)

# Test set: every row of the held-out engines, not downsampled. Only training has to be
# small (quantum training cost), predicting is cheap. Class balance stays realistic (~15% failing).
df_test_sub = df_test_full.reset_index(drop=True)

# Extract raw matrices as pure NumPy arrays
X_train = df_train_sub[feature_cols].values.astype(np.float64)
y_train = df_train_sub['target'].values
X_test = df_test_sub[feature_cols].values.astype(np.float64)
y_test = df_test_sub['target'].values

# Engine id and remaining cycles per row (for splitting by engine and for error analysis)
unit_train = df_train_sub['unit'].values
unit_test = df_test_sub['unit'].values
rul_train = df_train_sub['RUL'].values
rul_test = df_test_sub['RUL'].values

# ---------------------------------------------------------
# Step 6: Pure NumPy Standard Scaling (Zero Mean, Unit Var)
# ---------------------------------------------------------
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Guard against potential division by zero
std[std == 0] = 1.0

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# ---------------------------------------------------------
# Step 7: Pure NumPy PCA (Truncated SVD) to 4 Components
# ---------------------------------------------------------
U, S, Vt = np.linalg.svd(X_train_scaled, full_matrices=False)
components = Vt[:4, :]  # Shape: (4, 14)

# Calculate explained variance ratio
variances = (S ** 2) / (len(X_train_scaled) - 1)
total_var = np.sum(variances)
explained_var_ratio = variances[:4] / total_var

print(f"PCA Total Retained Variance (4 components): {np.sum(explained_var_ratio):.2%}")

# Project scaled features onto the 4 principal components
X_train_pca = np.dot(X_train_scaled, components.T)
X_test_pca = np.dot(X_test_scaled, components.T)

# ---------------------------------------------------------
# Step 8: Prepare Separate Classical ML and Quantum ML Data
# ---------------------------------------------------------

# Classical ML: Use PCA features directly
X_train_ml = X_train_pca
X_test_ml = X_test_pca

# Quantum ML: Rescale PCA features to [0, pi]
pca_min = np.min(X_train_pca, axis=0)
pca_max = np.max(X_train_pca, axis=0)

pca_range = pca_max - pca_min
pca_range[pca_range == 0] = 1.0

X_train_qml = (X_train_pca - pca_min) / pca_range * np.pi
X_test_qml = (X_test_pca - pca_min) / pca_range * np.pi

# Clip quantum angles to [0, pi]
X_train_qml = np.clip(X_train_qml, 0, np.pi)
X_test_qml = np.clip(X_test_qml, 0, np.pi)


# ---------------------------------------------------------
# Step 9: Save Separate Datasets
# ---------------------------------------------------------

# Dataset 1: Classical Machine Learning
ml_output = "data/cmapss_ml_4f_500.npz"

np.savez_compressed(
    ml_output,
    X_train=X_train_ml,
    X_test=X_test_ml,
    y_train=y_train,
    y_test=y_test,
    unit_train=unit_train,
    unit_test=unit_test,
    rul_train=rul_train,
    rul_test=rul_test
)

# Dataset 2: Quantum Machine Learning
qml_output = "data/cmapss_qml_4f_500.npz"

np.savez_compressed(
    qml_output,
    X_train=X_train_qml,
    X_test=X_test_qml,
    y_train=y_train,
    y_test=y_test,
    unit_train=unit_train,
    unit_test=unit_test,
    rul_train=rul_train,
    rul_test=rul_test
)


# ---------------------------------------------------------
# Step 10: Display Dataset Information
# ---------------------------------------------------------

print("\nSuccessfully preprocessed both datasets!")

print("\n--- Classical ML Dataset ---")
print(f"Saved to: {ml_output}")
print(f"X_train shape: {X_train_ml.shape}")
print(f"X_test shape:  {X_test_ml.shape}")
print(f"Feature range: [{X_train_ml.min():.4f}, {X_train_ml.max():.4f}]")

print("\n--- Quantum ML Dataset ---")
print(f"Saved to: {qml_output}")
print(f"X_train shape: {X_train_qml.shape}")
print(f"X_test shape:  {X_test_qml.shape}")
print(f"Feature range: [{X_train_qml.min():.4f}, {X_train_qml.max():.4f}]")

print("\nBoth datasets use identical labels and train-test splits.")
print(f"Train: {len(y_train)} rows, {np.mean(y_train):.0%} failing, {len(np.unique(unit_train))} engines")
print(f"Test:  {len(y_test)} rows, {np.mean(y_test):.1%} failing, {len(np.unique(unit_test))} engines")
