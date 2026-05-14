#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 00:04:07 2026

@author: tobiasschnitzer
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:12:13 2026

@author: tobiasschnitzer

"""

# === 0. System paths and imports ===
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from sklearn.model_selection import train_test_split

# Import TabPFN and HPO tools (adjust local path if necessary)
sys.path.insert(
    0,
    "/Users/tobiasschnitzer/Documents/orca/Tobi_BA_Project/Feature_Berechnung/Test_Features_Subst_Juli2025/Training/Training_Features_Molecules/PM3_Optimized/TabPFN/src"
)
from tabpfn_extensions.hpo import TunedTabPFNRegressor


# === CUSTOM METRICS ===
def r2_high(y_true, y_pred, threshold=30):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true >= threshold) | (y_pred >= threshold)
    if np.sum(mask) < 2:
        return np.nan
    return r2_score(y_true[mask], y_pred[mask])


def r2_weighted_high(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = y_true.copy()
    weights[weights < 1] = 1  # ensure no invalid weights
    return r2_score(y_true, y_pred, sample_weight=weights)


# === 1. Define directories ===
data_file = "DATA.xlsx"
output_dir = "5_TabPFN_2005_hpo_10"
os.makedirs(output_dir, exist_ok=True)

# === 2. Load data ===
df = pd.read_excel(data_file)

# Define target and feature columns
target_column = "conversions (%)"
excluded_columns = ["Intermediate", "SMILES_int", "Molecule", "SMILES"]

# Keep only rows with valid target values
df = df.dropna(subset=[target_column]).copy()

feature_columns = [col for col in df.columns if col not in excluded_columns + [target_column]]

X = df[feature_columns].values
y = df[target_column].values
indices = df.index.values  # preserve original row indices

# Train-test split including indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=2005
)

# === 3. Hyperparameter optimization using TunedTabPFNRegressor ===
n_trials = 10
hpo_metric = "rmse"  # alternatives: "mae", "mse", "rmse", "r2"

def custom_objective(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return -np.sqrt(mean_squared_error(y_val, y_pred))

regressor = TunedTabPFNRegressor(
    n_trials=n_trials,
    metric=hpo_metric,
    random_state=43,
    verbose=True,
    # objective_fn=custom_objective,  # optional
    # device="cpu",  # or "auto"
)

regressor.fit(X_train, y_train)

# Access best model
best_model = regressor.best_model_

# Predictions
y_pred = best_model.predict(X_test)
y_predtrain = best_model.predict(X_train)

# === 4. Compute performance metrics ===
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2_high_value = r2_high(y_test, y_pred)
r2_weighted_high_value = r2_weighted_high(y_test, y_pred)

# === 5. Save metrics ===
metrics_df = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MAE", "R2high", "R2weighthigh"],
    "Value": [r2, rmse, mae, r2_high_value, r2_weighted_high_value]
})
metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

# === 6. Save best hyperparameters ===
pd.DataFrame([regressor.best_params_]).to_csv(
    os.path.join(output_dir, "best_hyperparameters.csv"),
    index=False
)

# === 7. Plot: Real vs Predicted (Train = gray, Test = blue) ===
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_predtrain, alpha=0.3, color="gray", label="Training data")
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Test data")
min_val = 0
max_val = 100
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
plt.xlabel("Experimental conversion (%)")
plt.ylabel("Predicted conversion (%)")
plt.title("Tuned TabPFN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "real_vs_predicted_colored.png"), dpi=300)
plt.close()

# === 8. Save final model ===
joblib.dump(best_model, os.path.join(output_dir, "final_model_tabpfn.pkl"))

# === 9. Save predictions to Excel ===
df_predictions = df.copy()

# Initialize new columns
df_predictions["conversions_pred (%)"] = np.nan
df_predictions["Set"] = ""

# Assign training predictions
df_predictions.loc[idx_train, "conversions_pred (%)"] = y_predtrain
df_predictions.loc[idx_train, "Set"] = "Training"

# Assign test predictions
df_predictions.loc[idx_test, "conversions_pred (%)"] = y_pred
df_predictions.loc[idx_test, "Set"] = "Test"

# Move new columns to the end
base_columns = [col for col in df_predictions.columns if col not in ["conversions_pred (%)", "Set"]]
df_predictions = df_predictions[base_columns + ["conversions_pred (%)", "Set"]]

# Save Excel file
df_predictions.to_excel(
    os.path.join(output_dir, "DATA_predictions.xlsx"),
    index=False
)

print("Done. Additional file created: DATA_predictions.xlsx")