#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:56:24 2025

@author: tobiasschnitzer

Random Forest Regression with RFE Feature Selection and RandomizedSearchCV Hyperparameter Optimization

Predicts 'conversions (%)' using a RandomForestRegressor, RFE feature selection,
custom weighted R² metric, and extensive HPT. All results are saved for reproducibility.

"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import joblib

# ----------------------------- CONFIGURATION ------------------------------

RESULT_DIR = "2a_RandomForest_RandomSearchCV_RFE_10"
os.makedirs(RESULT_DIR, exist_ok=True)
RANDOM_SEED = 42
N_ITER = 2000
CV_FOLDS = 5

TARGET = "conversions (%)"
FEATURES = [
    "Molecule","SMILES","Molecular Weight (g/mol)","Rotatable Bond Count","TPSA",
    "2_binary","3_binary","4_binary","5_binary","6_binary","Molmasse_2","Molmasse_3","Molmasse_4","Molmasse_5","Molmasse_6",
    "VdW_2","VdW_3","VdW_4","VdW_5","VdW_6","BV_2","BV_3","BV_4","BV_5","BV_6",
    "Dipole_Magnitude_Debye","Fukui+","Fukui-","B–C distance (Å)","B–O distance 1 (Å)","B–O distance 2 (Å)",
    "Dihedral O–B–O / C–C–C (°)","HOMO Energy (eV)","LUMO Energy (eV)","HOMO-LUMO Gap (eV)","E_neutral (Eh)",
    "E_cation (Eh)","E_anion (Eh)","Ionization Energy (eV)","Electron Affinity (eV)","Mulliken_B_Charge",
    "Loewdin_B_Charge","Mayer_QA","Mayer_VA","Mayer_BVA","Mayer_FA"
]

# --------------------- Custom weighted R2 metric --------------------------

def r2_weighted_high(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = y_true.copy()
    weights[weights < 1] = 1
    return r2_score(y_true, y_pred, sample_weight=weights)

weighted_r2_scorer = make_scorer(r2_weighted_high, greater_is_better=True)

# --------------------- Data Loading & Cleaning ----------------------------

data = pd.read_excel("DATA.xlsx")
use_cols = FEATURES.copy()
if TARGET not in use_cols:
    use_cols.append(TARGET)
data = data[use_cols]
data = data.dropna()

non_numeric = ["Molecule", "SMILES"]
X = data.drop(columns=non_numeric + [TARGET])
y = data[TARGET]

# --------------------- Train/Test Split -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# --------------------- Feature Selection (RFE) ----------------------------

base_estimator = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=100)
rfe = RFE(base_estimator, n_features_to_select=10)
rfe.fit(X_train, y_train)
X_train_sel = rfe.transform(X_train)
X_test_sel = rfe.transform(X_test)
selected_features = X.columns[rfe.support_].tolist()

with open(os.path.join(RESULT_DIR, "Selected_Features.txt"), "w") as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

# --------------------- Hyperparameter Search ------------------------------

param_grid = {
    'n_estimators': [50, 100, 200, 300, 500, 800, 1000],
    'max_depth': [None] + list(range(3, 21, 2)),
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20, 30, 50],
    'min_samples_leaf': [1, 2, 3, 5, 7, 10, 20],
    'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.3, 0.5, None],
    'bootstrap': [True, False]
}

reg = RandomForestRegressor(random_state=RANDOM_SEED)
search = RandomizedSearchCV(
    reg,
    param_distributions=param_grid,
    n_iter=N_ITER,
    scoring=weighted_r2_scorer,
    cv=CV_FOLDS,
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_SEED,
    return_train_score=True
)
search.fit(X_train_sel, y_train)

# Save hyperparameter search results
cv_results = pd.DataFrame(search.cv_results_)
cv_results.to_csv(os.path.join(RESULT_DIR, "HPT.csv"), index=False)

# --------------------- Final Model Training -------------------------------

best_model = search.best_estimator_

joblib.dump({'model': best_model, 'features': selected_features}, os.path.join(RESULT_DIR, "model_rf_rfe.pkl"))

# --------------------- Evaluation & Metrics -------------------------------

def get_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "r2_weighted_high": r2_weighted_high(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

y_pred_train = best_model.predict(X_train_sel)
y_pred_test = best_model.predict(X_test_sel)

metrics_train = get_metrics(y_train, y_pred_train)
metrics_test = get_metrics(y_test, y_pred_test)

cv_scores = cross_val_score(
    best_model, X_train_sel, y_train,
    scoring=weighted_r2_scorer, cv=CV_FOLDS, n_jobs=-1
)
cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

results = {
    "Test_R2": metrics_test["R2"],
    "Test_r2_weighted_high": metrics_test["r2_weighted_high"],
    "Test_RMSE": metrics_test["RMSE"],
    "Test_MAE": metrics_test["MAE"],
    "Train_R2": metrics_train["R2"],
    "Train_r2_weighted_high": metrics_train["r2_weighted_high"],
    "CV_r2_weighted_high_mean": cv_mean,
    "CV_r2_weighted_high_std": cv_std
}
pd.DataFrame([results]).to_csv(os.path.join(RESULT_DIR, "Results.csv"), index=False)

# --------------------- Real vs Predicted Plot -----------------------------

plt.figure(figsize=(7,7))
plt.scatter(y_train, y_pred_train, c="gray", alpha=0.3, label="Train")
plt.scatter(y_test, y_pred_test, c="blue", alpha=0.3, label="Test")
minval = min(y.min(), y_pred_test.min())
maxval = max(y.max(), y_pred_test.max())
plt.plot([minval, maxval], [minval, maxval], ls="--", color="darkgreen", label="Optimal fit")
plt.xlabel("Experimental conversions (%)")
plt.ylabel("Predicted conversions (%)")
plt.title("Real vs. Predicted Conversions (Random Forest + RFE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "Real_vs_Predicted.png"), dpi=300)
plt.close()

print(f"All results and files saved in: {RESULT_DIR}")
