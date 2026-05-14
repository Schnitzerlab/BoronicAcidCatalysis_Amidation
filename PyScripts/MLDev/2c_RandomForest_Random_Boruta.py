#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 19:37:33 2025

@author: tobiasschnitzer

Random Forest Regression with Boruta Feature Selection, Custom Weighted R2,
and RandomizedSearchCV Hyperparameter Optimization.

This script predicts "conversions (%)" using Boruta-selected features from
chemical descriptors and a RandomForestRegressor. All results are saved for full reproducibility.

"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from boruta import BorutaPy
import matplotlib.pyplot as plt
import joblib

# ---------- SETTINGS ----------
RESULT_DIR = "2c_RandomForest_RandomSearchCV_Boruta"
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

# ---------- CUSTOM METRIC ----------
def r2_weighted_high(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = y_true.copy()
    weights[weights < 1] = 1
    return r2_score(y_true, y_pred, sample_weight=weights)

weighted_r2_scorer = make_scorer(r2_weighted_high, greater_is_better=True)

# ---------- DATA PREPARATION ----------
data = pd.read_excel("DATA.xlsx")
cols = FEATURES.copy()
if TARGET not in cols:
    cols.append(TARGET)
data = data[cols]
data = data.dropna()  # Drop rows with NaN

non_numeric = ["Molecule", "SMILES"]
X = data.drop(columns=non_numeric + [TARGET])
y = data[TARGET].values

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# ---------- BORUTA FEATURE SELECTION ----------
rf_for_boruta = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
boruta_selector = BorutaPy(
    estimator=rf_for_boruta,
    verbose=2,
    random_state=RANDOM_SEED
)
boruta_selector.fit(X_train.values, y_train)

selected_features = X_train.columns[boruta_selector.support_].tolist()
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# Save selected features
with open(os.path.join(RESULT_DIR, "Selected_Features.txt"), "w") as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

# ---------- HYPERPARAMETER TUNING ----------
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500, 800, 1000],
    'max_depth': [None] + list(range(3, 21, 2)),
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20, 30, 50],
    'min_samples_leaf': [1, 2, 3, 5, 7, 10, 20],
    'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.3, 0.5, None],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=RANDOM_SEED)
search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=N_ITER,
    scoring=weighted_r2_scorer,
    cv=CV_FOLDS,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=2,
    return_train_score=True
)
search.fit(X_train_sel, y_train)

pd.DataFrame(search.cv_results_).to_csv(os.path.join(RESULT_DIR, "HPT.csv"), index=False)

# ---------- FINAL MODEL TRAINING ----------
final_model = RandomForestRegressor(
    random_state=RANDOM_SEED, **search.best_params_
)
final_model.fit(X_train_sel, y_train)

joblib.dump(
    {'model': final_model, 'features': selected_features},
    os.path.join(RESULT_DIR, "model_rf_boruta.pkl")
)

# ---------- EVALUATION ----------
def get_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "r2_weighted_high": r2_weighted_high(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

y_pred_train = final_model.predict(X_train_sel)
y_pred_test = final_model.predict(X_test_sel)

metrics_train = get_metrics(y_train, y_pred_train)
metrics_test = get_metrics(y_test, y_pred_test)

cv_scores = cross_val_score(
    final_model, X_train_sel, y_train,
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

# ---------- PLOTTING ----------
plt.figure(figsize=(7,7))
plt.scatter(y_train, y_pred_train, c="gray", alpha=0.3, label="Train")
plt.scatter(y_test, y_pred_test, c="blue", alpha=0.3, label="Test")
minval = min(y.min(), y_pred_test.min())
maxval = max(y.max(), y_pred_test.max())
plt.plot([minval, maxval], [minval, maxval], ls="--", color="darkgreen", label="Optimal fit")
plt.xlabel("Experimental conversions (%)")
plt.ylabel("Predicted conversions (%)")
plt.title("Real vs. Predicted Conversions (Random Forest + Boruta)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "Real_vs_Predicted.png"), dpi=300)
plt.close()

print(f"All results and plots are saved in: {RESULT_DIR}")
