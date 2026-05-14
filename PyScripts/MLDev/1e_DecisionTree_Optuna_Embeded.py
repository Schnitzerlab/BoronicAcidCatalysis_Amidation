#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 12:02:39 2025

@author: tobiasschnitzer

DecisionTreeRegressor optimization using Optuna Bayesian Optimization
and embedded feature selection (feature importances).


"""

import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import joblib

# ----------------------------- CONFIGURATION ------------------------------

RESULT_DIR = "1e_Decision_Tree_Optuna_Embedded"
os.makedirs(RESULT_DIR, exist_ok=True)
RANDOM_SEED = 42
N_TRIALS = 1000
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

# --------------------- Optuna HPT Objective (all features) ----------------

def optuna_objective(trial):
    params = {
        "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(3, 31, 2))),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 3, 5, 7, 10, 15, 20, 30, 50]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 3, 5, 7, 10, 20]),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', 0.2, 0.3, 0.5, None]),
        "criterion": trial.suggest_categorical("criterion", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
    }
    model = DecisionTreeRegressor(random_state=RANDOM_SEED, **params)
    score = cross_val_score(
        model, X_train, y_train,
        cv=CV_FOLDS, scoring=weighted_r2_scorer, n_jobs=-1
    ).mean()
    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=True)
optuna_df = study.trials_dataframe()
optuna_df.to_csv(os.path.join(RESULT_DIR, "HPT.csv"), index=False)

# --------------------- Embedded Feature Selection --------------------------

# 1. Train best estimator on *all* train features:
best_params = study.best_params
embedded_model = DecisionTreeRegressor(random_state=RANDOM_SEED, **best_params)
embedded_model.fit(X_train, y_train)
importances = pd.Series(embedded_model.feature_importances_, index=X_train.columns)
selected_features = importances[importances > 0].index.tolist()
if len(selected_features) == 0:
    selected_features = list(X_train.columns)[:10]  # fallback

with open(os.path.join(RESULT_DIR, "Selected_Features.txt"), "w") as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# --------------------- Retrain Final Model on selected features ------------

final_model = DecisionTreeRegressor(random_state=RANDOM_SEED, **best_params)
final_model.fit(X_train_sel, y_train)
joblib.dump({'model': final_model, 'features': selected_features}, os.path.join(RESULT_DIR, "model_dt_embedded_optuna.pkl"))

# --------------------- Evaluation & Metrics -------------------------------

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

# --------------------- Real vs Predicted Plot -----------------------------

plt.figure(figsize=(7,7))
plt.scatter(y_train, y_pred_train, c="gray", alpha=0.3, label="Train")
plt.scatter(y_test, y_pred_test, c="blue", alpha=0.3, label="Test")
minval = min(y.min(), y_pred_test.min())
maxval = max(y.max(), y_pred_test.max())
plt.plot([minval, maxval], [minval, maxval], ls="--", color="darkgreen", label="Optimal fit")
plt.xlabel("Experimental conversions (%)")
plt.ylabel("Predicted conversions (%)")
plt.title("Real vs. Predicted Conversions (Decision Tree + Embedded Selection + Optuna)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "Real_vs_Predicted.png"), dpi=300)
plt.close()

print(f"All results and files saved in: {RESULT_DIR}")
