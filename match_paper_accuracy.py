"""
Match Research Paper Accuracy (81%)
-------------------------------------
Replicates the exact pipeline from the paper:
  1. Zero-value replacement for Glucose, BP, SkinThickness, BMI, Insulin
  2. XGBRegressor to predict RTML insulin (semi-supervised)
  3. ADASYN oversampling on training data
  4. GridSearchCV hyperparameter tuning
  5. MinMax scaling
  6. XGBoost classifier with tuned params
  7. Evaluates on balanced test (paper method) AND real holdout (real-world method)

Run from project root:
    python match_paper_accuracy.py
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, balanced_accuracy_score)
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import ADASYN

warnings.filterwarnings('ignore')

print("=" * 65)
print("  MATCHING RESEARCH PAPER ACCURACY - XGBoost + ADASYN")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and clean PIMA dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading & cleaning PIMA dataset...")
df = pd.read_csv('diabetes.csv')
df.drop(columns='DiabetesPedigreeFunction', inplace=True)

# Replace impossible zeros (paper explicitly mentions this)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for col in zero_cols:
    n = (df[col] == 0).sum()
    if n > 0:
        df[col] = df[col].replace(0, df[col][df[col] != 0].mean())
        print(f"      Fixed {n} zeros in {col}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Predict RTML insulin using XGBRegressor (paper's semi-supervised step)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Predicting RTML insulin with XGBRegressor...")
feat_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age']

ins_model = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=100,
                         random_state=0, verbosity=0)
ins_model.fit(df[feat_cols], df['Insulin'])

rtml = pd.read_excel('development_files/RTML without Insulin.xlsx')
if 'Insulin' in rtml.columns:
    rtml.drop(columns='Insulin', inplace=True)

# Fix zeros in RTML too
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']:
    if col in rtml.columns:
        n = (rtml[col] == 0).sum()
        if n > 0:
            rtml[col] = rtml[col].replace(0, rtml[col][rtml[col] != 0].mean())

rtml['Insulin'] = ins_model.predict(rtml[feat_cols]).clip(0)
print(f"      RTML insulin range: [{rtml['Insulin'].min():.0f}, {rtml['Insulin'].max():.0f}]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Merge datasets
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Merging datasets...")
cols_order = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "Age", "Outcome"]
merged = pd.concat({"x": df, "y": rtml[cols_order]})
print(f"      Merged: {merged.shape} | Classes: {merged['Outcome'].value_counts().to_dict()}")

X = merged.drop(columns='Outcome')
Y = merged['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0, stratify=Y
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4A: Baseline evaluation WITHOUT synthetic data (before ADASYN)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4A/6] Baseline evaluation WITHOUT ADASYN (before synthetic data)...")
cols_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

scaler_before = MinMaxScaler()
scaler_before.fit(X_train[cols_scale])
X_train_s_before = X_train.copy(); X_train_s_before[cols_scale] = scaler_before.transform(X_train[cols_scale])
X_test_s_before = X_test.copy(); X_test_s_before[cols_scale] = scaler_before.transform(X_test[cols_scale])

baseline_model = XGBClassifier(
    objective='binary:logistic',
    subsample=0.8,
    min_child_weight=1,
    scale_pos_weight=1,
    random_state=0,
    verbosity=0,
    nthread=-1
)
baseline_model.fit(X_train_s_before, y_train)

y_pred_before = baseline_model.predict(X_test_s_before)
y_proba_before = baseline_model.predict_proba(X_test_s_before)[:, 1]

acc_before = accuracy_score(y_test, y_pred_before)
f1_before = f1_score(y_test, y_pred_before)
auc_before = roc_auc_score(y_test, y_proba_before)
bacc_before = balanced_accuracy_score(y_test, y_pred_before)

print(f"      BEFORE (no ADASYN) Test Accuracy : {acc_before * 100:.2f}%")
print(f"      BEFORE (no ADASYN) Balanced Acc  : {bacc_before * 100:.2f}%")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_before = cross_val_score(baseline_model, X_train_s_before, y_train,
                            scoring='accuracy', cv=cv, n_jobs=-1).mean()
print(f"      BEFORE (no ADASYN) CV Accuracy   : {cv_before * 100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4B: ADASYN on training data only (test stays real/unbalanced)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4B/6] Applying ADASYN to training data only...")
ada = ADASYN(random_state=0, sampling_strategy='minority')
X_res, y_res = ada.fit_resample(X_train, y_train)
print(f"      After ADASYN: {dict(zip(*np.unique(y_res, return_counts=True)))}")

# Scale features
scaler_after = MinMaxScaler()
scaler_after.fit(X_res[cols_scale])
X_res_s = X_res.copy();  X_res_s[cols_scale] = scaler_after.transform(X_res[cols_scale])
X_test_s = X_test.copy(); X_test_s[cols_scale] = scaler_after.transform(X_test[cols_scale])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: GridSearchCV hyperparameter tuning (paper uses GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] GridSearchCV hyperparameter tuning (this may take ~1-2 min)...")
param_grid = {
    'max_depth'       : [3, 4, 5],
    'n_estimators'    : [100, 200, 300],
    'learning_rate'   : [0.05, 0.1, 0.15],
    'colsample_bytree': [0.7, 0.8],
    'gamma'           : [0, 1],
}

base_xgb = XGBClassifier(
    objective='binary:logistic',
    subsample=0.8,
    min_child_weight=1,
    scale_pos_weight=1,
    random_state=0,
    verbosity=0,
    nthread=-1
)

gs = GridSearchCV(base_xgb, param_grid, scoring='accuracy',
                  cv=cv, n_jobs=-1, verbose=0)
gs.fit(X_res_s, y_res)

print(f"\n      Best params : {gs.best_params_}")
print(f"      Best CV acc : {gs.best_score_ * 100:.2f}%  ← (paper reports 81% this way)")

best_model = gs.best_estimator_

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Evaluate on REAL held-out test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Evaluating on held-out test set...")
y_pred  = best_model.predict(X_test_s)
y_proba = best_model.predict_proba(X_test_s)[:, 1]

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
bacc = balanced_accuracy_score(y_test, y_pred)

print("\n" + "=" * 65)
print("  FINAL RESULTS")
print("=" * 65)
print(f"  BEFORE CV Accuracy           : {cv_before * 100:.2f}%")
print(f"  CV Accuracy  (paper method) : {gs.best_score_ * 100:.2f}%")
print(f"  BEFORE Test Accuracy         : {acc_before * 100:.2f}%")
print(f"  Test Accuracy (real holdout): {acc * 100:.2f}%")
print(f"  BEFORE Balanced Accuracy     : {bacc_before * 100:.2f}%")
print(f"  Balanced Accuracy           : {bacc * 100:.2f}%")
print(f"  BEFORE F1 Score             : {f1_before:.4f}")
print(f"  F1 Score                    : {f1:.4f}")
print(f"  BEFORE AUC-ROC              : {auc_before:.4f}")
print(f"  AUC-ROC                     : {auc:.4f}")

print("\n" + "=" * 65)
print("  SYNTHETIC DATA IMPACT (AFTER - BEFORE)")
print("=" * 65)
print(f"  Δ CV Accuracy               : {(gs.best_score_ - cv_before) * 100:+.2f}%")
print(f"  Δ Test Accuracy             : {(acc - acc_before) * 100:+.2f}%")
print(f"  Δ Balanced Accuracy         : {(bacc - bacc_before) * 100:+.2f}%")
print(f"  Δ F1 Score                  : {(f1 - f1_before):+.4f}")
print(f"  Δ AUC-ROC                   : {(auc - auc_before):+.4f}")

print("\n" + "=" * 65)
print("  CLASSIFICATION REPORT (real holdout)")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("=" * 65)
print("  CONFUSION MATRIX")
print("=" * 65)
print(f"\n              Predicted")
print(f"              Non-Diab  Diabetic")
print(f"  Actual Non-Diab  {cm[0][0]:5}     {cm[0][1]:5}")
print(f"  Actual Diabetic  {cm[1][0]:5}     {cm[1][1]:5}")
print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE if best so far
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
save = input("Save this model? It will overwrite diabetes_model.pkl [y/n]: ").strip().lower()
if save == 'y':
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler_after, f)
    print("✅ Saved: diabetes_model.pkl and scaler.pkl")
else:
    print("⏭  Skipped. Existing model unchanged.")

print("=" * 65)
print("\n💡 NOTE: Paper's 81% = cross-validation accuracy on ADASYN-balanced data.")
print("   Real-world holdout accuracy is naturally lower (imbalanced real test set).")
print("=" * 65)
