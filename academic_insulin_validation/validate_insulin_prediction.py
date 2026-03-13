import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def regression_metrics(y_true, y_pred, label: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{label}")
    print(f"  MAE  : {mae:.4f}   (lower is better)")
    print(f"  RMSE : {rmse:.4f}   (lower is better)")
    print(f"  R²   : {r2:.4f}   (closer to 1 is better)")


def clean_zeros(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            non_zero = out.loc[out[col] != 0, col]
            if len(non_zero) > 0:
                out[col] = out[col].replace(0, non_zero.mean())
    return out


print_header("ACADEMIC INSULIN VALIDATION")
print("Using local copies from academic_insulin_validation/ folder")

# -----------------------------------------------------------------------------
# 1) Load datasets
# -----------------------------------------------------------------------------
pima = pd.read_csv("diabetes.csv")
if "DiabetesPedigreeFunction" in pima.columns:
    pima = pima.drop(columns=["DiabetesPedigreeFunction"])

rtml = pd.read_excel("RTML without Insulin.xlsx")
if "Insulin" in rtml.columns:
    rtml = rtml.drop(columns=["Insulin"])

feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "BMI", "Age"]

# -----------------------------------------------------------------------------
# 2) Clean impossible zeros in PIMA and RTML (for insulin modeling)
# -----------------------------------------------------------------------------
pima = clean_zeros(pima, ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"])
rtml = clean_zeros(rtml, ["Glucose", "BloodPressure", "SkinThickness", "BMI"])

X = pima[feature_cols]
y = pima["Insulin"]

# -----------------------------------------------------------------------------
# 3) Train/Test on PIMA where true insulin is known
# -----------------------------------------------------------------------------
print_header("1) HOLDOUT TEST ON PIMA (TRUE INSULIN KNOWN)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

ins_model = XGBRegressor(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=100,
    random_state=0,
    verbosity=0
)
ins_model.fit(X_train, y_train)

y_pred_holdout = ins_model.predict(X_test).clip(0)
regression_metrics(y_test, y_pred_holdout, "Holdout metrics:")

# -----------------------------------------------------------------------------
# 4) Masking test: simulate missing insulin in PIMA
# -----------------------------------------------------------------------------
print_header("2) MASKING TEST ON PIMA (SIMULATED MISSING INSULIN)")
rng = np.random.default_rng(42)
mask_fraction = 0.20
mask_count = int(len(pima) * mask_fraction)
mask_idx = rng.choice(pima.index.to_numpy(), size=mask_count, replace=False)

masked_true_insulin = pima.loc[mask_idx, "Insulin"].copy()
masked_features = pima.loc[mask_idx, feature_cols]
masked_pred_insulin = ins_model.predict(masked_features).clip(0)

regression_metrics(masked_true_insulin, masked_pred_insulin, f"Masking metrics (masked {mask_count} rows):")

# -----------------------------------------------------------------------------
# 5) Predict insulin for RTML + sanity checks
# -----------------------------------------------------------------------------
print_header("3) RTML SANITY CHECK AFTER INSULIN PREDICTION")
rtml_pred = rtml.copy()
rtml_pred["Insulin"] = ins_model.predict(rtml_pred[feature_cols]).clip(0)

pima_ins = pima["Insulin"]
rtml_ins = rtml_pred["Insulin"]

neg_count = int((rtml_ins < 0).sum())
pima_p1 = float(pima_ins.quantile(0.01))
pima_p99 = float(pima_ins.quantile(0.99))
within_reasonable = ((rtml_ins >= pima_p1) & (rtml_ins <= pima_p99)).mean() * 100

print(f"Negative predictions count           : {neg_count}")
print(f"Predicted insulin range             : [{rtml_ins.min():.2f}, {rtml_ins.max():.2f}]")
print(f"PIMA insulin 1st-99th percentile    : [{pima_p1:.2f}, {pima_p99:.2f}]")
print(f"RTML within PIMA 1st-99th range     : {within_reasonable:.2f}%")

print("\nDistribution comparison (RTML predicted vs PIMA true):")
print(f"  Mean   : {rtml_ins.mean():.2f} vs {pima_ins.mean():.2f}")
print(f"  Median : {rtml_ins.median():.2f} vs {pima_ins.median():.2f}")
print(f"  Std    : {rtml_ins.std():.2f} vs {pima_ins.std():.2f}")

# Save predicted RTML for documentation
output_path = "RTML_with_predicted_insulin.csv"
rtml_pred.to_csv(output_path, index=False)

print("\nSaved:")
print(f"  - {output_path}")
print("\nDone. Use these metrics in your academic report.")
