# Save the trained model and scaler for web app deployment
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN

print("Loading and preparing data...")

# Load datasets
Pima_dataset = pd.read_csv('diabetes.csv')
Pima_dataset.drop(columns='DiabetesPedigreeFunction', axis=1, inplace=True)

# Load and merge RTML dataset
RTML_dataset = pd.read_excel('RTML without Insulin.xlsx')
RTML_dataset.drop(columns='Insulin', axis=1, inplace=True)

# Simple insulin imputation for RTML (using mean from PIMA)
RTML_dataset['Insulin'] = Pima_dataset['Insulin'].mean()

# Merge datasets
RTML_Merged = RTML_dataset[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "Outcome"]]
pieces = {"x": Pima_dataset, "y": RTML_Merged}
PIMA_RTML = pd.concat(pieces)

# Prepare data
X = PIMA_RTML.drop(columns='Outcome', axis=1)
Y = PIMA_RTML['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0, stratify=Y)

print("Applying ADASYN...")
ada = ADASYN(random_state=0, sampling_strategy='minority')
X_smote, y_smote = ada.fit_resample(X_train, y_train)

# Scale features
print("Scaling features...")
cols_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
scaler = MinMaxScaler()
scaler.fit(X_smote[cols_to_scale])
X_smote[cols_to_scale] = scaler.transform(X_smote[cols_to_scale])

# Train XGBoost model
print("Training XGBoost model...")
xgbc = XGBClassifier(
    colsample_bytree=0.8,
    gamma=1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1
)
xgbc.fit(X_smote, y_smote)

# Save model and scaler
print("Saving model and scaler...")
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(xgbc, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
print("   - diabetes_model.pkl")
print("   - scaler.pkl")
