# 🏥 Diabetes Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13.3-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-81%25-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.84-blue.svg)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red.svg)

**An AI-Powered Diabetes Risk Assessment System with Professional Web Interface**

[Features](#-features) • [How It Works](#%EF%B8%8F-how-the-system-works) • [Installation](#-installation) • [Usage](#-usage) • [Technical Details](#-technical-details)

</div>

---

## 📊 Project Overview

This project combines advanced machine learning techniques with an intuitive web interface to predict diabetes risk based on clinical measurements. It merges two research datasets (PIMA Indian and RTML Bangladesh) and employs XGBoost with ADASYN oversampling to achieve **81% accuracy** and **0.84 ROC-AUC score**.

### 🎯 Key Achievements

- ✅ **971 Patient Records** from two validated medical datasets
- ✅ **81% Prediction Accuracy** with balanced training approach
- ✅ **Professional Web Interface** with dark theme and real-time predictions
- ✅ **Clinical Recommendations** personalized based on diagnosis
- ✅ **Production-Ready Deployment** using Streamlit

---

## ✨ Features

### 🤖 Machine Learning Capabilities
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Optimization:** Hyperparameter-tuned for maximum performance
- **Class Balancing:** ADASYN oversampling technique
- **Feature Scaling:** MinMaxScaler for normalized inputs
- **Model Persistence:** Pre-trained model ready for deployment

### 💻 Web Application
- **Interactive Input Form:** 7 clinical measurement fields
- **Real-Time Predictions:** Instant diabetes risk assessment
- **Confidence Scoring:** Probability distribution visualization
- **Animated Progress Bars:** Visual confidence level indicators
- **Responsive Design:** Modern dark theme with excellent readability
- **Clinical Recommendations:** Personalized health guidance

### 📈 Data Processing
- **Data Fusion:** Combined PIMA + RTML datasets
- **Missing Value Handling:** Statistical imputation methods
- **Feature Engineering:** Optimized 7-feature input vector
- **Balanced Dataset:** 50/50 class distribution after ADASYN

---

## ⚙️ How The System Works

### 1️⃣ Data Collection & Preparation

**Two Datasets Combined:**
```
├── PIMA Indian Diabetes: 768 samples
└── RTML Bangladesh: 203 samples
    = 971 total patient records
```

**7 Clinical Features:**
| Feature | Description | Unit |
|---------|-------------|------|
| Pregnancies | Number of times pregnant | Count |
| Glucose | Plasma glucose concentration | mg/dL |
| Blood Pressure | Diastolic blood pressure | mm Hg |
| Skin Thickness | Triceps skin fold thickness | mm |
| Insulin | 2-hour serum insulin | μU/mL |
| BMI | Body Mass Index | kg/m² |
| Age | Age in years | Years |

### 2️⃣ Data Preprocessing

**Class Imbalance Correction:**
```
Before ADASYN:
├── Non-Diabetic (Class 0): 468 samples (65%)
└── Diabetic (Class 1): 233 samples (35%)

After ADASYN:
├── Non-Diabetic (Class 0): 468 samples (50%)
└── Diabetic (Class 1): 424 samples (50%)
```

**Feature Normalization:**
- Applied MinMaxScaler to normalize all features to [0,1] range
- Ensures equal weight for all clinical measurements
- Prevents feature dominance in model training

### 3️⃣ Model Training

**XGBoost Configuration:**
```python
Hyperparameters:
{
    'colsample_bytree': 0.8,    # Use 80% features per tree
    'gamma': 1,                  # Minimum loss reduction
    'max_depth': 3,              # Tree depth (prevents overfitting)
    'min_child_weight': 1,       # Minimum samples in leaf
    'subsample': 0.8             # Use 80% data per iteration
}
```

**Training Pipeline:**
1. Split data: 80% training, 20% testing (stratified)
2. Apply ADASYN to training data only
3. Scale features using MinMaxScaler
4. Train XGBoost with optimized parameters
5. Validate on unseen test data
6. Save model and scaler as pickle files

### 4️⃣ Prediction Workflow

**User Input → Prediction:**
```python
# Example Patient Data
Input:
├── Pregnancies: 6
├── Glucose: 148
├── Blood Pressure: 72
├── Skin Thickness: 35
├── Insulin: 0
├── BMI: 33.6
└── Age: 50

# System Processing
1. Create DataFrame with user input
2. Apply scaler.transform() → normalize values [0,1]
3. Feed to model.predict_proba() → get probabilities
4. Extract confidence scores:
   - Non-Diabetic: 38.04%
   - Diabetic: 61.96% ✓ (Selected)
5. Display results with recommendations
```

---

## 🚀 Installation

### Prerequisites
- Python 3.13.3 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Diabetes-Prediction-Using-Machine-Learning
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- streamlit==1.31.0
- pandas==2.2.0
- numpy==1.26.3
- scikit-learn==1.5.2
- xgboost==2.1.3
- imbalanced-learn==0.12.0
- openpyxl==3.1.2

### Step 4: Train the Model (Optional)
```bash
python save_model.py
```
This will generate:
- `diabetes_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Fitted MinMaxScaler

---

## 💻 Usage

### Launch the Web Application
```bash
streamlit run app.py
```

The application will open in your browser at:
- **Local URL:** http://localhost:8501
- **Network URL:** http://192.168.1.33:8501

### Using the Prediction System

1. **Enter Patient Data:**
   - Fill in all 7 clinical measurement fields
   - Use the tooltips for guidance on normal ranges

2. **Click "Predict Diabetes Risk"**
   - The system processes the input instantly
   - Model applies scaling and makes prediction

3. **View Results:**
   - **Risk Assessment:** High Risk (Diabetic) or Low Risk (Non-Diabetic)
   - **Confidence Level:** Percentage with animated progress bar
   - **Clinical Recommendations:** Personalized health advice
   - **Detailed Analytics:** Risk score, category, and diagnosis

### Demo Scenarios

**High Risk Patient:**
```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Age: 50
→ Expected: Diabetic (High Confidence)
```

**Low Risk Patient:**
```
Pregnancies: 1
Glucose: 85
Blood Pressure: 66
Skin Thickness: 29
Insulin: 0
BMI: 26.6
Age: 31
→ Expected: Non-Diabetic (High Confidence)
```

**Moderate Risk Patient:**
```
Pregnancies: 2
Glucose: 110
Blood Pressure: 74
Skin Thickness: 25
Insulin: 94
BMI: 28.0
Age: 35
→ Expected: Variable (Medium Confidence)
```

---

## 📁 Project Structure

```
Diabetes-Prediction-Using-Machine-Learning/
│
├── app.py                                  # Streamlit web application
├── save_model.py                           # Model training script
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── README_WebApp.md                        # Web app documentation
│
├── diabetes_model.pkl                      # Trained XGBoost model
├── scaler.pkl                              # Fitted MinMaxScaler
│
├── diabetes.csv                            # PIMA dataset
├── RTML without Insulin.xlsx               # RTML dataset
│
├── Clean_work_PIMA+RTML_ADSYN.ipynb       # Main research notebook
├── Clean_work_PIMA+RTML.ipynb             # Notebook without ADASYN
├── Clean_work_Only_PIMA_ADSYN.ipynb       # PIMA with ADASYN
├── Clean_work_Only_PIMA_Without_Smote.ipynb # PIMA without balancing
└── Clean_work_Only_PIMA.ipynb             # PIMA baseline notebook
```

---

## 🔬 Technical Details

### Machine Learning Architecture

```
┌─────────────────────────────────────────────┐
│         USER INTERFACE (Streamlit)          │
│  • Input Form (7 clinical measurements)     │
│  • Prediction Button                        │
│  • Results Dashboard                        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│       DATA PREPROCESSING LAYER              │
│  • Convert to DataFrame                     │
│  • Apply MinMaxScaler                       │
│  • Normalize to [0,1] range                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         ML MODEL (XGBoost)                  │
│  • Trained on 971 balanced samples         │
│  • 81% accuracy, 0.84 AUC-ROC              │
│  • Returns probability distribution         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         PREDICTION ENGINE                   │
│  • Class 0 (Non-Diabetic) probability      │
│  • Class 1 (Diabetic) probability          │
│  • Select class with higher probability    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│       RESULT PRESENTATION                   │
│  • Color-coded risk level                  │
│  • Confidence percentage                    │
│  • Clinical recommendations                 │
│  • Interactive visualizations               │
└─────────────────────────────────────────────┘
```

### Model Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 81% | Overall correct predictions |
| **ROC-AUC Score** | 0.84 | Area under ROC curve (Excellent) |
| **Training Samples** | 971 | Total patient records |
| **Features** | 7 | Clinical measurements |
| **Classes** | 2 | Diabetic / Non-Diabetic |

### Data Processing Pipeline

**1. Data Loading:**
```python
# PIMA Dataset
pima_df = pd.read_csv('diabetes.csv')

# RTML Dataset
rtml_df = pd.read_excel('RTML without Insulin.xlsx')
```

**2. Feature Alignment:**
```python
# Drop DiabetesPedigreeFunction from PIMA
pima_df = pima_df.drop('DiabetesPedigreeFunction', axis=1)

# Both datasets now have 7 features + 1 target
```

**3. Missing Value Imputation:**
```python
# Impute RTML insulin missing values with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
rtml_df['Insulin'] = imputer.fit_transform(rtml_df[['Insulin']])
```

**4. Dataset Merging:**
```python
# Concatenate datasets
combined_df = pd.concat([pima_df, rtml_df], ignore_index=True)
# Total: 971 samples
```

**5. Class Balancing with ADASYN:**
```python
from imblearn.over_sampling import ADASYN

# Apply ADASYN to training data
adasyn = ADASYN(random_state=42)
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

# Result: 50/50 class distribution
```

**6. Feature Scaling:**
```python
from sklearn.preprocessing import MinMaxScaler

# Fit and transform training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)

# Transform test data (no fitting)
X_test_scaled = scaler.transform(X_test)
```

**7. Model Training:**
```python
from xgboost import XGBClassifier

# Initialize with optimized hyperparameters
xgb_model = XGBClassifier(
    colsample_bytree=0.8,
    gamma=1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train_scaled, y_train_balanced)
```

---

## 🎨 User Interface Features

### Design Elements
- **Dark Theme:** Professional gradient background (#0a0e27 to #0d1b2a)
- **Blue Accents:** Light blue (#93c5fd) for headers and highlights
- **Color Coding:**
  - 🔴 Red (#fca5a5) for diabetic/high risk
  - 🟢 Green (#86efac) for non-diabetic/low risk
- **Animations:** Smooth transitions and glowing effects
- **Responsive Layout:** Adapts to different screen sizes

### Interactive Components
1. **Input Form:** 7 fields with tooltips and validation
2. **Prediction Button:** Gradient design with hover effects
3. **Result Cards:** Animated progress bars for confidence
4. **Metric Cards:** Hover animations with detailed analytics
5. **Recommendations:** Personalized based on diagnosis
6. **Footer:** Model information and disclaimer

---

## 📊 Datasets Information

### PIMA Indian Diabetes Dataset
- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Samples:** 768 female patients
- **Age Range:** 21+ years
- **Features:** 8 clinical measurements (reduced to 7)
- **Target:** Diabetes outcome (0/1)

### RTML Bangladesh Dataset
- **Source:** Research Team Medical Laboratory
- **Samples:** 203 patients
- **Features:** 7 clinical measurements
- **Missing Data:** 109 insulin values (imputed)
- **Target:** Diabetes diagnosis (0/1)

### Combined Dataset Statistics
```
Total Samples: 971
Features: 7 clinical measurements
Target Distribution:
├── Class 0 (Non-Diabetic): 65% (before balancing)
└── Class 1 (Diabetic): 35% (before balancing)

After ADASYN:
├── Class 0: 50%
└── Class 1: 50%
```

---

## 🛠️ Development Tools

### Core Libraries
- **XGBoost:** Gradient boosting framework
- **Scikit-learn:** Machine learning utilities
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing
- **Imbalanced-learn:** ADASYN oversampling
- **Streamlit:** Web application framework

### Development Environment
- **Python Version:** 3.13.3
- **IDE:** VS Code / Jupyter Notebook
- **Version Control:** Git
- **Package Manager:** pip

---

## 📝 Clinical Recommendations

### For Diabetic Prediction (High Risk)
- 🏥 Immediate consultation with an endocrinologist required
- 📊 Monitor blood glucose levels regularly (daily if possible)
- 💊 Follow prescribed medication and dietary plan strictly
- 🏃 Regular physical activity: At least 30 minutes per day
- ⚖️ Weight management program - maintain healthy BMI
- 🔍 Regular check-ups: Monitor HbA1c levels quarterly
- 👣 Foot care: Check for injuries and maintain hygiene
- 👁️ Eye examinations: Annual diabetic retinopathy screening

### For Non-Diabetic Prediction (Low Risk)
- ✅ Maintain healthy lifestyle and current good habits
- 📅 Annual diabetes screening recommended
- 🥗 Balanced diet: Focus on whole grains, fruits, and vegetables
- 🏃 Regular exercise: 150 minutes of moderate activity per week
- 📊 Monitor risk factors: Keep BMI, blood pressure in check
- 🚫 Limit sugar intake and processed foods
- 💧 Stay hydrated: Drink adequate water daily
- 🧘 Stress management: Practice meditation or yoga

---

## ⚠️ Disclaimer

**IMPORTANT:** This AI-powered tool is designed for **educational and research purposes only**. It should **NOT** replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for accurate medical guidance and diabetes management.

The predictions are based on statistical models and may not account for all individual health factors. Use this tool as a supplementary risk assessment, not as a definitive diagnostic tool.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations
- UI/UX improvements

---

## 📄 License

This project is developed for educational purposes. Please ensure compliance with dataset usage terms and medical software regulations in your jurisdiction.

---

## 👨‍💻 Author

**Diabetes Prediction System**  
Developed as part of Machine Learning research combining PIMA Indian and RTML Bangladesh datasets.

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the project repository.

---

## 🙏 Acknowledgments

- **PIMA Indian Diabetes Dataset:** National Institute of Diabetes and Digestive and Kidney Diseases
- **RTML Bangladesh Dataset:** Research Team Medical Laboratory
- **XGBoost Library:** Tianqi Chen and Carlos Guestrin
- **Streamlit Framework:** Streamlit Inc.
- **ADASYN Algorithm:** Haibo He et al.

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**© 2026 Diabetes Prediction System | Powered by Machine Learning**

</div>
