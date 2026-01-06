# 🏥 Diabetes Prediction Web Application

A beautiful, interactive web application for predicting diabetes risk using machine learning.

## 📋 Features

- 🎨 **Modern UI** - Clean, professional interface with custom styling
- 📊 **Real-time Predictions** - Instant diabetes risk assessment
- 💡 **Clinical Recommendations** - Personalized advice based on prediction
- 📈 **Model Metrics** - Display accuracy, confidence scores, and risk categories
- 🔒 **Validated Inputs** - Input validation with helpful tooltips

## 🚀 Quick Start

### Step 1: Save the Trained Model

First, run the model training script to save the model files:

```bash
python save_model.py
```

This will create:
- `diabetes_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Feature scaler

### Step 2: Launch the Web App

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 🛠️ Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn openpyxl
```

## 📱 Using the Web App

1. **Enter Patient Information:**
   - Number of Pregnancies
   - Glucose Level (mg/dL)
   - Blood Pressure (mm Hg)
   - Skin Thickness (mm)
   - Insulin Level (μU/mL)
   - BMI (Body Mass Index)
   - Age (years)

2. **Click "Predict Diabetes Risk"**

3. **View Results:**
   - Diagnosis (Diabetic/Non-Diabetic)
   - Confidence Score
   - Risk Category
   - Personalized Clinical Recommendations

## 🎯 For Presentation

### Demo Scenario 1: High Risk Patient
```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Age: 50
```
**Expected:** High Risk - Diabetic

### Demo Scenario 2: Low Risk Patient
```
Pregnancies: 1
Glucose: 85
Blood Pressure: 66
Skin Thickness: 29
Insulin: 0
BMI: 26.6
Age: 31
```
**Expected:** Low Risk - Non-Diabetic

### Demo Scenario 3: Moderate Case
```
Pregnancies: 2
Glucose: 110
Blood Pressure: 74
Skin Thickness: 25
Insulin: 94
BMI: 28.0
Age: 35
```

## 🌐 Deployment Options

### Option 1: Local Network (For Presentation)
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Access from other devices: `http://YOUR_IP:8501`

### Option 2: Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Deploy your app
4. Get a public URL like `https://yourapp.streamlit.app`

### Option 3: Heroku (Free Tier)
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT
   ```
2. Deploy to Heroku

### Option 4: AWS/Azure/GCP
Use container deployment with Docker

## 📊 Model Information

- **Algorithm:** XGBoost with ADASYN oversampling
- **Accuracy:** 81%
- **ROC-AUC Score:** 0.84
- **Training Samples:** 971 patients
- **Features:** 7 clinical measurements
- **Datasets:** PIMA Indian Diabetes + RTML Bangladesh

## 🎨 Customization

### Change Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify Colors
Edit the CSS in `app.py`:
```python
st.markdown("""
    <style>
    .main-header {
        color: #your-color;
    }
    </style>
""", unsafe_allow_html=True)
```

## ⚠️ Disclaimer

This is a prediction tool for educational and research purposes. It should **NOT** replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers.

## 📞 Support

For issues or questions, refer to:
- Streamlit Documentation: https://docs.streamlit.io
- Model Training Notebook: `Clean_work_PIMA+RTML_ADSYN.ipynb`

## 🎓 Perfect for Presentations

This web app is ideal for:
- ✅ Live demonstrations
- ✅ Interactive Q&A sessions
- ✅ Showcasing ML model capabilities
- ✅ Comparing different patient profiles
- ✅ Explaining model predictions to non-technical audiences

---

**Good luck with your presentation! 🚀**
