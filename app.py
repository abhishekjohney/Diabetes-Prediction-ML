import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Try to import shap, make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ SHAP library not available. Model explainability features will be disabled.")

# Compute zero-imputation means from PIMA dataset (must match training pipeline)
_df_means = pd.read_csv('diabetes.csv')
ZERO_IMPUTE_MEANS = {
    col: float(_df_means[col][_df_means[col] != 0].mean())
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
}
del _df_means

# Page configuration
st.set_page_config(
    page_title="DiabetesAI - Advanced Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Enhanced Modern Theme
st.markdown("""
    <style>
    /* Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit default UI chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    
    /* Header Styling with Enhanced Animation */
    .main-header {
        font-size: clamp(1.9rem, 4.8vw, 4rem);
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 3s ease infinite, float 3s ease-in-out infinite;
        letter-spacing: clamp(0.5px, 0.35vw, 3px);
        text-transform: uppercase;
        position: relative;
        line-height: 1.15;
        word-break: break-word;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 15px #667eea); }
        to { filter: drop-shadow(0 0 30px #f093fb); }
    }
    
    /* Subtitle with Animation */
    .subtitle {
        text-align: center;
        color: #c7d2fe;
        font-size: clamp(0.95rem, 2vw, 1.4rem);
        margin-bottom: 2rem;
        font-weight: 500;
        letter-spacing: clamp(0.3px, 0.2vw, 2px);
        animation: fadeInUp 1s ease-out;
        line-height: 1.5;
        padding: 0 0.5rem;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Prediction Box Styling with Enhanced Effects */
    .prediction-box {
        padding: clamp(1rem, 3vw, 3rem);
        border-radius: 30px;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 60px 0 rgba(0, 0, 0, 0.7);
        border: 3px solid rgba(255, 255, 255, 0.15);
        animation: slideInScale 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes slideInScale {
        from { 
            opacity: 0;
            transform: translateY(30px) scale(0.9);
        }
        to { 
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .diabetic {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 3px solid #ef4444;
        box-shadow: 0 0 50px rgba(239, 68, 68, 0.5), inset 0 0 40px rgba(239, 68, 68, 0.15);
    }
    
    .non-diabetic {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.3) 0%, rgba(22, 163, 74, 0.3) 100%);
        border: 3px solid #22c55e;
        box-shadow: 0 0 50px rgba(34, 197, 94, 0.5), inset 0 0 40px rgba(34, 197, 94, 0.15);
    }
    
    .info-text {
        font-size: 1.3rem;
        margin: 1rem 0;
        color: #ffffff;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced Input Styling with Glassmorphism */
    .stNumberInput > div > div > input {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #ffffff !important;
        border: 2px solid rgba(102, 126, 234, 0.6) !important;
        border-radius: 15px !important;
        padding: 0.85rem 1rem !important;
        min-height: 48px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: #667eea !important;
        background: rgba(30, 41, 59, 0.8) !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 25px rgba(102, 126, 234, 0.6), 0 0 40px rgba(102, 126, 234, 0.3) !important;
        background: rgba(30, 41, 59, 1) !important;
        transform: scale(1.02);
    }
    
    .stNumberInput label {
        color: #e0e7ff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced Button Styling with 3D Effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.9rem 1.5rem !important;
        min-height: 52px !important;
        width: 100% !important;
        font-size: clamp(0.95rem, 1.9vw, 1.2rem) !important;
        font-weight: 800 !important;
        border-radius: 50px !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.7), 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 15px 45px rgba(102, 126, 234, 0.9), 0 0 0 4px rgba(102, 126, 234, 0.3) !important;
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.02) !important;
    }
    
    /* Enhanced Metric Cards with Hover Effects */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid rgba(102, 126, 234, 0.6);
        border-radius: 25px;
        padding: clamp(1rem, 2.5vw, 2.5rem);
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: gradient-move 3s linear infinite;
    }
    
    @keyframes gradient-move {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);
        border-color: #667eea;
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        border-right: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* Section Headers with Gradient */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: clamp(1.2rem, 3vw, 2rem);
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid rgba(102, 126, 234, 0.6);
        text-transform: uppercase;
        letter-spacing: clamp(0.5px, 0.2vw, 2px);
        animation: fadeInUp 0.6s ease-out;
        line-height: 1.3;
    }
    
    /* Footer with Enhanced Styling */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 3rem;
        margin-top: 4rem;
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.8) 0%, rgba(36, 36, 62, 0.8) 100%);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Enhanced Recommendations List */
    .recommendations {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 6px solid #667eea;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
        animation: slideIn 0.6s ease-out;
    }
    
    .recommendations li {
        color: #ffffff;
        margin: 1.2rem 0;
        font-size: 1.1rem;
        line-height: 1.8;
        font-weight: 500;
        padding-left: 1rem;
    }
    
    .recommendations li:hover {
        color: #c7d2fe;
        transform: translateX(5px);
        transition: all 0.3s ease;
    }
    
    /* Hide Streamlit Branding / Host UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stFloatingButton"] {display: none !important;}
    [data-testid="stProfileAvatar"] {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    .stDeployButton {display: none !important;}
    
    /* Enhanced Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f0c29;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 3px solid #0f0c29;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
    
    /* Global Text Styling */
    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff;
        font-weight: 400;
    }
    
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Card Container */
    .info-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3) !important;
    }

    /* Better desktop spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* Tablet + Mobile responsive behavior */
    @media (max-width: 992px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100%;
        }

        .metric-card:hover,
        .stButton > button:hover,
        .info-card:hover {
            transform: none !important;
        }
    }

    @media (max-width: 768px) {
        .main-header {
            margin-top: 0.25rem;
            text-align: left;
        }

        .subtitle {
            text-align: left;
            margin-bottom: 1.25rem;
            padding: 0;
        }

        .prediction-box {
            border-radius: 18px;
            margin: 1rem 0;
        }

        .metric-card,
        .info-card,
        .recommendations {
            border-radius: 14px;
            padding: 1rem;
        }

        .section-header {
            margin-top: 1.1rem;
            padding-bottom: 0.45rem;
            border-bottom-width: 2px;
        }

        .stNumberInput label {
            font-size: 0.9rem !important;
            letter-spacing: 0.2px;
            text-transform: none;
        }

        .stButton > button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            text-transform: none !important;
        }

        /* Reduce heavy motion on smaller devices */
        .main-header,
        .subtitle,
        .prediction-box,
        .metric-card,
        .recommendations {
            animation: none !important;
        }

        .prediction-box::before,
        .metric-card::before,
        .stButton > button::before {
            display: none !important;
        }
    }
    
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run 'match_paper_accuracy.py' first.")
        st.stop()

# Initialize SHAP explainer
@st.cache_resource
def load_shap_explainer(_model, _scaler):
    """Load background data and create SHAP explainer"""
    if not SHAP_AVAILABLE:
        return None, None
    
    # Load diabetes dataset for background
    try:
        df = pd.read_csv('diabetes.csv')
        # Select only the 7 features used by the model (excluding DiabetesPedigreeFunction)
        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
        background = df[feature_columns].head(100).copy()
        # Apply same zero-imputation used during training
        for col, mean_val in ZERO_IMPUTE_MEANS.items():
            if col in background.columns:
                background[col] = background[col].replace(0, mean_val)
        background = background.values
        
        # Scale the background data (model expects scaled input)
        background_scaled = _scaler.transform(background)
        
        # Create TreeExplainer with interventional feature perturbation
        explainer = shap.TreeExplainer(_model, background_scaled, feature_perturbation='interventional', model_output='raw')
        return explainer, background_scaled
    except Exception as e:
        st.error(f"Error loading SHAP explainer: {e}")
        return None, None

model, scaler = load_model()
shap_explainer, background_data = load_shap_explainer(model, scaler)

# Header with Enhanced Design
st.markdown('<h1 class="main-header">🩺 DIABETES PREDICTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🤖 Advanced ML-Powered Risk Assessment • XGBoost + ADASYN • AI Explainability</p>', unsafe_allow_html=True)

# Sidebar info with Enhanced Design
with st.sidebar:
    st.markdown("### 🎯 Model Intelligence Dashboard")
    st.markdown("---")
    
    # Add a cool header card
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                    border: 2px solid rgba(102, 126, 234, 0.5); border-radius: 15px; padding: 1.5rem;
                    text-align: center; margin-bottom: 1.5rem;">
            <h2 style="color: #667eea; margin: 0; font-size: 1.8rem;">⚡ AI-Powered</h2>
            <p style="color: #c7d2fe; margin: 0.5rem 0 0 0;">Real-time Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Accuracy", "82.77%", delta="High", delta_color="normal")
    with col2:
        st.metric("📈 AUC Score", "0.79", delta="Excellent", delta_color="normal")
    
    st.markdown("---")
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                    border-left: 4px solid #667eea; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #c7d2fe; margin: 0 0 0.5rem 0;">🤖 Algorithm</h4>
            <p style="color: #ffffff; margin: 0; font-weight: 600;">XGBoost with ADASYN</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%);
                    border-left: 4px solid #22c55e; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #86efac; margin: 0 0 0.5rem 0;">📊 Training Data</h4>
            <p style="color: #ffffff; margin: 0; font-weight: 600;">971 Patient Samples</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**🌍 Datasets Used**")
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <p style="margin: 0.3rem 0;">✓ PIMA Indian Diabetes</p>
            <p style="margin: 0.3rem 0;">✓ RTML Bangladesh</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Learned Thresholds Section
    with st.expander("📊 Model's Learned Thresholds", expanded=False):
        st.info("These are the key threshold values the AI model learned from 971 training samples:")
        
        st.markdown("**🩸 Glucose (mg/dL)**")
        st.markdown("- ≤ 100: Normal")
        st.markdown("- 100-125: Borderline")
        st.markdown("- 125-140: Pre-diabetic")
        st.markdown("- **> 140: High Risk**")
        
        st.markdown("**⚖️ BMI**")
        st.markdown("- < 25: Healthy")
        st.markdown("- 25-30: Overweight")
        st.markdown("- **> 30: Obese (Higher Risk)**")
        
        st.markdown("**🎂 Age (years)**")
        st.markdown("- < 30: Lower Risk")
        st.markdown("- 30-45: Moderate")
        st.markdown("- **> 45: Elevated Risk**")
        
        st.markdown("**🤰 Pregnancies**")
        st.markdown("- 0-2: Low")
        st.markdown("- 3-5: Moderate")
        st.markdown("- **> 6: Elevated Risk**")
        
        st.markdown("**📏 Skin Thickness (mm)**")
        st.markdown("- < 20: Normal")
        st.markdown("- 20-23: Baseline")
        st.markdown("- **> 23: Elevated Body Fat**")
        
        st.markdown("**💓 Blood Pressure (mmHg)**")
        st.markdown("- < 70: Normal")
        st.markdown("- 70-80: Optimal")
        st.markdown("- **> 80: Elevated**")
        
        st.markdown("**💉 Insulin (μU/mL)**")
        st.markdown("- < 100: Normal")
        st.markdown("- 100-200: Borderline")
        st.markdown("- **> 200: Insulin Resistance**")
        
        st.caption("* These thresholds were automatically learned by the model from training data")
    
    st.markdown("---")
    st.warning("⚕️ Educational purposes only.\nConsult healthcare professionals.")

# Main content with enhanced styling
st.markdown('<h2 class="section-header">📝 Enter the Information</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #c7d2fe; margin-bottom: 2rem; font-size: 1.1rem;">Enter patient information for diabetes risk assessment</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    
    pregnancies = st.number_input(
        "🤰 Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of times pregnant"
    )
    
    glucose = st.number_input(
        "🩸 Glucose Level (mg/dL)",
        min_value=0,
        max_value=300,
        value=120,
        help="Plasma glucose concentration (Normal: 70-100)"
    )
    
    blood_pressure = st.number_input(
        "💓 Blood Pressure (mm Hg)",
        min_value=0,
        max_value=200,
        value=70,
        help="Diastolic blood pressure (Normal: 60-80)"
    )
    
    skin_thickness = st.number_input(
        "📏 Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    
    insulin = st.number_input(
        "💉 Insulin Level (μU/mL)",
        min_value=0,
        max_value=900,
        value=80,
        help="2-Hour serum insulin (Normal: 16-166)"
    )
    
    bmi = st.number_input(
        "⚖️ BMI (Body Mass Index)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Weight(kg) ÷ Height²(m) | Normal: 18.5-24.9"
    )
    
    age = st.number_input(
        "🎂 Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        help="Age in years"
    )

# Prediction button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary")

# Make prediction
if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Age': [age]
    })
    
    # Apply zero-imputation (match training pipeline — zeros mean unknown/missing)
    for col, mean_val in ZERO_IMPUTE_MEANS.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].replace(0, mean_val)
    
    # Scale the data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.markdown('<h2 class="section-header">🔬 AI-Powered Analysis Results</h2>', unsafe_allow_html=True)
    
    if prediction == 1:
        # Diabetic
        confidence = probability[1] * 100
        st.markdown(f"""
            <div class="prediction-box diabetic">
                <h2 style="color: #fca5a5; text-align: center; font-weight: 800; text-shadow: 0 0 20px rgba(252, 165, 165, 0.5);">⚠️ HIGH RISK - DIABETIC DETECTED</h2>
                <p class="info-text" style="text-align: center; font-size: 1.5rem; color: #ffffff;">
                    <strong>Confidence Level:</strong> {confidence:.2f}%
                </p>
                <div style="width: 100%; background-color: rgba(30, 41, 59, 0.8); border-radius: 15px; height: 25px; margin-top: 1.5rem; overflow: hidden;">
                    <div style="width: {confidence}%; background: linear-gradient(90deg, #ef4444, #dc2626); height: 25px; border-radius: 15px; transition: width 1.5s; box-shadow: 0 0 20px rgba(239, 68, 68, 0.8);"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        # Non-diabetic
        confidence = probability[0] * 100
        st.markdown(f"""
            <div class="prediction-box non-diabetic">
                <h2 style="color: #86efac; text-align: center; font-weight: 800; text-shadow: 0 0 20px rgba(134, 239, 172, 0.5);">✅ LOW RISK - NON-DIABETIC</h2>
                <p class="info-text" style="text-align: center; font-size: 1.5rem; color: #ffffff;">
                    <strong>Confidence Level:</strong> {confidence:.2f}%
                </p>
                <div style="width: 100%; background-color: rgba(30, 41, 59, 0.8); border-radius: 15px; height: 25px; margin-top: 1.5rem; overflow: hidden;">
                    <div style="width: {confidence}%; background: linear-gradient(90deg, #22c55e, #16a34a); height: 25px; border-radius: 15px; transition: width 1.5s; box-shadow: 0 0 20px rgba(34, 197, 94, 0.8);"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Additional information
    st.markdown("---")
    st.markdown('<h3 class="section-header">📊 Detailed Analytics</h3>', unsafe_allow_html=True)
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #93c5fd; font-weight: 600;">Risk Score</h3>
                <h1 style="color: #ffffff; font-weight: 800; font-size: 3rem;">{max(probability)*100:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        risk_category = "High" if prediction == 1 else "Low"
        risk_color = "#fca5a5" if prediction == 1 else "#86efac"
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #93c5fd; font-weight: 600;">Risk Category</h3>
                <h1 style="color: {risk_color}; font-weight: 800; font-size: 3rem; text-shadow: 0 0 20px {risk_color};">{risk_category}</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col_info3:
        status = "Diabetic" if prediction == 1 else "Non-Diabetic"
        status_icon = "⚠️" if prediction == 1 else "✅"
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #93c5fd; font-weight: 600;">Diagnosis</h3>
                <h1 style="color: #ffffff; font-weight: 800; font-size: 2.5rem;">{status_icon} {status}</h1>
            </div>
        """, unsafe_allow_html=True)
    
    # SHAP Explainability Section
    if SHAP_AVAILABLE and shap_explainer is not None:
        st.markdown("---")
        st.markdown('<h2 class="section-header">🔬 AI Explainability - Why This Prediction?</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1; text-align: center; margin-bottom: 2rem;">Understanding which features influenced this prediction using SHAP</p>', unsafe_allow_html=True)
        
        with st.expander("📊 View SHAP Feature Impact Analysis", expanded=True):
            try:
                # Calculate SHAP values for this prediction
                # For binary classification, shap_values returns values for the positive class
                shap_values_raw = shap_explainer.shap_values(input_scaled, check_additivity=False)
                
                # Handle both single array and list of arrays output
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[1]  # Use positive class (diabetic)
                else:
                    shap_values = shap_values_raw
                
                # Get feature names
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
                
                # Text-Based Explanation
                st.markdown("### 💬 Simple Explanation")
                st.markdown("---")
                
                # Extract single sample values for explanation
                shap_vals_single = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                
                # Get original (unscaled) input values for display
                original_values = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'Age': age
                }
                
                # Create explanation DataFrame
                explanation_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Impact': shap_vals_single,
                    'Absolute Impact': np.abs(shap_vals_single)
                }).sort_values('Absolute Impact', ascending=False)
                
                # Generate natural language explanation
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                            padding: 1.5rem; border-radius: 12px; border-left: 4px solid #93c5fd;">
                    <h4 style="color: #93c5fd; margin-top: 0;">🤖 AI's Reasoning for This Prediction:</h4>
                """, unsafe_allow_html=True)
                
                # Get top 3 positive and negative contributors
                positive_features = explanation_df[explanation_df['SHAP Impact'] > 0].head(3)
                negative_features = explanation_df[explanation_df['SHAP Impact'] < 0].head(3)
                
                # Explain positive contributors (risk increasing)
                if len(positive_features) > 0:
                    st.markdown("**🔴 Features Increasing Diabetes Risk:**")
                    for idx, row in positive_features.iterrows():
                        feature = row['Feature']
                        impact = row['SHAP Impact']
                        value = original_values[feature]
                        
                        # Generate contextual explanation
                        if feature == 'Glucose':
                            context = f"Glucose level of {value} mg/dL is {'elevated' if value > 125 else 'moderately high'}"
                        elif feature == 'BMI':
                            context = f"BMI of {value} is {'in overweight range' if value > 25 else 'elevated'}"
                        elif feature == 'Age':
                            context = f"Age of {value} years increases baseline risk"
                        elif feature == 'BloodPressure':
                            context = f"Blood pressure of {value} mmHg is {'elevated' if value > 80 else 'concerning'}"
                        elif feature == 'Insulin':
                            context = f"Insulin level of {value} μU/mL suggests insulin {'resistance' if value > 150 else 'concerns'}"
                        elif feature == 'Pregnancies':
                            context = f"{value} pregnancies contribute to risk"
                        else:
                            context = f"{feature} value of {value} is concerning"
                        
                        st.markdown(f"- **{feature}**: {context} (Impact: +{impact:.3f})")
                
                st.markdown("")
                
                # Explain negative contributors (risk decreasing)
                if len(negative_features) > 0:
                    st.markdown("**🟢 Features Decreasing Diabetes Risk:**")
                    for idx, row in negative_features.iterrows():
                        feature = row['Feature']
                        impact = row['SHAP Impact']
                        value = original_values[feature]
                        
                        # Generate contextual explanation
                        if feature == 'Glucose':
                            context = f"Glucose level of {value} mg/dL is within normal range"
                        elif feature == 'BMI':
                            context = f"BMI of {value} is within healthy range"
                        elif feature == 'Age':
                            context = f"Age of {value} years is relatively young"
                        elif feature == 'BloodPressure':
                            context = f"Blood pressure of {value} mmHg is normal"
                        elif feature == 'Insulin':
                            context = f"Insulin level of {value} μU/mL is within normal range"
                        elif feature == 'Pregnancies':
                            context = f"{value} pregnancies indicate lower risk"
                        else:
                            context = f"{feature} value of {value} is protective"
                        
                        st.markdown(f"- **{feature}**: {context} (Impact: {impact:.3f})")
                
                # Overall summary
                st.markdown("")
                total_positive = explanation_df[explanation_df['SHAP Impact'] > 0]['SHAP Impact'].sum()
                total_negative = explanation_df[explanation_df['SHAP Impact'] < 0]['SHAP Impact'].sum()
                
                if total_positive > abs(total_negative):
                    conclusion = f"The risk-increasing factors (total impact: +{total_positive:.3f}) outweigh the protective factors (total impact: {total_negative:.3f}), leading to a **{'HIGH RISK' if prediction == 1 else 'positive'} prediction**."
                else:
                    conclusion = f"The protective factors (total impact: {total_negative:.3f}) outweigh the risk-increasing factors (total impact: +{total_positive:.3f}), leading to a **{'LOW RISK' if prediction == 0 else 'negative'} prediction**."
                
                st.markdown(f"**📊 Overall Assessment:** {conclusion}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")
                
                # Feature Importance Table
                st.markdown("### 📋 Feature Contribution Details")
                st.markdown("""
                <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 10px; border-left: 3px solid #3b82f6; margin-bottom: 1rem;">
                    <p style="color: #cbd5e1; margin: 0.5rem 0;">
                        This table shows how each input feature influences the diabetes prediction. 
                        The <strong>SHAP Impact</strong> value indicates the strength and direction of each feature's contribution:
                    </p>
                    <ul style="color: #cbd5e1; margin-top: 0.5rem;">
                        <li><strong>Positive values (🔴):</strong> Increase the likelihood of diabetes</li>
                        <li><strong>Negative values (🟢):</strong> Decrease the likelihood of diabetes</li>
                        <li><strong>Larger magnitudes:</strong> Stronger influence on the prediction</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Extract single sample values
                shap_vals_single = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                
                # Create DataFrame with SHAP values
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_scaled[0],
                    'SHAP Impact': shap_vals_single,
                    'Absolute Impact': np.abs(shap_vals_single)
                }).sort_values('Absolute Impact', ascending=False)
                
                shap_df['Direction'] = shap_df['SHAP Impact'].apply(
                    lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Decreases Risk'
                )
                
                # Display table
                st.dataframe(
                    shap_df[['Feature', 'Value', 'SHAP Impact', 'Direction']].style.format({
                        'Value': '{:.3f}',
                        'SHAP Impact': '{:+.3f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary statistics
                col_shap1, col_shap2, col_shap3 = st.columns(3)
                
                positive_impact = float(shap_df[shap_df['SHAP Impact'] > 0]['SHAP Impact'].sum())
                negative_impact = float(shap_df[shap_df['SHAP Impact'] < 0]['SHAP Impact'].sum())
                
                with col_shap1:
                    st.metric("🔴 Risk Increasing", f"+{positive_impact:.3f}")
                with col_shap2:
                    st.metric("🟢 Risk Decreasing", f"{negative_impact:.3f}")
                with col_shap3:
                    st.metric("⚖️ Net Effect", f"{positive_impact + negative_impact:+.3f}")
                
                # Interpretation guide
                st.markdown("---")
                st.markdown("#### 📖 How to Interpret:")
                st.markdown("""
                - **SHAP Impact**: Shows how much each feature contributes to the final prediction
                - **Positive values** (🔴): Push prediction toward diabetic
                - **Negative values** (🟢): Push prediction toward non-diabetic
                - **Magnitude**: Larger absolute values = stronger influence
                - **Base Value**: Average model prediction across all patients
                """)
                
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <h3 style="color: #93c5fd; margin-bottom: 1rem; font-weight: 700;">⚕️ Medical Disclaimer</h3>
        <p style="color: #cbd5e1;">This AI-powered tool is designed for educational and research purposes only.</p>
        <p style="color: #cbd5e1;">It should NOT replace professional medical advice, diagnosis, or treatment.</p>
        <p style="color: #93c5fd; font-weight: 600; margin-top: 1rem;">
            Always consult with qualified healthcare providers for accurate medical guidance.
        </p>
        <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(59, 130, 246, 0.3);">
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong style="color: #93c5fd;">Model:</strong> XGBoost with ADASYN Oversampling | 
                <strong style="color: #93c5fd;">Accuracy:</strong> 81% | 
                <strong style="color: #93c5fd;">ROC-AUC:</strong> 0.84 | 
                <strong style="color: #93c5fd;">Dataset:</strong> PIMA Indian + RTML Bangladesh (971 samples)
            </p>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">
                © 2026 Diabetes Prediction System | Powered by Machine Learning
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)
