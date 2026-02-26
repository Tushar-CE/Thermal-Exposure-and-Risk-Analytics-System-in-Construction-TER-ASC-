import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from PIL import Image
warnings.filterwarnings('ignore')

# Only Neural Network - Lighter version
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import hashlib
import time
from functools import lru_cache

# Page configuration
st.set_page_config(
    page_title="TR Heat Stress Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (minimized - removed duplicates, combined selectors)
st.markdown("""
    <style>
    .stApp {background: linear-gradient(145deg,#0f2027 0%,#203a43 50%,#2c5364 100%);}
    .main-header {font-size:2.8rem;font-weight:700;text-align:center;color:white;padding:1.5rem;background:rgba(255,255,255,0.08);backdrop-filter:blur(12px);border-radius:20px;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.1);box-shadow:0 8px 32px rgba(0,0,0,0.2);}
    .main-header-small {font-size:1.5rem;font-weight:600;text-align:center;color:white;padding:.8rem;background:rgba(255,255,255,0.08);backdrop-filter:blur(12px);border-radius:12px;margin:1rem 0;}
    .section-header {font-size:1.6rem;font-weight:600;color:white;padding:.8rem 1rem;margin:1.2rem 0 1rem 0;background:rgba(255,255,255,0.05);backdrop-filter:blur(5px);border-radius:12px;border-left:5px solid #4ECDC4;}
    .pred-card {background:rgba(255,255,255,0.07);backdrop-filter:blur(8px);padding:.8rem;border-radius:12px;text-align:center;color:white;border:1px solid rgba(255,255,255,0.1);transition:.3s;margin:.3rem 0;box-shadow:0 4px 12px rgba(0,0,0,0.1);}
    .pred-card:hover {transform:translateY(-2px);background:rgba(255,255,255,0.12);}
    .pred-card h3 {margin:0;font-size:.9rem;font-weight:500;color:rgba(255,255,255,0.9);text-transform:uppercase;}
    .pred-card h1 {margin:.2rem 0;font-size:1.8rem;font-weight:700;}
    .level-box-very-hot {background:linear-gradient(135deg,#8B0000,#FF0000);padding:1.2rem;border-radius:16px;color:white;text-align:center;margin:.8rem 0;border:1px solid rgba(255,255,255,0.15);box-shadow:0 6px 20px rgba(0,0,0,0.2);}
    .level-box-hot {background:linear-gradient(135deg,#FF4500,#FF8C00);padding:1.2rem;border-radius:16px;color:white;text-align:center;margin:.8rem 0;}
    .level-box-warm {background:linear-gradient(135deg,#FFA500,#FFD700);padding:1.2rem;border-radius:16px;color:white;text-align:center;margin:.8rem 0;}
    .level-box-slightly-warm {background:linear-gradient(135deg,#FFD700,#FFFF00);padding:1.2rem;border-radius:16px;color:#333;text-align:center;margin:.8rem 0;}
    .level-box-comfortable {background:linear-gradient(135deg,#006400,#228B22);padding:1.2rem;border-radius:16px;color:white;text-align:center;margin:.8rem 0;}
    .level-box h2 {margin:0;font-size:2rem;font-weight:700;}
    .rec-card {background:rgba(255,255,255,0.07);backdrop-filter:blur(8px);padding:1.2rem;border-radius:16px;color:white;border:1px solid rgba(255,255,255,0.1);height:100%;transition:.3s;}
    .rec-card:hover {background:rgba(255,255,255,0.12);transform:translateY(-3px);}
    .rec-card h3 {margin-top:0;margin-bottom:1rem;font-size:1.3rem;border-bottom:2px solid rgba(255,255,255,0.2);padding-bottom:.5rem;}
    [data-testid="stSidebar"] {background:linear-gradient(195deg,#1a2f3f 0%,#1e3a4a 100%);border-right:1px solid rgba(255,255,255,0.1);}
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {color:white;}
    .image-container {background:rgba(255,255,255,0.05);backdrop-filter:blur(8px);border-radius:16px;padding:1rem;border:1px solid rgba(255,255,255,0.1);margin:.5rem 0;}
    .section-title {background:linear-gradient(90deg,#1e3c72,#2a5298);padding:10px 18px;border-radius:8px;text-align:center;color:white;font-size:18px;font-weight:600;margin-top:18px;margin-bottom:8px;box-shadow:0 3px 8px rgba(0,0,0,0.2);}
    .kpi-card {background-color:rgba(255,255,255,0.08);padding:6px 8px;border-radius:8px;text-align:center;margin:4px 0;}
    .kpi-title {font-size:13px;font-weight:600;margin-bottom:2px;}
    .kpi-value {font-size:18px;font-weight:700;margin:2px 0;}
    .kpi-desc {font-size:10px;opacity:.75;}
    .center-thermal-box {max-width:500px;margin:8px auto;padding:12px 16px;border-radius:10px;text-align:center;}
    div[role="tablist"] button {font-size:17px !important;padding:12px 20px !important;border-radius:10px !important;margin-right:5px !important;background-color:#f0f2f6 !important;color:#333 !important;font-weight:600;transition:.3s;}
    div[role="tablist"] button[aria-selected="true"] {background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);color:white !important;box-shadow:0 4px 12px rgba(0,0,0,0.25);transform:scale(1.05);}
    .map-section-title {background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);padding:10px 16px;border-radius:8px;text-align:center;color:white;font-size:18px;font-weight:600;margin:5px 0 12px;box-shadow:0 3px 8px rgba(0,0,0,0.2);}
    .map-container {background:rgba(255,255,255,0.06);padding:12px;border-radius:10px;text-align:center;max-width:600px;margin:0 auto;}
    @media (max-width:768px) {
        .main-header {font-size:1.8rem !important;padding:1rem !important;}
        .section-header {font-size:1.2rem !important;padding:.5rem !important;}
        .pred-card h3 {font-size:.7rem !important;}
        .pred-card h1 {font-size:1.2rem !important;}
        .kpi-title {font-size:11px !important;}
        .kpi-value {font-size:14px !important;}
        .rec-card {font-size:12px !important;padding:.8rem !important;}
        div[role="tablist"] button {font-size:14px !important;padding:8px 12px !important;}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h3 class="main-header">Thermal Exposure and Risk Analytics System in Construction (TER-ASC) </h3>', unsafe_allow_html=True)

# File paths
CSV_PATH = "EXBD.csv"
PET_MAP_PATH = "PET_Max_April.jpg"

# Work Type Data (unchanged)
work_data = {
    "Rest (R)": {"M": 115, "PET_AL": 35, "description": "Sitting, light activities", "base_factor": 0.15},
    "Light (LW)": {"M": 180, "PET_AL": 35.5, "description": "Standing, light hand work", "base_factor": 0.20},
    "Moderate (MW)": {"M": 300, "PET_AL": 32, "description": "Walking, moderate lifting", "base_factor": 0.25},
    "Heavy (HW)": {"M": 415, "PET_AL": 31, "description": "Heavy lifting, shoveling", "base_factor": 0.28},
    "Very Heavy (VHW)": {"M": 520, "PET_AL": 30, "description": "Very intense labor", "base_factor": 0.30}
}

activity_to_work = {
    2.1: "Light (LW)", 2.2: "Light (LW)", 2.6: "Moderate (MW)",
    3.2: "Heavy (HW)", 3.8: "Heavy (HW)", 4.0: "Very Heavy (VHW)"
}

# OPTIMIZATION 1: Enhanced caching with TTL and file hash checking
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_file_hash(filepath):
    """Get file hash for cache invalidation"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_training_data():
    """Load training data with optimized loading"""
    try:
        if not os.path.exists(CSV_PATH):
            # Return None but don't show error - will use sample data
            return None, None
        
        # OPTIMIZATION 2: Only load necessary columns
        usecols = ['T', 'RH', 'WS', 'PET', 'PMV', 'PPD', 'SET', 'RWS', 'CE', 
                   'Height', 'PETH', 'PMVH']
        df = pd.read_csv(CSV_PATH, encoding='utf-8', usecols=usecols)
        
        # Create dataframes efficiently
        hs_df = pd.DataFrame({
            'T(0C)': pd.to_numeric(df['T'], errors='coerce'),
            'RH(%)': pd.to_numeric(df['RH'], errors='coerce'),
            'WS(m/s)': pd.to_numeric(df['WS'], errors='coerce'),
            'PET(0C)': pd.to_numeric(df['PET'], errors='coerce'),
            'PMV': pd.to_numeric(df['PMV'], errors='coerce'),
            'PPD(%)': pd.to_numeric(df['PPD'], errors='coerce'),
            'SET (0C)': pd.to_numeric(df['SET'], errors='coerce'),
            'RWS(m/s)': pd.to_numeric(df['RWS'], errors='coerce'),
            'CE(0C)': pd.to_numeric(df['CE'], errors='coerce')
        }).dropna()
        
        bh_df = pd.DataFrame({
            'Height(m)': pd.to_numeric(df['Height'], errors='coerce'),
            'PET(0C)': pd.to_numeric(df['PETH'], errors='coerce'),
            'PMV': pd.to_numeric(df['PMVH'], errors='coerce')
        }).dropna()
        
        if len(bh_df) > 0:
            bh_df = bh_df[bh_df['Height(m)'] > 0].sort_values('Height(m)')
        
        return hs_df, bh_df
        
    except Exception:
        return None, None

# OPTIMIZATION 3: Lazy loading with progress indicator
@st.cache_resource(ttl=7200, show_spinner=False)  # Cache model for 2 hours
def train_lightweight_neural_network(X_scaled, y_dict):
    """Train smaller, faster neural networks"""
    models = {}
    scores = {}
    
    for target, y in y_dict.items():
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # OPTIMIZATION 4: Smaller network (128-64-32 instead of 256-128-64-32)
        nn = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # Reduced size
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,  # Larger batch size for faster training
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,  # Reduced iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,  # Reduced patience
            verbose=False
        )
        
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        scores[target] = r2_score(y_test, y_pred)
        models[target] = nn
    
    return models, scores

# OPTIMIZATION 5: Load data with minimal overhead
with st.spinner("🔄 Loading models..."):
    hs_df, bh_df = load_training_data()

# Generate sample data if needed (optimized)
if hs_df is None or len(hs_df) == 0:
    with st.spinner("📊 Generating sample data..."):
        np.random.seed(42)
        n_samples = 300  # Reduced from 500
        
        T_range = np.random.uniform(25, 40, n_samples)
        RH_range = np.random.uniform(40, 80, n_samples)
        WS_range = np.random.uniform(0.5, 5, n_samples)
        
        # Vectorized calculations
        PET_calc = T_range + 5 - 0.8 * WS_range + 0.015 * (RH_range - 40) + np.random.normal(0, 1, n_samples)
        PET_calc = np.clip(PET_calc, 25, 48)
        
        PMV_calc = 1.5 + (T_range - 25) * 0.12 - 0.15 * WS_range + 0.02 * (RH_range - 40) + np.random.normal(0, 0.2, n_samples)
        PMV_calc = np.clip(PMV_calc, 1, 4)
        
        hs_df = pd.DataFrame({
            'T(0C)': T_range, 'RH(%)': RH_range, 'WS(m/s)': WS_range,
            'PET(0C)': PET_calc, 'PMV': PMV_calc,
            'PPD(%)': 5 + 85 * (1 - np.exp(-0.5 * (PMV_calc - 1))),
            'SET (0C)': PET_calc - 1 + np.random.normal(0, 0.5, n_samples),
            'RWS(m/s)': WS_range * 0.8 + np.random.normal(0, 0.2, n_samples),
            'CE(0C)': 2 + 0.5 * WS_range + np.random.normal(0, 0.3, n_samples)
        })
        
        # Height data
        heights = [0, 20, 40, 60, 80, 100]  # Reduced points
        bh_df = pd.DataFrame({
            'Height(m)': heights,
            'PET(0C)': [PET_calc.mean() - i * 0.07 for i in range(len(heights))],
            'PMV': [PMV_calc.mean() - i * 0.03 for i in range(len(heights))]
        })

# Sidebar (unchanged but optimized)
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")
    st.markdown("---")
    
    st.markdown("### 🌤️ Weather")
    col1, col2 = st.columns(2)
    with col1:
        T = st.number_input("Temp (°C)", 20.0, 50.0, 34.0, 0.1)
    with col2:
        RH = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    
    WS = st.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    st.markdown("---")
    
    st.markdown("### 👕 Clothing")
    clo = st.select_slider(
        "Clothing insulation (clo)",
        options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00],
        value=0.57
    )
    st.markdown("---")
    
    st.markdown("### 💪 Activity")
    met = st.select_slider(
        "Metabolic rate (met)",
        options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0],
        value=3.2
    )
    st.markdown("---")
    
    st.markdown("### 🏗️ Height")
    height = st.slider("Working height (m)", 0, 100, 0, 1)
    st.markdown("---")
    
    st.markdown("### 📊 Productivity")
    baseline_productivity = st.number_input("Baseline Productivity (units/hr)", min_value=1.0, value=100.0, step=5.0)
    st.markdown("---")
    
    st.markdown("### 🤖 Model")
    st.info("Neural Network (AI Based ML)")

# Prepare training data
features = ['T(0C)', 'RH(%)', 'WS(m/s)']
targets = ['PET(0C)', 'PMV', 'PPD(%)', 'SET (0C)', 'RWS(m/s)', 'CE(0C)']

X = hs_df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare targets dict for training
y_dict = {target: hs_df[target].values for target in targets if target in hs_df.columns}

# Train models
models, model_scores = train_lightweight_neural_network(X_scaled, y_dict)

# OPTIMIZATION 6: Physics-based fallback function
@lru_cache(maxsize=128)
def physics_based_prediction(T, RH, WS, target):
    """Cached physics-based predictions"""
    if target == 'PET(0C)':
        return T + 5 + 0.015*(RH-40) - 0.8*WS
    elif target == 'PMV':
        return 1.5 + (T-25)*0.12 - 0.15*WS + 0.02*(RH-40)
    elif target == 'PPD(%)':
        return 50
    elif target == 'SET (0C)':
        return T + 3
    elif target == 'RWS(m/s)':
        return WS * 0.8
    elif target == 'CE(0C)':
        return 2 + 0.5*WS
    return 0

# Make predictions efficiently
input_data = np.array([[T, RH, WS]])
input_scaled = scaler.transform(input_data)

predictions = {}
for target in targets:
    if target in models:
        predictions[target] = models[target].predict(input_scaled)[0]
    else:
        predictions[target] = physics_based_prediction(T, RH, WS, target)

# Apply calibrated adjustments
predictions['PET(0C)'] = np.clip(predictions['PET(0C)'] + clo * 0.5 + (met - 2.0) * 0.3, 20, 50)
predictions['PMV'] = np.clip(predictions['PMV'] + clo * 0.3 + (met - 2.0) * 0.2, 0, 3.5)
predictions['PPD(%)'] = np.clip(predictions['PPD(%)'] + clo * 2 + (met - 2.0) * 1.5, 5, 90)

# OPTIMIZATION 7: Only show model performance occasionally
if st.sidebar.button("📊 Show Model Performance"):
    with st.sidebar.expander("Model Performance", expanded=True):
        for target, score in model_scores.items():
            st.metric(f"{target} R²", f"{score:.3f}")

# Display predictions (unchanged - keep all features)
st.markdown('<div class="section-title">📊 Predicted Outputs (Ground Level)</div>', unsafe_allow_html=True)

kpi_data = [
    ("PET", f'{predictions["PET(0C)"]:.1f}°C', "Physiological Equivalent Temp", "#FF6B6B"),
    ("PMV", f'{predictions["PMV"]:.2f}', "Predicted Mean Vote", "#4ECDC4"),
    ("PPD", f'{predictions["PPD(%)"]:.1f}%', "% Dissatisfied", "#FFD93D"),
    ("SET", f'{predictions["SET (0C)"]:.1f}°C', "Standard Effective Temp", "#96CEB4"),
    ("RWS", f'{predictions["RWS(m/s)"]:.1f} m/s', "Relative Wind Speed", "#FFB347"),
    ("CE",  f'{predictions["CE(0C)"]:.1f}°C', "Cooling Effect", "#45B7D1"),
]

for i in range(0, len(kpi_data), 3):
    cols = st.columns(3, gap="small")
    for col, (title, value, desc, color) in zip(cols, kpi_data[i:i+3]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="color:{color};">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# Thermal sensation (unchanged)
st.markdown('<div class="section-title">🌡️ Thermal Sensation Scale</div>', unsafe_allow_html=True)

pet = predictions["PET(0C)"]
pmv = predictions["PMV"]

if pet > 41.0:
    pet_class, pet_desc, pet_icon, pet_box = "VERY HOT", "Extreme heat stress - Unsafe for any physical work", "🔥", "level-box-very-hot"
    work_recommendation, productivity_impact = "🚫 SUSPEND ALL OUTDOOR WORK", "25-30% loss"
elif pet > 35.0:
    pet_class, pet_desc, pet_icon, pet_box = "HOT", "High heat stress - Significant strain", "🌡️", "level-box-hot"
    work_recommendation, productivity_impact = "⚠️ LIMIT WORK TO 45 MIN SESSIONS", "15-25% loss"
elif pet > 29.0:
    pet_class, pet_desc, pet_icon, pet_box = "WARM", "Moderate heat stress - Reduced work capacity", "☀️", "level-box-warm"
    work_recommendation, productivity_impact = "✓ NORMAL WORK WITH BREAKS", "5-15% loss"
elif pet > 23.0:
    pet_class, pet_desc, pet_icon, pet_box = "SLIGHTLY WARM", "Slight heat stress - Comfortable", "⛅", "level-box-slightly-warm"
    work_recommendation, productivity_impact = "✓ NORMAL WORK", "0-5% loss"
else:
    pet_class, pet_desc, pet_icon, pet_box = "COMFORTABLE", "No heat stress - Optimal conditions", "✅", "level-box-comfortable"
    work_recommendation, productivity_impact = "✓ NORMAL WORK", "0% loss"

if pmv >= 3.0:
    pmv_class = "EXTREME DISCOMFORT"
elif pmv >= 2.5:
    pmv_class = "VERY DISCOMFORT"
elif pmv >= 2.0:
    pmv_class = "MODERATE DISCOMFORT"
elif pmv >= 1.5:
    pmv_class = "DISCOMFORT"
elif pmv >= 1.0:
    pmv_class = "SLIGHT DISCOMFORT"
else:
    pmv_class = "COMFORTABLE"

st.markdown(f'''
<div class="{pet_box} center-thermal-box">
    <h2>{pet_icon} {pet_class}</h2>
    <h2>{pmv_class}</h2>
    <p>{pet_desc}</p>
    <p style="font-weight:600;">{work_recommendation}</p>
    <p>Productivity Impact: {productivity_impact} (varies by work type)</p>
</div>
''', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ PET Map", "📈 Vertical Analysis", "📊 Productivity", "📋 Safety Guide"])

with tab1:
    st.markdown('<div class="map-section-title">Maximum PET Distribution (April 2021–2024)</div>', unsafe_allow_html=True)
    
    # Center the map using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="map-container" style="text-align: center;">', unsafe_allow_html=True)
        if os.path.exists(PET_MAP_PATH):
            pet_map = Image.open(PET_MAP_PATH)
            # Larger size: 600x450 instead of 400x300
            pet_map.thumbnail((600, 450))
            st.image(pet_map, use_column_width=True)
        else:
            st.warning("PET map image not found")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">📈 Vertical Heat Stress Profile</div>', unsafe_allow_html=True)
    
    ground_pet = predictions["PET(0C)"]
    ground_pmv = predictions["PMV"]
    
    if 'Height(m)' in bh_df.columns and 'PET(0C)' in bh_df.columns and len(bh_df) > 0:
        heights_original = bh_df['Height(m)'].values
        pet_pattern = bh_df['PET(0C)'].values
        pmv_pattern = bh_df['PMV'].values if 'PMV' in bh_df.columns else pet_pattern * 0.08
        
        # Ensure monotonic decrease
        for i in range(1, len(pet_pattern)):
            if pet_pattern[i] > pet_pattern[i-1]:
                pet_pattern[i] = pet_pattern[i-1] - 0.1
            if pmv_pattern[i] > pmv_pattern[i-1]:
                pmv_pattern[i] = pmv_pattern[i-1] - 0.01
        
        pet_relative = pet_pattern / pet_pattern[0]
        pmv_relative = pmv_pattern / pmv_pattern[0]
        
        pet_profile = ground_pet * pet_relative
        pmv_profile = ground_pmv * pmv_relative
        
        # OPTIMIZATION 9: Fewer points for smoother rendering
        heights_smooth = np.linspace(0, 100, 100)  # Reduced from 200
        pet_smooth = np.interp(heights_smooth, heights_original, pet_profile)
        pmv_smooth = np.interp(heights_smooth, heights_original, pmv_profile)
        
        fig_vertical = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_vertical.add_trace(
            go.Scatter(x=heights_smooth, y=pet_smooth, name="PET", 
                      line=dict(color='#FF6B6B', width=3), mode='lines'),
            secondary_y=False
        )
        
        fig_vertical.add_trace(
            go.Scatter(x=heights_smooth, y=pmv_smooth, name="PMV", 
                      line=dict(color='#4ECDC4', width=3), mode='lines'),
            secondary_y=True
        )
        
        # Threshold zones
        fig_vertical.add_hrect(y0=41, y1=50, line_width=0, fillcolor="red", opacity=0.1, secondary_y=False,
                              annotation_text="Very Hot", annotation_position="top right")
        fig_vertical.add_hrect(y0=35, y1=41, line_width=0, fillcolor="orange", opacity=0.1, secondary_y=False,
                              annotation_text="Hot", annotation_position="top right")
        fig_vertical.add_hrect(y0=29, y1=35, line_width=0, fillcolor="yellow", opacity=0.1, secondary_y=False,
                              annotation_text="Warm", annotation_position="top right")
        
        # Ground marker
        fig_vertical.add_trace(
            go.Scatter(x=[0], y=[ground_pet], mode='markers', 
                      marker=dict(size=14, color='white', symbol='circle'),
                      name=f'Ground: {ground_pet:.1f}°C'),
            secondary_y=False
        )
        
        if height > 0:
            pet_at_height = np.interp(height, heights_smooth, pet_smooth)
            fig_vertical.add_vline(x=height, line_dash="dash", line_color="white", line_width=1.5)
            fig_vertical.add_trace(
                go.Scatter(x=[height], y=[pet_at_height], mode='markers',
                          marker=dict(size=12, color='yellow', symbol='star'),
                          name=f'Working Height: {height}m'),
                secondary_y=False
            )
        
        pet_at_100 = np.interp(100, heights_smooth, pet_smooth)
        reduction = ground_pet - pet_at_100
        
        fig_vertical.update_layout(
            title=f"Vertical Profile - Ground PET: {ground_pet:.1f}°C → 100m: {pet_at_100:.1f}°C (Reduction: {reduction:.1f}°C)",
            xaxis_title="Working Height (m)",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            hovermode='x unified'
        )
        
        fig_vertical.update_xaxes(gridcolor='rgba(255,255,255,0.1)', dtick=10)
        fig_vertical.update_yaxes(title_text="PET (°C)", secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
        fig_vertical.update_yaxes(title_text="PMV", secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig_vertical, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">📊 Loss by Work Type</div>', unsafe_allow_html=True)
    
    current_work_key = activity_to_work.get(met, "Heavy (HW)")
    
    @lru_cache(maxsize=32)
    def calculate_productivity_cached(pet_value, work_type_key, baseline):
        """Cached productivity calculation"""
        work = work_data[work_type_key]
        PET_AL = work["PET_AL"]
        base_factor = work["base_factor"]
        
        if pet_value <= PET_AL:
            return 0
        else:
            delta_pet = pet_value - PET_AL
            PL = 30 * (1 - np.exp(-base_factor * delta_pet))
            return min(PL, 30)
    
    current_pl = calculate_productivity_cached(pet, current_work_key, baseline_productivity)
    
    # Bar chart
    work_types = list(work_data.keys())
    loss_by_work = [calculate_productivity_cached(pet, wt, baseline_productivity) for wt in work_types]
    
    colors_work = ['#4ECDC4' if l < 5 else '#FFA500' if l < 15 else '#FF4500' if l < 25 else '#8B0000' for l in loss_by_work]
    
    fig = go.Figure(data=[
        go.Bar(x=work_types, y=loss_by_work, marker_color=colors_work,
               text=[f"{l:.1f}%" for l in loss_by_work], textposition='auto')
    ])
    
    fig.update_layout(
        title=f"Productivity Loss by Work Type at PET={pet:.1f}°C",
        xaxis_title="Work Type",
        yaxis_title="Loss (%)",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if current_pl == 0:
        st.success(f"✅ Current Work ({current_work_key}): No productivity loss")
    elif current_pl < 5:
        st.info(f"ℹ️ Current Work ({current_work_key}): {current_pl:.1f}% loss - Minimal impact")
    elif current_pl < 15:
        st.warning(f"⚠️ Current Work ({current_work_key}): {current_pl:.1f}% loss - Moderate impact")
    elif current_pl < 25:
        st.error(f"🔴 Current Work ({current_work_key}): {current_pl:.1f}% loss - High impact")
    else:
        st.error(f"🚫 Current Work ({current_work_key}): {current_pl:.1f}% loss - Severe impact")

with tab4:
    st.markdown('<h4 class="section-header">Personalized Safety Guidelines</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="rec-card"><h3 style="color: #FF6B6B;">🏗️ For Safety Engineers/Site Managers</h3>', unsafe_allow_html=True)
        if pet > 41:
            st.markdown(f"**Heat Risk:** 🔥 VERY HOT (PET: {pet:.1f}°C)\n**Activity Level:** {current_work_key} ({met:.1f} met)\n\n**Work Schedule:**\n- 🚫 SUSPEND ALL OUTDOOR WORK\n- ✅ Only emergency tasks in shaded areas\n- 🔄 If critical: 15 min work / 45 min rest in AC\n\n**Hydration:**\n- 💧 ORS stations every 15m\n- 💧 Mandatory drinking every 15 min\n- ❄️ Ice-cooled water only\n\n**Cooling:**\n- 🌳 AC rest areas mandatory\n- ❄️ Cooling vests required for {current_work_key}\n- 🏥 Medical team on site\n\n**Productivity:** {current_pl:.1f}% loss expected")
        elif pet > 35:
            st.markdown(f"**Heat Risk:** 🌡️ HOT (PET: {pet:.1f}°C)\n**Activity Level:** {current_work_key} ({met:.1f} met)\n\n**Work Schedule:**\n- ⚠️ Limit {current_work_key} work to 45 min sessions\n- 🔄 45 min work / 15 min rest\n- 🌅 Schedule heavy work before 10 AM\n\n**Hydration:**\n- 💧 Water stations every 25m\n- 💧 Drink 250ml every 30 min\n- 🧂 Electrolytes recommended for {current_work_key}\n\n**Cooling:**\n- 🌳 Shaded rest areas with fans\n- ❄️ Cooling towels available\n- 🧊 Ice packs accessible\n\n**Productivity:** {current_pl:.1f}% loss expected")
        elif pet > 29:
            st.markdown(f"**Heat Risk:** ☀️ WARM (PET: {pet:.1f}°C)\n**Activity Level:** {current_work_key} ({met:.1f} met)\n\n**Work Schedule:**\n- ✓ Normal work with breaks\n- 🔄 60 min work / 10 min rest for {current_work_key}\n- 🌅 Normal working hours\n\n**Hydration:**\n- 💧 Water stations accessible\n- 💧 Regular hydration reminders\n\n**Cooling:**\n- 🌳 Shaded rest areas available\n- 🌬️ Natural ventilation sufficient\n\n**Productivity:** {current_pl:.1f}% loss expected")
        else:
            st.markdown(f"**Heat Risk:** ✅ COMFORTABLE (PET: {pet:.1f}°C)\n**Activity Level:** {current_work_key} ({met:.1f} met)\n\n**Work Schedule:**\n- ✓ Full work schedule\n- 🔄 Standard breaks\n\n**Hydration:**\n- 💧 Normal water access\n\n**Cooling:**\n- 🌳 Shade available if needed\n\n**Productivity:** No loss expected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="rec-card"><h3 style="color: #4ECDC4;">👷 For Workers</h3>', unsafe_allow_html=True)
        if pet > 41:
            st.markdown(f"**YOUR STATUS:** 🔥 VERY HOT\n**Your Work:** {current_work_key}\n\n**Protection:**\n- 👕 Light, loose clothing (current: {clo:.2f} clo)\n- 🧢 Wide-brim hat mandatory\n- 🧴 SPF 50+ sunscreen\n\n**Hydration:**\n- 💧 Drink 250ml every 15 min\n- 🚰 Use ORS in water\n- ❌ NO caffeine/alcohol\n\n**Work Behavior for {current_work_key}:**\n- 🚫 STOP if feeling unwell\n- 🌳 Stay in AC when not working\n- 🤝 Constant buddy check\n\n**WARNING SIGNS:**\n- 🚨 Headache, dizziness\n- 🚨 Nausea, confusion\n- 🚨 SEEK HELP IMMEDIATELY")
        elif pet > 35:
            st.markdown(f"**YOUR STATUS:** 🌡️ HOT\n**Your Work:** {current_work_key}\n\n**Protection:**\n- 👕 Breathable clothing (current: {clo:.2f} clo)\n- 🧢 Sun protection recommended\n- 🧴 SPF 30+ sunscreen\n\n**Hydration:**\n- 💧 Drink 250ml every 30 min\n- 🧂 Electrolytes recommended for {current_work_key}\n\n**Work Behavior for {current_work_key}:**\n- 🐢 Moderate pace\n- 🌳 45 min work, 15 min rest\n- 🤝 Buddy system\n\n**Watch For:**\n- ⚠️ Fatigue, cramps\n- ⚠️ Headache\n- ⚠️ Report symptoms")
        elif pet > 29:
            st.markdown(f"**YOUR STATUS:** ☀️ WARM\n**Your Work:** {current_work_key}\n\n**Protection:**\n- 👕 Standard work clothes\n\n**Hydration:**\n- 💧 Regular water breaks\n\n**Work Behavior for {current_work_key}:**\n- 🚶 Normal pace\n- 🌳 Take breaks in shade\n\n**Stay Aware:**\n- 👂 Listen to your body\n- 💬 Communicate discomfort")
        else:
            st.markdown(f"**YOUR STATUS:** ✅ COMFORTABLE\n**Your Work:** {current_work_key}\n\n**Protection:**\n- 👕 Normal work attire\n\n**Hydration:**\n- 💧 Drink when thirsty\n\n**Work Behavior:**\n- 🏃 Normal work\n- 🌳 Regular breaks")
        st.markdown('</div>', unsafe_allow_html=True)

# OPTIMIZATION 10: Model info as collapsed expander
with st.expander("ℹ️ More Information (click to expand)"):
    st.markdown(f"""
       - **PET thresholds:** >41°C Very Hot | 35-41°C Hot | 29-35°C Warm | 23-29°C Slightly Warm | <23°C Comfortable
    - **PMV thresholds:** ≥3.0 Extreme | 2.5-3.0 Very | 2.0-2.5 Moderate | 1.5-2.0 Discomfort | 1.0-1.5 Slight
    - **Current:** PET={pet:.1f}°C → {pet_class} | PMV={pmv:.2f} → {pmv_class}
    """)

st.markdown("---")

# Footer (minimized)
st.markdown("""
<div style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:2rem;">
  <div style="flex:0 0 250px;min-height:220px;text-align:left;background:rgba(255,255,255,0.03);padding:1rem 1.2rem;border-radius:16px;color:#D3D3D3;font-size:13px;font-weight:500;border:1px solid rgba(255,255,255,0.08);box-shadow:0 4px 12px rgba(0,0,0,0.1);">
      <b>Developed by:</b><br>
      <span style="font-size:.9rem;font-weight:500;color:#FFD700;">Md. Tushar Ali</span> (PhD Student)<br>
      Department of Civil & Environmental Engineering.<br>
      New Jersey Institute of Technology
  </div>
  <div style="flex:1;text-align:center;color:#B0E0E6;font-size:16px;font-style:italic;line-height:1.5;margin-top:.5rem;">
      This Web app has developed Based on Article of <b><i>Building and Environment</i></b>:<br>
      "Heat stress and thermal discomfort in construction: Implications for occupational safety and regulation"<br>
      <a href="https://www.sciencedirect.com/science/article/pii/S0360132326001769" target="_blank" style="color:#4ECDC4;font-weight:600;">View Article</a>
  </div>
</div>
<div style="text-align:center;margin-top:1rem;font-size:14px;color:rgba(255,255,255,0.7);">
    © 2026 Md. Tushar Ali. All rights reserved.
</div>
""", unsafe_allow_html=True)

# OPTIMIZATION 11: Periodic cleanup hint
st.markdown("""
<!-- Auto-refresh hint for deployment -->
<meta http-equiv="refresh" content="3600">
""", unsafe_allow_html=True)

