import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

# Load dataset & models
@st.cache_data
def load_data():
    df = pd.read_csv('India_Soil_S2_Synthetic_Linked_1200.csv')
    models = {
        'N': joblib.load('models/xgb_optimized_N_kg_ha.joblib'),
        'P': joblib.load('models/xgb_optimized_P_kg_ha.joblib'),
        'K': joblib.load('models/xgb_optimized_K_kg_ha.joblib')
    }
    return df, models

df, models = load_data()

st.title("🌱 **Soil NPK Predictor**")
st.markdown("**Sentinel-2 + XGBoost | R²: N=0.38, P=0.24, K=0.42**")

# Input coordinates
col1, col2 = st.columns(2)
lat = col1.number_input("📍 Latitude", 8.0, 35.0, 19.9993)
lon = col2.number_input("📍 Longitude", 68.0, 97.0, 73.7900)

if st.button("🚀 **PREDICT NPK**", type="primary"):
    # Find nearest location
    coords = df[['latitude', 'longitude']].values
    target = np.array([[lat, lon]])
    distances = haversine_distances(target, coords) * 6371
    closest_idx = np.argmin(distances)
    closest_row = df.iloc[closest_idx]
    
    # Prepare features (same as training)
    s2_bands = closest_row[['S2_B2', 'S2_B3', 'S2_B4', 'S2_B8', 'S2_B11', 'S2_B12']].values
    s2_indices = closest_row[['S2_NDVI', 'S2_SAVI', 'S2_BSI']].values
    b11_b12 = s2_bands[4] / (s2_bands[5] + 1e-6)
    b8_b4 = s2_bands[3] / (s2_bands[2] + 1e-6)
    vis_sum = s2_bands[0] + s2_bands[1] + s2_bands[2]
    
    features = np.array([
        s2_bands[0], s2_bands[1], s2_bands[2], s2_bands[3], s2_bands[4], s2_bands[5],
        s2_indices[0], s2_indices[1], s2_indices[2], b11_b12, b8_b4, vis_sum
    ]).reshape(1, -1)
    
    # Predict
    predictions = {
        'N_kg_ha': float(models['N'].predict(features)[0]),
        'P_kg_ha': float(models['P'].predict(features)[0]),
        'K_kg_ha': float(models['K'].predict(features)[0])
    }
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌾 Nitrogen", f"{predictions['N_kg_ha']:.1f}", f"{closest_row['N_kg_ha']:.1f}")
    with col2:
        st.metric("🔬 Phosphorus", f"{predictions['P_kg_ha']:.1f}", f"{closest_row['P_kg_ha']:.1f}")
    with col3:
        st.metric("🍌 Potassium", f"{predictions['K_kg_ha']:.1f}", f"{closest_row['K_kg_ha']:.1f}")
    
    st.success(f"✅ **Match found**: {closest_row['state']}, {closest_row['district']} ({closest_row['soil_type']} soil)")

