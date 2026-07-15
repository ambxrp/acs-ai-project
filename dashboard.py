import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.pipeline import AnimalManagementPipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="ACS Early Warning System", 
    page_icon="🐾", 
    layout="wide"
)

# --- Environment & Backend Initialization ---
@st.cache_resource
def load_pipeline():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    pipeline = AnimalManagementPipeline(
        data_path="data/acs_data.csv",
        model_dir="models",
        api_key=api_key
    )
    pipeline.load_system() 
    return pipeline

pipeline = load_pipeline()

if pipeline is None:
    st.error("❌ API Key Error: GEMINI_API_KEY could not be retrieved. Please check your .env file.")
    st.stop()

# --- 1. Sidebar: Scenario Controls ---
st.sidebar.title("⚙️ Scenario Controls")
st.sidebar.markdown("Configure the timeframe and environmental variables to simulate a predictive scenario.")

district = st.sidebar.selectbox("Target City Council District", options=list(range(1, 11)), index=2)

# Using the two separate sliders as requested
month = st.sidebar.slider("Target Month of Year", min_value=1, max_value=12, value=7)
week = st.sidebar.slider("Target Week of Year", min_value=1, max_value=53, value=28)

base_temp, base_precip = pipeline.data_processor.get_weather_baseline(month)

st.sidebar.markdown("### Weather Forecast")
temp = st.sidebar.slider("Temperature Forecast (°F)", min_value=40.0, max_value=110.0, value=float(base_temp), step=1.0)
precip = st.sidebar.slider("Precipitation Forecast (in)", min_value=0.0, max_value=8.0, value=float(base_precip), step=0.1)

generate_btn = st.sidebar.button("Generate Memo", type="primary", use_container_width=True)

# --- NEW: Sidebar: Advanced Model Management ---
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Advanced: Model Management"):
    st.markdown("Use your backend's `train_system` functionality to retrain the Random Forest models with new parameters.")
    aug_factor = st.number_input("Data Augmentation Factor", min_value=1, max_value=20, value=5)
    n_est = st.number_input("RF Estimators (Trees)", min_value=10, max_value=500, value=100, step=10)
    
    if st.button("Retrain Models", use_container_width=True):
        with st.spinner("Aggregating data and retraining backend models..."):
            try:
                pipeline.train_system(
                    augment=True, 
                    augmentation_factor=aug_factor, 
                    n_estimators=n_est
                )
                pipeline.load_system() # Reload the newly trained models into memory
                st.success("✅ Models retrained and saved successfully!")
                # Clear the cache so the app uses the fresh models
                st.cache_resource.clear() 
            except Exception as e:
                st.error(f"Training failed: {e}")

# --- Main Dashboard Panel ---
st.title("🐾 ACS Capacity & Stray Early Warning System")
st.markdown("Proactively manage capacity strain, field deployments, and animal intake risks across San Antonio.")

if generate_btn:
    with st.spinner("Evaluating historical datasets and generating operational memo..."):
        try:
            results = pipeline.run_scenario(
                district_id=district,
                month=month,
                week_of_year=week,
                temp_override=temp,
                precip_override=precip
            )
            
            predictions = results['predictions']
            insights = results['insights']
            memo = results['operational_memo']

            st.subheader("📊 Projected Weekly Case Volumes", divider="gray")
            col1, col2, col3 = st.columns(3)
            col1.metric("Projected Total Cases", predictions['predicted_total_count'])
            col2.metric("Projected Stray Reports", predictions['predicted_stray_count'])
            col3.metric("Projected Aggressive Incidents", predictions['predicted_aggressive_count'])

            st.subheader("⚠️ Capacity Warning Level", divider="gray")
            strain_score = predictions['predicted_capacity_strain_score']
            
            if strain_score < 40:
                st.success(f"🟢 **Low Strain (Score: {strain_score}/100)** - Expected volume is well within normal operational thresholds.")
            elif strain_score <= 75:
                st.warning(f"🟠 **Medium Strain (Score: {strain_score}/100)** - Elevated volume expected. Preemptively coordinate mobile clinics and foster networks.")
            else:
                st.error(f"🔴 **High Strain (Score: {strain_score}/100)** - CRITICAL: High risk of exceeding shelter capacity. Immediate action required.")

            st.subheader(f"📍 District {district} Historical Context", divider="gray")
            st.caption(f"Based on {insights['total_historical_calls']:,} historical service records for this district.")
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.markdown("**Top Historical Call Types**")
                for ctype in insights['top_call_types']:
                    st.markdown(f"- {ctype}")
                    
            with h_col2:
                st.markdown("**Common Incident Hotspots (Streets)**")
                for loc in insights['top_locations']:
                    st.markdown(f"- {loc}")

            st.subheader("📋 Operational Command & Early Warning Memo", divider="gray")
            st.markdown(memo)
            
            with st.expander("Show Raw Markdown for Copying"):
                st.code(memo, language='markdown')

        except Exception as e:
            st.error(f"An error occurred while running the prediction pipeline: {e}")
else:
    st.info("👈 Adjust the scenario parameters in the sidebar and click **'Generate Memo'** to execute the pipeline.")