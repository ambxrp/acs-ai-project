import os
import json
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
from src.pipeline import AnimalManagementPipeline

# District mappings for the dashboard
DISTRICT_NAMES = {
    1: "District 1 (Downtown / Central)",
    2: "District 2 (Eastside)",
    3: "District 3 (Southside / Brooks)",
    4: "District 4 (Southwest)",
    5: "District 5 (Westside)",
    6: "District 6 (Far Westside)",
    7: "District 7 (Northwest)",
    8: "District 8 (Far Northwest / Medical Center)",
    9: "District 9 (North Central)",
    10: "District 10 (Northeast)"
}

# Set layout parameters
st.set_page_config(
    page_title="ACS Early Warning System", 
    page_icon="🐾", 
    layout="wide"
)

# Initialize backend pipeline from environment
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

# Configure sidebar scenario options
st.sidebar.title("⚙️ Scenario Controls")
st.sidebar.markdown("Configure the timeframe and environmental variables to simulate a predictive scenario.")

district = st.sidebar.selectbox(
    "Target City Council District", 
    options=list(DISTRICT_NAMES.keys()), 
    format_func=lambda x: DISTRICT_NAMES[x],
    index=2
)

month = st.sidebar.slider("Target Month of Year", min_value=1, max_value=12, value=7)
week = st.sidebar.slider("Target Week of Year", min_value=1, max_value=53, value=28)

base_temp, base_precip = pipeline.data_processor.get_weather_baseline(month)

st.sidebar.markdown("### Weather Forecast")
temp = st.sidebar.slider("Temperature Forecast (°F)", min_value=40.0, max_value=110.0, value=float(base_temp), step=1.0)
precip = st.sidebar.slider("Precipitation Forecast (in)", min_value=0.0, max_value=8.0, value=float(base_precip), step=0.1)

generate_btn = st.sidebar.button("Generate Memo", type="primary", use_container_width=True)

# Advanced model training controls
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
                pipeline.load_system()
                st.success("✅ Models retrained and saved successfully!")
                st.cache_resource.clear() 
            except Exception as e:
                st.error(f"Training failed: {e}")

# Render main panel headers
st.title("🐾 ACS Capacity & Stray Early Warning System")
st.markdown("Proactively manage capacity strain, field deployments, and animal intake risks across San Antonio.")

# Initialize session state for scenario outputs
if 'scenario_results' not in st.session_state:
    st.session_state.scenario_results = None

# Store execution outputs when generated
if generate_btn:
    with st.spinner("Evaluating historical datasets and generating operational memo..."):
        try:
            st.session_state.scenario_results = pipeline.run_scenario(
                district_id=district,
                month=month,
                week_of_year=week,
                temp_override=temp,
                precip_override=precip
            )
        except Exception as e:
            st.error(f"An error occurred while running the prediction pipeline: {e}")
            st.session_state.scenario_results = None

# Render scenario outputs if available
if st.session_state.scenario_results is not None:
    results = st.session_state.scenario_results
    predictions = results['predictions']
    insights = results['insights']
    memo = results['operational_memo']

    # Display weekly metrics
    st.subheader("📊 Projected Weekly Case Volumes", divider="gray")
    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Total Cases", predictions['predicted_total_count'])
    col2.metric("Projected Stray Reports", predictions['predicted_stray_count'])
    col3.metric("Projected Aggressive Incidents", predictions['predicted_aggressive_count'])

    # Display capacity warning cards
    st.subheader("⚠️ Capacity Warning Level", divider="gray")
    strain_score = predictions['predicted_capacity_strain_score']
    
    if strain_score < 40:
        st.success(f"🟢 **Low Strain (Score: {strain_score}/100)** - Expected volume is well within normal operational thresholds.")
    elif strain_score <= 75:
        st.warning(f"🟠 **Medium Strain (Score: {strain_score}/100)** - Elevated volume expected. Preemptively coordinate mobile clinics and foster networks.")
    else:
        st.error(f"🔴 **High Strain (Score: {strain_score}/100)** - CRITICAL: High risk of exceeding shelter capacity. Immediate action required.")

    # Render mapping components and historical stats
    st.subheader(f"📍 {DISTRICT_NAMES[district]} Context", divider="gray")
    
    map_col, text_col = st.columns([1.5, 1])
    
    with map_col:
        try:
            import math
            
            with open("data/Council_Districts.geojson", "r") as f:
                geojson_data = json.load(f)

            # Convert coordinate points to standard latitude longitude format
            def convert_coords(coords):
                if isinstance(coords[0], (int, float)):
                    x, y = coords[0], coords[1]
                    lon = (x / 20037508.34) * 180
                    lat = (math.atan(math.exp((y / 20037508.34) * math.pi)) * 2 - math.pi / 2) * 180 / math.pi
                    return [lon, lat]
                return [convert_coords(c) for c in coords]

            sample = geojson_data['features'][0]['geometry']['coordinates']
            while isinstance(sample[0], list):
                sample = sample[0]
                
            if sample[0] < -180 or sample[0] > 180:
                for feature in geojson_data['features']:
                    feature['geometry']['coordinates'] = convert_coords(feature['geometry']['coordinates'])

            # Find matching district element
            target_feature = None
            for feature in geojson_data['features']:
                prop_dist = feature['properties'].get('District')
                if prop_dist is not None and int(float(prop_dist)) == district:
                    target_feature = feature
                    break
            
            # Draw folium map layers
            if target_feature:
                single_feature_geojson = {
                    "type": "FeatureCollection",
                    "features": [target_feature]
                }
                district_bounds = folium.GeoJson(single_feature_geojson).get_bounds()
                
                m = folium.Map()
                m.fit_bounds(district_bounds)
            else:
                m = folium.Map(location=[29.4241, -98.4936], zoom_start=10)

            # Apply style rules to target shape
            def style_fn(feature):
                feature_dist = int(float(feature['properties'].get('District', 0)))
                is_target = feature_dist == district
                
                return {
                    'fillColor': '#ff4b4b' if is_target else '#3186cc',
                    'color': 'black',
                    'weight': 3 if is_target else 1,
                    'fillOpacity': 0.6 if is_target else 0.1,
                }

            folium.GeoJson(
                geojson_data,
                style_function=style_fn,
                tooltip=folium.GeoJsonTooltip(fields=['District'], aliases=['District: '])
            ).add_to(m)

            st_folium(m, height=350, use_container_width=True, returned_objects=[])
            
        except FileNotFoundError:
            st.warning("⚠️ Map data not found. Please ensure 'Council_Districts.geojson' is located in your 'data/' folder.")

    with text_col:
        st.caption(f"Based on {insights['total_historical_calls']:,} historical service records for this district.")
        st.markdown("**Top Historical Call Types**")
        for ctype in insights['top_call_types']:
            st.markdown(f"- {ctype}")
        
        st.markdown("")
        st.markdown("**Common Incident Hotspots (Streets)**")
        for loc in insights['top_locations']:
            st.markdown(f"- {loc}")

    # Render memo field values
    st.subheader("📋 Operational Command & Early Warning Memo", divider="gray")
    st.markdown(memo)
    
    with st.expander("Show Raw Markdown for Copying"):
        st.code(memo, language='markdown')

else:
    st.info("👈 Adjust the scenario parameters in the sidebar and click **'Generate Memo'** to execute the pipeline.")