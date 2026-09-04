#Python liberaries used to develop the project
import streamlit as st
import tensorflow as tf
import numpy as np
import json, requests, os
import plotly.express as px
from PIL import Image

# Page Setup & Styling
st.set_page_config(page_title="Crop Disease Detection Model",page_icon="🌱",layout="wide")
st.markdown("""
<style>
    .metric-card {background-color: #f8f9fa; padding: 18px; border-radius: 10px; border: 1px solid #e9ecef; margin-bottom: 12px;}
    .high-risk {border-left: 5px solid #dc3545;}
    .med-risk {border-left: 5px solid #ffc107;}
    .low-risk {border-left: 5px solid #28a745;}
        div.stButton > button:first-child {background-color: #1e7e34 !important; color: white !important; font-weight: 600 !important; border-radius: 6px !important; height: 44px !important; width: 100% !important;}
</style>
""", unsafe_allow_html=True)

# MODEL 1st : CNN (ResNet50V2) for Disease Classifier 
MODEL_PATH = "crop_disease_resnet50v2.keras"
LABELS_PATH = "class_labels.json"
@st.cache_resource
def load_resnet_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None
@st.cache_data
def load_class_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return json.load(f)
    return {}
model = load_resnet_model()
class_labels = load_class_labels()
def get_disease_metadata(disease_name: str):
    clean_name = disease_name.replace("___", " - ").replace("_", " ")
    lower_name = disease_name.lower()
    is_healthy = "healthy" in lower_name
    
    if is_healthy:
        return {"name": clean_name,"type": "Healthy / No Pathogen","t_opt": (18, 30), "h_thresh": 95, "incubation": "N/A (Crop is healthy)", "actions": ["Maintain regular soil moisture and fertility balance.", "Continue standard weekly pest/pathogen monitoring."]}
    elif "bacterial" in lower_name:
        return {"name": clean_name, "type": "Bacterial Pathogen", "t_opt": (24, 32), "h_thresh": 80, "incubation": "3 to 7 days under wet, warm conditions" , "actions": ["Apply Copper Oxychloride or Streptocycline formulation.", "Avoid overhead sprinkler irrigation to reduce splash dispersal."]}
    elif "virus" in lower_name or "mosaic" in lower_name:
        return {"name": clean_name, "type": "Viral Infection (Vector-borne)", "t_opt": (22, 34), "h_thresh": 65, "incubation": "10 to 21 days post-insect feeding", "actions": ["Control insect vectors (whiteflies/aphids) using Imidacloprid or Neem oil.", "Rogue out and destroy severely stunted plants immediately."]}
    else:  
        return {"name": clean_name, "type": "Fungal Infection", "t_opt": (20, 28), "h_thresh": 75, "incubation": "7 to 12 days at optimal temperature", "actions": ["Apply broad-spectrum fungicide (Mancozeb, Chlorothalonil).", "Prune lower canopy foliage to improve intra-row ventilation."] }
#Resizes, scales, and passes the image through ResNet50V2.
def run_cnn_inference(image: Image.Image):
    if model is None or not class_labels:
        fallback = {"name": "Model Not Loaded", "type": "Error", "t_opt": (20, 30), "h_thresh": 75, "incubation": "N/A", "actions": ["Verify crop_disease_resnet50v2.keras is in this folder."]}
        return fallback, 0.0, False

    img_resized = image.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)
    
    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx] * 100.0)

    raw_name = class_labels.get(str(class_idx), "Unknown Class")
    disease_info = get_disease_metadata(raw_name)
    is_healthy = "healthy" in raw_name.lower()

    return disease_info, confidence, is_healthy

# MODEL 2: Microclimate Risk Engine
def calculate_microclimate_risk(temp, hum, rain, disease_info):
    t_min, t_max = disease_info["t_opt"]
    h_thresh = disease_info["h_thresh"]
    #Temperature Index 
    if t_min <= temp <= t_max:
        i_t = 1.0
    else:
        diff = min(abs(temp - t_min), abs(temp - t_max))
        i_t = max(0.2, 1.0 - (diff * 0.08))
    #Humidity Index
    if hum >= h_thresh:
        i_h = 1.0
    else:
        i_h = max(0.1, hum / h_thresh)
    #Rainfall Index 
    i_r = min(rain / 150.0, 1.0)
    # Weighted Microclimate Risk Score ( Humidity 40%, Temp 35%, Rain 25% )
    r_env = ((0.35 * i_t) + (0.40 * i_h) + (0.25 * i_r)) * 100.0
    return i_t, i_h, i_r, r_env

# MODEL 3: Decision Fusion Engine
def fuse_risk_index(confidence, r_env, is_healthy=False):
    if is_healthy:
        return 5.0, "Minimal"
    r_total = (0.50 * confidence) + (0.50 * r_env)

    if r_total >= 70:
        label = "High Alert"
    elif r_total >= 45:
        label = "Moderate Risk"
    else:
        label = "Low Spread Risk"
    return r_total, label

# Live Weather API using wheather API (https://www.weatherapi.com/)
WEATHERAPI_KEY ="3026e7729a3d42c685583451260309" 
def fetch_live_weather(city_name: str):
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={city_name}&days=7&aqi=no&alerts=no"
        res = requests.get(url, timeout=6)
        if res.status_code != 200:
            return None, f"Error: City not found or invalid API key (Status code {res.status_code})"
        data = res.json()
        location = data["location"]
        current = data["current"]
        current_data = {"resolved_name": f"{location['name']}, {location['country']}", "lat": float(location["lat"]), "lon": float(location["lon"]), "temp": float(current["temp_c"]), "humidity": float(current["humidity"]), "rain": float(current["precip_mm"])}
        return current_data, None

    except Exception as e:
        return None, str(e)

# Sidebar: User Controls & Inputs
st.sidebar.title("⚙️ Input leaf image and location")
uploaded_file = st.sidebar.file_uploader("Upload Photo of the Leaf want to Analyze", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.subheader("📍🗺️ Set Your Location for Analysis")
city_input = st.sidebar.text_input("Enter Your Location (City):", value="Bhopal")
weather_data, err = fetch_live_weather(city_input)
if weather_data:
    st.sidebar.success(f"Connected to the live weather API: {weather_data['resolved_name']}")
    temperature = weather_data["temp"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rain"]
    st.sidebar.info(f"🌡️ {temperature}°C | 💧 {humidity}% | 🌧️ {rainfall:.1f} mm/d")
#Location on Map
    map_data = [{"lat": weather_data["lat"], "lon": weather_data["lon"]}]
    st.sidebar.caption("🌍 Your Geographical Location📍")
    st.sidebar.map(map_data, zoom=9)
else:
    st.sidebar.error(f"⚠️ Please Enter a Valid City Name: {err or 'City not found'}")
    temperature = 0.0
    humidity = 0.0
    rainfall = 0.0

run_pipeline = st.sidebar.button("RUN DIAGNOSIS & RISK ANALYSIS")

# Main Execution & Visual Outputs
st.title("Crop Disease Detection & Risk Analytics with Prevention Recommendations")
if run_pipeline:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # 1. CNN Model Execution
        disease_info, confidence, is_healthy = run_cnn_inference(image)
        # 2. Microclimate Model Execution
        i_t, i_h, i_r, r_env = calculate_microclimate_risk(temperature, humidity, rainfall, disease_info)
        # 3. Decision Fusion Execution
        r_total, risk_label = fuse_risk_index(confidence, r_env, is_healthy)
#Fronted part of the backed results of the models
    # Diagnostic Summary interface
        st.subheader("📋 Diagnostic Summary")
        column1, column2 = st.columns([1, 1.8])
        with column1:
            st.image(image, use_container_width=True, caption="Leaf Sample Processed for Diagnosis")
        with column2:           
    #summary of model 1
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:#6c757d; font-size:13px; margin:0;">MODEL 1: CNN Model Prediction</p>
                <h3 style="color:#212529; margin:4px 0;">{disease_info['name']}</h3>
                <p style="margin:2px 0;"><b>Category of the Disease:</b> {disease_info['type']}</p>
                <p style="margin:2px 0;"><b>Detection Confidence:</b> {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    #summary of model 2
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:#6c757d; font-size:13px; margin:0;">MODEL 2: Microclimate Risk Engine</p>
                <p style="margin:4px 0 2px 0;"><b>Inputs:</b> {temperature}°C | {humidity}% RH | {rainfall:.1f} mm/d</p>
                <p style="margin:2px 0 0 0;"><b>Environment Risk :</b> {r_env:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    #summary of model 3
            card_class = "high-risk" if r_total >= 70 else ("med-risk" if r_total >= 45 else "low-risk")
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <p style="color:#6c757d; font-size:13px; margin:0;">MODEL 3: Fused Spread Risk of the detected disease</p>
                <h2 style="margin:4px 0; color:#111;">{r_total:.1f}%</h2>
                <p style="margin:0; font-weight:600;">Status: {risk_label}</p>
            </div>
            """, unsafe_allow_html=True)
    #Preventive measures and actions
        if rainfall > 2.0:
            spray_status = "⚠️ Delay Spray: Rainfall washout risk (>2 mm/d)."
            spray_badge_color = "#fde8e8; color: #991b1b"
        elif humidity > 90.0:
            spray_status = "⚠️ Restricted Window: High humidity prevents proper leaf drying."
            spray_badge_color = "#fef3c7; color: #92400e"
        else:
            spray_status = "✅ Favorable Spray Window: Good conditions for foliar application."
            spray_badge_color = "#def7ec; color: #03543f"

        action_bullets = "".join([f"<li style='margin-bottom: 4px;'>{act}</li>" for act in disease_info["actions"]])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top:0; color:#212529;">🛡️ Prescribed Steps to Manage the Disease</h3>
            <ul style="padding-left: 20px; margin-bottom: 12px;">
                {action_bullets}
            </ul>
            <!-- Spray Advisory Alert Pill -->
            <div style="background:{spray_badge_color}; padding:8px 12px; border-radius:6px; font-weight:600; font-size:13px; margin-bottom:12px;">
                {spray_status}
            </div>
            <p style="margin: 2px 0;"><b>Optimal Pathogen Temperature🌡️:</b> {disease_info['t_opt'][0]}–{disease_info['t_opt'][1]}°C</p>
            <p style="margin: 2px 0;"><b>Critical Humidity💧Cutoff:</b> &ge;{disease_info['h_thresh']}%</p>
            <p style="margin: 2px 0;"><b>Incubation Period⏳:</b> {disease_info['incubation']}</p>
        </div>
        """, unsafe_allow_html=True)

# Visual represenation through 2 graphs( Radar chart and horizontal Bar chart)
        st.markdown("---")
        st.subheader("📊 Environmental & Diagnostic Breakdown")
        graph_col1, graph_col2 = st.columns([1, 1])
        with graph_col1:
# GRAPH 1: Parameter Radar Chart
            st.subheader("1. Parameter Sensitivity Radar")
            categories = ['Temperature', 'Humidity', 'Rainfall', 'Vision Confidence']
            values = [i_t * 100, i_h * 100, i_r * 100, confidence]
            fig_radar = px.line_polar(r=values, theta=categories, line_close=True)
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)
        with graph_col2:
# GRAPH 2: Environmental Matrix 
            st.subheader("2. Risk Weight Breakdown vs. Total Risk")
            metrics_labels = ['Rainfall Effect', 'Temperature Effect', 'Humidity Effect', 'CNN Confidence', 'Final Risk Index']
            metrics_vals = [i_r * 25.0, i_t * 35.0, i_h * 40.0, confidence * 0.5, r_total]
            fig_bar = px.bar(x=metrics_vals, y=metrics_labels, orientation='h', labels={'x': 'Contribution Score (%)', 'y': 'Factor'}, color=metrics_labels)
            fig_bar.update_layout(showlegend=False, xaxis_range=[0, 100])
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("⚠️ Please upload a leaf image in the left panel to execute diagnosis.")
else:
    st.info("""👣 **STEPS TO START THE ANALYSIS:**  
                1. 🖼️ Upload a leaf image,  
                2. 📍 Select a city,  
                3. 🖱️ Click on **RUN DIAGNOSIS & RISK ANALYSIS** to start the analysis.  
                """, icon="ℹ️")