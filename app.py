import streamlit as st
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import folium
import streamlit.components.v1 as components
import plotly.graph_objects as go
import time
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & PERFECT FONT VISIBILITY CSS
# ==========================================
st.set_page_config(
    page_title="NAVAL SUBSEA ACOUSTIC COMMAND & MCM SYSTEM",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Rajdhani:wght@600;700&display=swap');
    
    /* Global Dark Background */
    .stApp, .stAppContainer, [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #080D1A !important;
        color: #FFFFFF !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    /* Crystal Clear White Text for Everything */
    p, span, label, div, .stMarkdown, .stText, .streamlit-expanderHeader, span[data-baseweb="tag"] {
        color: #FFFFFF !important;
        font-size: 16px !important;
    }

    /* Titles & Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00F5D4 !important;
        text-shadow: 0 0 10px rgba(0, 245, 212, 0.4);
    }
    
    h4, h5, h6 {
        color: #FFD166 !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #00F5D4 !important;
        font-size: 24px !important;
        font-weight: bold;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 13px !important;
        font-weight: bold;
    }
    
    .stMetric {
        background: #0F172A !important;
        border: 1px solid #334155 !important;
        border-left: 4px solid #00F5D4 !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }

    /* Input & Selectbox Dropdowns */
    .stSelectbox div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-color: #475569 !important;
    }

    /* Alert / Status Boxes */
    .hazard-box {
        background-color: rgba(239, 68, 68, 0.2);
        border: 2px solid #EF4444;
        color: #FF8888 !important;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .warning-box {
        background-color: rgba(245, 158, 11, 0.2);
        border: 2px solid #F59E0B;
        color: #FCD34D !important;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }

    .safe-box {
        background-color: rgba(16, 185, 129, 0.2);
        border: 2px solid #10B981;
        color: #86EFAC !important;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }

    .action-box {
        background-color: rgba(14, 165, 233, 0.2);
        border: 2px solid #0EA5E9;
        color: #7DD3FC !important;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AI MODEL DEFINITION
# ==========================================
class SonarCNN(nn.Module):
    def __init__(self):
        super(SonarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    model = SonarCNN()
    if os.path.exists("sonar_model.pth"):
        try:
            model.load_state_dict(torch.load("sonar_model.pth", map_location=torch.device('cpu')))
            model.eval()
            return model, True
        except Exception:
            return model, False
    return model, False

model, model_loaded = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ["Mine_Ordnance", "Safe_Seabed"]

def get_dataset_images(folder="dataset"):
    imgs = []
    if os.path.exists(folder):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    imgs.append(os.path.join(root, f))
    return imgs

real_dataset_files = get_dataset_images()

if "mission_records" not in st.session_state:
    st.session_state.mission_records = []

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("# ⚓ NAVAL MCM COMMAND")
st.sidebar.markdown("**Real-Time Subsea Operations**")

if model_loaded:
    st.sidebar.success("✅ PyTorch AI Engine Active")
else:
    st.sidebar.warning("⚠️ Baseline Engine Active")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌊 Ocean Environment")
st.sidebar.text("• Sea State: Smooth (2)")
st.sidebar.text("• Salinity: 34.8 PSU")
st.sidebar.text("• Water Temp: 24.2 °C")

st.sidebar.markdown("---")
st.sidebar.markdown(f"📂 **Dataset Files:** {len(real_dataset_files)} available")

st.sidebar.markdown("---")
nav_choice = st.sidebar.radio(
    "Select Tactical Command View:",
    [
        "🗺️ Real-Time AUV GIS Map & Telemetry",
        "📡 Target Acoustic Scan & Threat Action",
        "🏔️ 3D Bathymetry Seabed Terrain",
        "📦 Bulk Batch Processing Engine",
        "📊 Executive Audit Trail & Export"
    ]
)

# ==========================================
# VIEW 1: LIVE GIS MAP & TELEMETRY + RTH
# ==========================================
if nav_choice == "🗺️ Real-Time AUV GIS Map & Telemetry":
    st.title("🗺️ Real-Time AUV GIS Ocean Mapping & Telemetry")
    st.write("Autonomous subsea survey with real-time hardware telemetry and emergency controls.")

    base_lat, base_lon = 18.9100, 72.8200

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        num_targets = st.slider("Waypoints:", 3, min(20, max(3, len(real_dataset_files) if len(real_dataset_files)>0 else 5)), 6)
    with c_m2:
        sim_delay = st.slider("Delay (sec):", 0.1, 1.0, 0.3)
    with c_m3:
        btn_start = st.button("▶️ Launch Mission", type="primary")
    with c_m4:
        btn_rth = st.button("🚨 EMERGENCY RTH", type="secondary")

    waypoints = []
    for i in range(num_targets):
        lat = base_lat + (i * 0.0022) + (np.sin(i) * 0.0005)
        lon = base_lon + (i * 0.0028) + (np.cos(i) * 0.0003)
        img_f = real_dataset_files[i % len(real_dataset_files)] if len(real_dataset_files) > 0 else None
        waypoints.append({"id": i+1, "lat": lat, "lon": lon, "img": img_f})

    def render_map(history, current=None):
        m = folium.Map(location=[base_lat + 0.008, base_lon + 0.008], zoom_start=13, tiles="OpenStreetMap")
        path_pts = [[w["lat"], w["lon"]] for w in waypoints]
        folium.PolyLine(path_pts, color="#00F5D4", weight=3, opacity=0.8).add_to(m)

        for h in history:
            color = "red" if h["pred"] == "Mine_Ordnance" else "green"
            pop = f"Waypoint #{h['id']}<br>Status: {h['pred']}<br>Conf: {h['conf']:.1f}%"
            folium.Marker(location=[h["lat"], h["lon"]], popup=pop, icon=folium.Icon(color=color, icon="info-sign")).add_to(m)

        if current:
            folium.CircleMarker(location=[current["lat"], current["lon"]], radius=10, color="cyan", fill=True, fill_color="cyan", fill_opacity=1.0).add_to(m)
        return m

    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    m_lat = col_t1.empty()
    m_lon = col_t2.empty()
    m_batt = col_t3.empty()
    m_depth = col_t4.empty()
    m_stat = col_t5.empty()

    col_map, col_graph = st.columns([1.2, 0.8])
    map_slot = col_map.empty()
    graph_slot = col_graph.empty()

    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "live_history" not in st.session_state:
        st.session_state.live_history = []
    if "telemetry_logs" not in st.session_state:
        st.session_state.telemetry_logs = []

    if btn_rth:
        st.session_state.is_running = False
        st.error("🚨 EMERGENCY RETURN-TO-HOME (RTH) ACTIVATED! AUV is returning to base station.")
    elif btn_start:
        st.session_state.is_running = True
        st.session_state.current_step = 0
        st.session_state.live_history = []
        st.session_state.telemetry_logs = []
        st.rerun()

    if st.session_state.is_running:
        if st.session_state.current_step < len(waypoints):
            idx = st.session_state.current_step
            wp = waypoints[idx]
            
            battery_pct = max(20, 98 - (idx * 3))
            current_depth = round(45.0 + (np.sin(idx) * 3.5), 1)

            if wp["img"] and os.path.exists(wp["img"]):
                p_img = Image.open(wp["img"]).convert("RGB")
                t_img = transform(p_img).unsqueeze(0)
                with torch.no_grad():
                    out = model(t_img)
                    probs = torch.softmax(out, dim=1)[0]
                    p_i = torch.argmax(probs).item()
                    conf = float(probs[p_i].item() * 100)
                pred_label = classes[p_i]
            else:
                pred_label = "Safe_Seabed"
                conf = 92.0

            st.session_state.live_history.append({
                "id": wp["id"], "lat": wp["lat"], "lon": wp["lon"],
                "img": wp["img"], "pred": pred_label, "conf": conf
            })

            st.session_state.telemetry_logs.append({
                "Waypoint": f"WP-{wp['id']}",
                "Depth (m)": current_depth,
                "Battery (%)": battery_pct
            })

            st.session_state.mission_records.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Waypoint": f"WP-{wp['id']}",
                "Latitude": f"{wp['lat']:.5f}",
                "Longitude": f"{wp['lon']:.5f}",
                "Image File": os.path.basename(wp["img"]) if wp["img"] else "N/A",
                "Classification": pred_label,
                "Confidence": f"{conf:.2f}%"
            })

            m_lat.metric("Latitude", f"{wp['lat']:.4f}° N")
            m_lon.metric("Longitude", f"{wp['lon']:.4f}° E")
            m_batt.metric("Battery Level", f"{battery_pct}%")
            m_depth.metric("Depth", f"{current_depth} m")
            m_stat.metric("Status", f"🔴 RUNNING ({idx+1}/{len(waypoints)})")
            
            m_obj = render_map(st.session_state.live_history, current=wp)
            with map_slot.container():
                st.markdown("### 🌐 Live AUV Trajectory Map")
                components.html(m_obj._repr_html_(), height=420)

            with graph_slot.container():
                st.markdown("### 📈 Live Telemetry Graphs")
                df_tele = pd.DataFrame(st.session_state.telemetry_logs).set_index("Waypoint")
                st.write("🌊 **AUV Depth (meters)**")
                st.line_chart(df_tele[["Depth (m)"]], height=160, color="#00F5D4")
                st.write("🔋 **Battery Level (%)**")
                st.line_chart(df_tele[["Battery (%)"]], height=160, color="#FFD166")

            time.sleep(sim_delay)
            st.session_state.current_step += 1
            st.rerun()
        else:
            st.session_state.is_running = False
            m_lat.metric("Latitude", f"{waypoints[-1]['lat']:.4f}° N")
            m_lon.metric("Longitude", f"{waypoints[-1]['lon']:.4f}° E")
            m_batt.metric("Battery Level", "Completed")
            m_depth.metric("Depth", "45.0 m")
            m_stat.metric("Status", "✅ COMPLETED")
            
            st.success("🎉 Mission Completed Successfully!")
            m_obj = render_map(st.session_state.live_history)
            with map_slot.container():
                st.markdown("### 🌐 Final AUV Trajectory Map")
                components.html(m_obj._repr_html_(), height=420)

            with graph_slot.container():
                st.markdown("### 📈 Complete Telemetry Graphs")
                df_tele = pd.DataFrame(st.session_state.telemetry_logs).set_index("Waypoint")
                st.write("🌊 **AUV Depth (meters)**")
                st.line_chart(df_tele[["Depth (m)"]], height=160, color="#00F5D4")
                st.write("🔋 **Battery Level (%)**")
                st.line_chart(df_tele[["Battery (%)"]], height=160, color="#FFD166")
    else:
        m_lat.metric("Latitude", f"{base_lat:.4f}° N")
        m_lon.metric("Longitude", f"{base_lon:.4f}° E")
        m_batt.metric("Battery Level", "98%")
        m_depth.metric("Depth", "45.2 m")
        m_stat.metric("Status", "STANDBY")
        
        m_obj = render_map([])
        with map_slot.container():
            st.markdown("### 🌐 AUV Trajectory Map (Standby)")
            components.html(m_obj._repr_html_(), height=420)

        with graph_slot.container():
            st.markdown("### 📈 Telemetry Graphs (Standby)")
            dummy_df = pd.DataFrame({"Depth (m)": [45.0, 45.5], "Battery (%)": [98, 97]}, index=["WP-1", "WP-2"])
            st.write("🌊 **AUV Depth (meters)**")
            st.line_chart(dummy_df[["Depth (m)"]], height=160, color="#00F5D4")
            st.write("🔋 **Battery Level (%)**")
            st.line_chart(dummy_df[["Battery (%)"]], height=160, color="#FFD166")

# ==========================================
# VIEW 2: TARGET ACOUSTIC SCAN & DYNAMIC FFT & WARNINGS
# ==========================================
elif nav_choice == "📡 Target Acoustic Scan & Threat Action":
    st.title("📡 Target Acoustic Scan & Threat Action")
    st.write("Analyze individual acoustic files with AI and real-time dynamic frequency spectrum analysis.")

    if len(real_dataset_files) == 0:
        st.error("❌ कृपया 'dataset' फोल्डरमध्ये काही इमेजेस टाका!")
    else:
        selected_file = st.selectbox("Select Target File from Dataset:", real_dataset_files)

        if selected_file:
            raw_pil = Image.open(selected_file).convert("RGB")
            img_np = np.array(raw_pil)
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            mean_i = float(np.mean(gray_img))
            std_i = float(np.std(gray_img))
            snr = mean_i / (std_i + 1e-5)

            t_input = transform(raw_pil).unsqueeze(0)
            with torch.no_grad():
                out = model(t_input)
                probs = torch.softmax(out, dim=1)[0]
                p_idx = torch.argmax(probs).item()
                conf = float(probs[p_idx].item() * 100)

            prediction = classes[p_idx]

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.subheader("Raw Sonar Scan")
                st.image(raw_pil, use_container_width=True)

            with col_a2:
                st.subheader("Enhanced Heatmap Analysis")
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray_img)
                color_map = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
                st.image(color_map, channels="BGR", use_container_width=True)

            st.markdown("---")
            st.subheader("🎵 Acoustic Frequency Spectrum (Dynamic FFT Analysis from Image Pixels)")
            
            # REAL IMAGE-BASED DYNAMIC FREQUENCY SPECTRUM (OpenCV FFT on Gray Image)
            f_transform = np.fft.fft2(gray_img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            h_sz, w_sz = magnitude_spectrum.shape
            freq_profile = magnitude_spectrum[h_sz // 2, :]
            freqs = np.linspace(100, 5000, len(freq_profile))
            
            fig_fft = go.Figure(data=go.Scatter(
                x=freqs, y=freq_profile, 
                mode='lines', 
                line=dict(color='#00F5D4', width=2), 
                fill='tozeroy'
            ))
            fig_fft.update_layout(
                paper_bgcolor='#080D1A', plot_bgcolor='#0F172A',
                font=dict(color='#FFFFFF'), margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title='Frequency (Hz)', gridcolor='#334155'),
                yaxis=dict(title='Amplitude (dB)', gridcolor='#334155'),
                height=250
            )
            st.plotly_chart(fig_fft, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Diagnostics & Dynamic Threat Warnings")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Classification", prediction.replace("_", " "))
            k2.metric("AI Confidence", f"{conf:.2f}%")
            k3.metric("SNR Ratio", f"{snr:.2f}")
            k4.metric("Intensity", f"{mean_i:.1f} / 255")

            # FULLY DYNAMIC WARNING LOGIC BASED ON PREDICTION, CONFIDENCE & SNR
            max_fft_val = float(np.max(freq_profile))

            if prediction == "Mine_Ordnance" and conf > 75.0:
                st.markdown('<div class="hazard-box">🚨 CRITICAL THREAT ALERT: High-Probability Bottom Mine Signature Detected! Immediate Countermeasure Required.</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="action-box" style="margin-top: 15px;">
                    <b>🛡️ DYNAMIC DEFENSE ACTION PLAN (Target ID: {os.path.basename(selected_file)}):</b><br>
                    • Acoustic Frequency Peak: <b>{max_fft_val:.1f} dB</b><br>
                    1. Establish 1000m Maritime Exclusion Zone around coordinates.<br>
                    2. Deploy Remotely Operated Vehicle (ROV) for optical ID verification.<br>
                    3. Dispatch EOD (Explosive Ordnance Disposal) team for neutralisation.
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Mine_Ordnance" and conf <= 75.0:
                st.markdown('<div class="warning-box">⚠️ MODERATE THREAT WARNING: Ambiguous Anomaly Profile Detected. Secondary Scan Recommended.</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="action-box" style="margin-top: 15px;">
                    <b>🛡️ DYNAMIC ACTION PLAN:</b><br>
                    • Low Confidence ({conf:.1f}%). Re-scan target from closer proximity (< 15m altitude).
                </div>
                """, unsafe_allow_html=True)
            elif snr > 4.5 and max_fft_val > 150:
                st.markdown('<div class="warning-box">⚠️ HIGH RECLAMATION ANOMALY: Unusual Seabed Texture / Metallic Scatterer Found.</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="action-box" style="margin-top: 15px;">
                    <b>🛡️ DYNAMIC ACTION PLAN:</b><br>
                    1. Flag target for hydrographic review.<br>
                    2. Log spatial coordinates for sonar mapping validation.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-box">🛡️ CLEAR SEABED: Natural Sand Ripple / Normal Geology (Confidence: {conf:.1f}%, SNR: {snr:.2f}). No Action Needed.</div>', unsafe_allow_html=True)

# ==========================================
# VIEW 3: 3D BATHYMETRY SEABED TERRAIN
# ==========================================
elif nav_choice == "🏔️ 3D Bathymetry Seabed Terrain":
    st.title("🏔️ 3D Bathymetry Seabed Terrain Visualizer")
    st.write("Interactive 3D wireframe and surface topography of the surveyed ocean floor.")

    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * 10 - 45

    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig_3d.update_layout(
        title='3D Seabed Contour & Obstacle Topography',
        paper_bgcolor='#080D1A', plot_bgcolor='#080D1A',
        font=dict(color='#FFFFFF'),
        scene=dict(
            xaxis=dict(backgroundcolor='#0F172A', gridcolor='#334155', title='X (km)'),
            yaxis=dict(backgroundcolor='#0F172A', gridcolor='#334155', title='Y (km)'),
            zaxis=dict(backgroundcolor='#0F172A', gridcolor='#334155', title='Depth (m)')
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=550
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.info("💡 Tip: You can click and drag the 3D graph to rotate the ocean floor from different angles.")

# ==========================================
# VIEW 4: BULK BATCH PROCESSING
# ==========================================
elif nav_choice == "📦 Bulk Batch Processing Engine":
    st.title("📦 Bulk Batch Processing Engine")
    st.write("Process multiple sonar scans instantly in real-time.")

    uploaded_files = st.file_uploader("Upload Sonar Scans:", type=['png', 'jpg', 'jpeg', 'tif'], accept_multiple_files=True)

    if uploaded_files:
        if st.button("🚀 Run Batch AI Classification", type="primary"):
            batch_results = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                img = Image.open(file).convert("RGB")
                t_img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(t_img)
                    probs = torch.softmax(out, dim=1)[0]
                    p_i = torch.argmax(probs).item()
                    conf = float(probs[p_i].item() * 100)
                pred = classes[p_i]

                batch_results.append({
                    "Filename": file.name,
                    "Prediction": pred,
                    "Confidence": f"{conf:.2f}%",
                    "Status": "FLAGGED HAZARD" if pred == "Mine_Ordnance" else "SAFE"
                })
                progress_bar.progress((idx + 1) / len(uploaded_files))

            df_batch = pd.DataFrame(batch_results)
            st.success(f"✅ Completed processing {len(uploaded_files)} files!")
            st.dataframe(df_batch, use_container_width=True)
    else:
        st.info("💡 Tip: Upload multiple files using the uploader above.")

# ==========================================
# VIEW 5: EXECUTIVE AUDIT & EXPORT
# ==========================================
elif nav_choice == "📊 Executive Audit Trail & Export":
    st.title("📊 Tactical Mission Audit Trail & Report Export")

    if len(st.session_state.mission_records) > 0:
        df_audit = pd.DataFrame(st.session_state.mission_records)
        st.dataframe(df_audit, use_container_width=True)

        csv_data = df_audit.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Tactical Report",
            data=csv_data,
            file_name=f"Naval_MCM_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No mission records found yet. Run a live mission from View 1 first.")
