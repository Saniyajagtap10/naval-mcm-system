import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
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
    
    .stApp, .stAppContainer, [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #080D1A !important;
        color: #FFFFFF !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    p, span, label, div, .stMarkdown, .stText, .streamlit-expanderHeader, span[data-baseweb="tag"] {
        color: #FFFFFF !important;
        font-size: 16px !important;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00F5D4 !important;
        text-shadow: 0 0 10px rgba(0, 245, 212, 0.4);
    }
    
    h4, h5, h6 {
        color: #FFD166 !important;
        font-family: 'Orbitron', sans-serif !important;
    }

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

    .stSelectbox div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-color: #475569 !important;
    }

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
# 2. AI MODEL & PROCEDURAL GENERATOR
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

# Generate real-time synthetic sonar image dynamically
def generate_realtime_sonar_scan(seed_val):
    np.random.seed(seed_val)
    base = np.random.normal(120, 30, (224, 224)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (15, 15), 0)
    
    for i in range(0, 224, 20):
        cv2.line(base, (0, i), (224, i + np.random.randint(-10, 10)), (80, 80, 80), 1)

    is_threat = (seed_val % 3 == 0) or (seed_val % 5 == 0)
    if is_threat:
        center_x = np.random.randint(60, 164)
        center_y = np.random.randint(60, 164)
        cv2.circle(base, (center_x, center_y), np.random.randint(12, 22), (255, 255, 255), -1)
        cv2.circle(base, (center_x + 5, center_y + 5), np.random.randint(4, 8), (20, 20, 20), -1)
        pred = "Mine_Ordnance"
        conf = float(np.random.uniform(76.5, 98.9))
    else:
        pred = "Safe_Seabed"
        conf = float(np.random.uniform(82.0, 99.4))

    img_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb), pred, conf

if "mission_records" not in st.session_state:
    st.session_state.mission_records = []

# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
st.sidebar.markdown("# ⚓ NAVAL MCM COMMAND")
st.sidebar.markdown("**Real-Time Multi-AUV Operations**")
st.sidebar.success("✅ Real-Time Procedural Engine Active")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Live Environmental Controls")
turbidity = st.sidebar.slider("Water Turbidity (NTU):", 0.5, 15.0, 2.3, 0.1)
current_speed = st.sidebar.slider("Current Velocity (Knots):", 0.0, 5.0, 1.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.text(f"• Sea State: {int(current_speed*1.5)} (Moderate)")
st.sidebar.text("• Salinity: 34.8 PSU")
st.sidebar.text(f"• Water Temp: {24.2 - (turbidity*0.05):.1f} °C")

st.sidebar.markdown("---")
nav_choice = st.sidebar.radio(
    "Select Tactical Command View:",
    [
        "🗺️ Multi-AUV Swarm GIS Map & Telemetry",
        "📡 Target Acoustic Scan & Threat Action",
        "🏔️ 3D Bathymetry Seabed Terrain",
        "📊 Executive Audit Trail & Report"
    ]
)

base_lat, base_lon = 18.9100, 72.8200

# ==========================================
# VIEW 1: MULTI-AUV SWARM GIS MAP & TELEMETRY
# ==========================================
if nav_choice == "🗺️ Multi-AUV Swarm GIS Map & Telemetry":
    st.title("🗺️ Multi-AUV Swarm GIS Ocean Mapping & Telemetry")
    st.write("Live autonomous multi-agent subsea survey (Alpha, Bravo, Charlie) with real-time changing classifications.")

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        num_targets = st.slider("Waypoints per AUV:", 3, 10, 5)
    with c_m2:
        sim_delay = st.slider("Delay (sec):", 0.1, 1.0, 0.3)
    with c_m3:
        btn_start = st.button("▶️ Launch Swarm Mission", type="primary")
    with c_m4:
        btn_rth = st.button("🚨 EMERGENCY RTH", type="secondary")

    auv_routes = {
        "AUV-Alpha": [{"id": i+1, "lat": base_lat + (i * 0.002), "lon": base_lon + (i * 0.0025), "seed": i + 10} for i in range(num_targets)],
        "AUV-Bravo": [{"id": i+1, "lat": base_lat + 0.003 + (i * 0.0018), "lon": base_lon - 0.002 + (i * 0.0022), "seed": i + 50} for i in range(num_targets)],
        "AUV-Charlie": [{"id": i+1, "lat": base_lat - 0.003 + (i * 0.0022), "lon": base_lon + 0.003 + (i * 0.0015), "seed": i + 90} for i in range(num_targets)]
    }

    def render_swarm_map(history_dict, current_dict=None):
        m = folium.Map(location=[base_lat, base_lon + 0.002], zoom_start=13, tiles="OpenStreetMap")
        colors = {"AUV-Alpha": "blue", "AUV-Bravo": "purple", "AUV-Charlie": "orange"}
        
        for auv_name, pts in auv_routes.items():
            path_pts = [[w["lat"], w["lon"]] for w in pts]
            folium.PolyLine(path_pts, color=colors[auv_name], weight=2.5, opacity=0.7, tooltip=auv_name).add_to(m)

        for auv_name, hist_list in history_dict.items():
            for h in hist_list:
                col = "red" if h["pred"] == "Mine_Ordnance" else "green"
                pop = f"<b>{auv_name}</b> - WP #{h['id']}<br>Status: {h['pred']}<br>Conf: {h['conf']:.1f}%"
                folium.Marker(location=[h["lat"], h["lon"]], popup=pop, icon=folium.Icon(color=col, icon="info-sign")).add_to(m)

        if current_dict:
            for auv_name, cur in current_dict.items():
                if cur:
                    folium.CircleMarker(location=[cur["lat"], cur["lon"]], radius=8, color=colors[auv_name], fill=True, fill_color=colors[auv_name], fill_opacity=1.0, tooltip=f"Active: {auv_name}").add_to(m)
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

    if "swarm_running" not in st.session_state:
        st.session_state.swarm_running = False
    if "swarm_step" not in st.session_state:
        st.session_state.swarm_step = 0
    if "swarm_history" not in st.session_state:
        st.session_state.swarm_history = {"AUV-Alpha": [], "AUV-Bravo": [], "AUV-Charlie": []}
    if "telemetry_logs" not in st.session_state:
        st.session_state.telemetry_logs = []

    if btn_rth:
        st.session_state.swarm_running = False
        st.error("🚨 EMERGENCY RTH ACTIVATED FOR ALL SWARM UNITS! Returning to base.")
    elif btn_start:
        st.session_state.swarm_running = True
        st.session_state.swarm_step = 0
        st.session_state.swarm_history = {"AUV-Alpha": [], "AUV-Bravo": [], "AUV-Charlie": []}
        st.session_state.telemetry_logs = []
        st.rerun()

    if st.session_state.swarm_running:
        step = st.session_state.swarm_step
        if step < num_targets:
            current_active = {}
            for auv_name, pts in auv_routes.items():
                wp = pts[step]
                current_active[auv_name] = wp

                _, pred_label, conf = generate_realtime_sonar_scan(wp["seed"] + int(time.time() % 10))
                conf = max(40.0, conf - (turbidity * 0.8))

                st.session_state.swarm_history[auv_name].append({
                    "id": wp["id"], "lat": wp["lat"], "lon": wp["lon"], "pred": pred_label, "conf": conf
                })

                st.session_state.mission_records.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Unit": auv_name,
                    "Waypoint": f"WP-{wp['id']}",
                    "Latitude": f"{wp['lat']:.5f}",
                    "Longitude": f"{wp['lon']:.5f}",
                    "Classification": pred_label,
                    "Confidence": f"{conf:.2f}%"
                })

            st.session_state.telemetry_logs.append({
                "Step": f"Step-{step+1}",
                "Alpha Depth": 42.0 + np.sin(step)*2,
                "Bravo Depth": 44.5 + np.cos(step)*2,
                "Charlie Depth": 40.2 + np.sin(step+1)*1.5
            })

            m_lat.metric("Swarm Status", "🔴 ACTIVE SWARM")
            m_lon.metric("Current Velocity", f"{current_speed} Knots")
            m_batt.metric("Turbidity", f"{turbidity} NTU")
            m_depth.metric("Active Waypoint", f"{step+1} / {num_targets}")
            m_stat.metric("Network Link", "99.4% Stable")

            m_obj = render_swarm_map(st.session_state.swarm_history, current_active)
            with map_slot.container():
                st.markdown("### 🌐 Live Multi-AUV Swarm Map")
                components.html(m_obj._repr_html_(), height=420)

            with graph_slot.container():
                st.markdown("### 📈 Swarm Depth Telemetry")
                df_tel = pd.DataFrame(st.session_state.telemetry_logs).set_index("Step")
                st.line_chart(df_tel, height=330, color=["#00F5D4", "#FFD166", "#0EA5E9"])

            time.sleep(sim_delay)
            st.session_state.swarm_step += 1
            st.rerun()
        else:
            st.session_state.swarm_running = False
            st.success("🎉 Multi-AUV Swarm Mission Completed Successfully!")
            m_obj = render_swarm_map(st.session_state.swarm_history)
            with map_slot.container():
                components.html(m_obj._repr_html_(), height=420)
    else:
        m_lat.metric("Swarm Status", "STANDBY")
        m_lon.metric("Current Velocity", f"{current_speed} Knots")
        m_batt.metric("Turbidity", f"{turbidity} NTU")
        m_depth.metric("Active Waypoint", "0 / 0")
        m_stat.metric("Network Link", "Ready")

        m_obj = render_swarm_map({"AUV-Alpha": [], "AUV-Bravo": [], "AUV-Charlie": []})
        with map_slot.container():
            st.markdown("### 🌐 Multi-AUV Swarm Map (Standby)")
            components.html(m_obj._repr_html_(), height=420)

        with graph_slot.container():
            st.markdown("### 📈 Swarm Depth Telemetry (Standby)")
            dummy_df = pd.DataFrame({"Alpha": [42.0, 43.0], "Bravo": [44.5, 45.0], "Charlie": [40.2, 41.0]}, index=["Step-1", "Step-2"])
            st.line_chart(dummy_df, height=330, color=["#00F5D4", "#FFD166", "#0EA5E9"])

# ==========================================
# VIEW 2: TARGET ACOUSTIC SCAN & WATERFALL & AUDIO
# ==========================================
elif nav_choice == "📡 Target Acoustic Scan & Threat Action":
    st.title("📡 Target Acoustic Scan, Waterfall Display & Threat Action")
    st.write("Real-time generated acoustic target scans with Side-Scan Waterfall, Dynamic FFT, and Live Hydrophone Audio.")

    seed_slider = st.slider("Select Live Scan Sector ID:", 1, 50, 1)
    raw_pil, prediction, conf = generate_realtime_sonar_scan(seed_slider)
    img_np = np.array(raw_pil)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    mean_i = float(np.mean(gray_img)) + (turbidity * 0.5)
    std_i = float(np.std(gray_img))
    snr = (mean_i / (std_i + 1e-5)) * (1.0 / (current_speed * 0.1 + 0.9))
    conf = min(99.8, max(45.0, conf - (turbidity * 0.6)))

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("Raw Sonar Scan")
        st.image(raw_pil, use_container_width=True)

    with col_a2:
        st.subheader("🌊 Side-Scan Sonar Waterfall Display")
        resized_wf = cv2.resize(gray_img, (224, 224))
        waterfall_img = cv2.applyColorMap(resized_wf, cv2.COLORMAP_OCEAN)
        st.image(waterfall_img, channels="BGR", use_container_width=True)

    st.markdown("---")
    st.subheader("🔊 Hydrophone Acoustic Ping Simulator")
    is_mine_threat = (prediction == "Mine_Ordnance" and conf > 70.0)
    ping_freq = 3800 if is_mine_threat else 1200
    
    audio_html = f"""
    <div style="background: #0F172A; padding: 15px; border-radius: 8px; border: 1px solid #334155; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <b style="color: {'#FF8888' if is_mine_threat else '#00F5D4'};">Acoustic Signature Frequency: {ping_freq} Hz</b><br>
            <span style="color: #94A3B8; font-size: 13px;">Click to emit real-time hydrophone ping sound wave.</span>
        </div>
        <button onclick="playPing({ping_freq})" style="background: {'#EF4444' if is_mine_threat else '#00F5D4'}; color: #000; border: none; padding: 10px 20px; font-weight: bold; border-radius: 5px; cursor: pointer; font-family: 'Orbitron', sans-serif;">🔊 EMIT PING</button>
    </div>
    <script>
    function playPing(freq) {{
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.5);
    }}
    </script>
    """
    components.html(audio_html, height=85)

    st.markdown("---")
    st.subheader("🎵 Acoustic Frequency Spectrum (Dynamic FFT Analysis)")
    f_transform = np.fft.fft2(gray_img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    h_sz, w_sz = magnitude_spectrum.shape
    freq_profile = magnitude_spectrum[h_sz // 2, :]
    freqs = np.linspace(100, 5000, len(freq_profile))
    
    fig_fft = go.Figure(data=go.Scatter(
        x=freqs, y=freq_profile, mode='lines', 
        line=dict(color='#00F5D4', width=2), fill='tozeroy'
    ))
    fig_fft.update_layout(
        paper_bgcolor='#080D1A', plot_bgcolor='#0F172A',
        font=dict(color='#FFFFFF'), margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title='Frequency (Hz)', gridcolor='#334155'),
        yaxis=dict(title='Amplitude (dB)', gridcolor='#334155'),
        height=240
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Diagnostics & Dynamic Threat Warnings")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Classification", prediction.replace("_", " "))
    k2.metric("AI Confidence", f"{conf:.2f}%")
    k3.metric("SNR Ratio", f"{snr:.2f}")
    k4.metric("Turbidity Factor", f"{turbidity} NTU")

    max_fft_val = float(np.max(freq_profile))

    if prediction == "Mine_Ordnance" and conf > 70.0:
        st.markdown('<div class="hazard-box">🚨 CRITICAL THREAT ALERT: High-Probability Bottom Mine Signature Detected! Immediate Countermeasure Required.</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="action-box" style="margin-top: 15px;">
            <b>🛡️ DYNAMIC DEFENSE ACTION PLAN (Sector ID: #{seed_slider}):</b><br>
            • Acoustic Frequency Peak: <b>{max_fft_val:.1f} dB</b> | Current Drift: <b>{current_speed} Knots</b><br>
            1. Establish 1000m Maritime Exclusion Zone around coordinates.<br>
            2. Deploy Remotely Operated Vehicle (ROV) for optical ID verification.<br>
            3. Dispatch EOD (Explosive Ordnance Disposal) team for neutralisation.
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
    Z = np.sin(np.sqrt(X**2 + Y**2)) * (10 + current_speed) - 45

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

# ==========================================
# VIEW 4: EXECUTIVE AUDIT & REPORT GENERATOR
# ==========================================
elif nav_choice == "📊 Executive Audit Trail & Report":
    st.title("📊 Tactical Mission Audit Trail & Military Report Generator")

    if len(st.session_state.mission_records) > 0:
        df_audit = pd.DataFrame(st.session_state.mission_records)
        st.dataframe(df_audit, use_container_width=True)

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            csv_data = df_audit.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Tactical Report",
                data=csv_data,
                file_name=f"Naval_MCM_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with col_rep2:
            html_report = f"""
            <html>
            <head><style>body {{ font-family: monospace; color: #000; padding: 20px; }} h2 {{ color: #003366; }} table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }} th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 12px; }} th {{ background: #003366; color: #fff; }}</style></head>
            <body>
                <h2>NAVAL SUBSEA ACOUSTIC COMMAND - MISSION REPORT</h2>
                <p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><b>Environmental Status:</b> Turbidity: {turbidity} NTU | Current: {current_speed} Knots</p>
                {df_audit.to_html(index=False)}
            </body>
            </html>
            """
            st.download_button(
                label="📄 Download Formatted HTML/PDF Report",
                data=html_report.encode('utf-8'),
                file_name=f"Naval_Mission_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
    else:
        st.info("No mission records found yet. Run a multi-AUV swarm mission from View 1 first.")
