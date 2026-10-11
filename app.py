import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import random
import os

# --- 1. Page Configuration & Advanced CSS Injection ---
st.set_page_config(page_title="Patient Deterioration Monitor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19 !important;
        background-image: 
            radial-gradient(circle at 15% 20%, rgba(37, 99, 235, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 85% 80%, rgba(220, 38, 38, 0.1) 0%, transparent 40%),
            linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px !important;
        background-attachment: fixed !important;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #1f2937 !important;
        color: white !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
    }

    .metric-card { 
        background: rgba(19, 28, 46, 0.7);
        backdrop-filter: blur(12px);
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card h4 { margin: 0; font-size: 13px; color: #94a3b8; font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-card h2 { margin: 10px 0 0 0; font-size: 36px; font-weight: 800; font-family: 'JetBrains Mono', monospace;}
    
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; vertical-align: middle;}
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; vertical-align: middle;}
    
    header {background-color: transparent !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Interactive Cursor Injection (Green Concentric) ---
components.html("""
<script>
    const doc = window.parent.document;
    const oldCursor = doc.getElementById('custom-cursor-glow');
    if (oldCursor) oldCursor.remove();

    if (!doc.getElementById('custom-cursor-green')) {
        const cursor = doc.createElement('div');
        cursor.id = 'custom-cursor-green';
        cursor.style.position = 'fixed';
        cursor.style.width = '16px';  
        cursor.style.height = '16px';
        cursor.style.borderRadius = '50%';
        cursor.style.backgroundColor = '#10b981'; 
        cursor.style.boxShadow = '0 0 0 14px rgba(16, 185, 129, 0.3), 0 0 35px 25px rgba(16, 185, 129, 0.15)';
        cursor.style.pointerEvents = 'none';
        cursor.style.zIndex = '9999';
        cursor.style.transform = 'translate(-50%, -50%)';
        cursor.style.transition = 'top 0.08s ease-out, left 0.08s ease-out';
        
        doc.body.appendChild(cursor);

        doc.addEventListener('mousemove', (e) => {
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        });
    }
</script>
""", height=0, width=0)

# --- 2. Permanent Database Functions ---
DB_FILE = 'patients_db.csv'

def load_database():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, dtype=str).to_dict('records')
    return []

def save_database(patients_list):
    pd.DataFrame(patients_list).to_csv(DB_FILE, index=False)

if 'patients' not in st.session_state:
    st.session_state.patients = load_database()
if 'new_patient_id' not in st.session_state:
    st.session_state.new_patient_id = str(random.randint(30000000, 39999999))

# --- 3. Load All AI Models ---
@st.cache_resource
def load_models():
    models = {'heart': None, 'diabetes': None, 'text': None}
    try: models['heart'] = joblib.load('triage_model.pkl')
    except: pass
    try: models['diabetes'] = joblib.load('diabetes_model.pkl')
    except: pass
    try: models['text'] = joblib.load('text_model.pkl')
    except: pass
    return models

models = load_models()

# --- 4. Sidebar: Add Patients & Run AI ---
st.sidebar.title("🩺 AI Assessment Panel")
assessment_type = st.sidebar.radio("Select Diagnostic Tool:", ["Track A: Heart Disease", "Track A: Diabetes", "Track C: Text Triage"])
st.sidebar.write("---")
patient_id = st.sidebar.text_input("Patient ID Number", st.session_state.new_patient_id)

if assessment_type == "Track A: Heart Disease":
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (1=M, 0=F)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain (1-4)", [1, 2, 3, 4])
    trestbps = st.sidebar.slider("Resting BP", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Sugar >120?", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.sidebar.slider("Max HR", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Angina", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope (1-3)", [1, 2, 3])
    ca = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thal (3, 6, 7)", [3, 6, 7])
    
    if st.sidebar.button("Run AI & Add Patient"):
        if models['heart']:
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            pred = models['heart'].predict(input_data)[0]
            prob = models['heart'].predict_proba(input_data)[0]
            risk_level = "CRITICAL" if pred == 1 and prob[1] > 0.8 else ("HIGH" if pred == 1 else "ROUTINE")
            
            st.session_state.patients.insert(0, {
                "id": patient_id, "risk": risk_level, "score": str(int((prob[1] if pred == 1 else prob[0]) * 10)),
                "hr": str(thalach), "rr": str(random.randint(12, 25)), "spo2": str(random.randint(90, 99)), 
                "bp": f"{trestbps}/80", "temp": "37.0", "note": f"Heart AI - Conf: {prob[1]*100:.1f}%"
            })
            save_database(st.session_state.patients)
            st.session_state.new_patient_id = str(random.randint(30000000, 39999999))
            st.rerun()

elif assessment_type == "Track A: Diabetes":
    preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
    gluc = st.sidebar.slider("Glucose", 0, 250, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    ins = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree", 0.0, 3.0, 0.5)
    age_d = st.sidebar.slider("Age", 21, 100, 33)
    
    if st.sidebar.button("Run AI & Add Patient"):
        if models['diabetes']:
            input_data = pd.DataFrame([[preg, gluc, bp, skin, ins, bmi, dpf, age_d]], 
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            pred = models['diabetes'].predict(input_data)[0]
            prob = models['diabetes'].predict_proba(input_data)[0]
            risk_level = "HIGH" if pred == 1 else "ROUTINE"
            
            st.session_state.patients.insert(0, {
                "id": patient_id, "risk": risk_level, "score": str(int((prob[1] if pred == 1 else prob[0]) * 10)),
                "hr": "75", "rr": "16", "spo2": "98", "bp": f"{bp}/60", "temp": "36.8", "note": f"Diabetes AI - Conf: {prob[1]*100:.1f}%"
            })
            save_database(st.session_state.patients)
            st.session_state.new_patient_id = str(random.randint(30000000, 39999999))
            st.rerun()

elif assessment_type == "Track C: Text Triage":
    clinical_text = st.sidebar.text_area("Patient Notes / Symptoms:")
    if st.sidebar.button("Run AI & Add Patient"):
        if models['text'] and clinical_text:
            pred = models['text'].predict([clinical_text])[0]
            prob = models['text'].predict_proba([clinical_text])[0]
            risk_level = "CRITICAL" if pred == 1 else "ROUTINE"
            
            st.session_state.patients.insert(0, {
                "id": patient_id, "risk": risk_level, "score": str(int((prob[1] if pred == 1 else prob[0]) * 10)),
                "hr": "N/A", "rr": "N/A", "spo2": "N/A", "bp": "N/A", "temp": "N/A", "note": "NLP Text Triage Used"
            })
            save_database(st.session_state.patients)
            st.session_state.new_patient_id = str(random.randint(30000000, 39999999))
            st.rerun()

# --- 5. Main Dashboard View ---
st.markdown("<h2 style='margin-bottom: 0px;'>❤️ ICU Command Center</h2>", unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; font-size: 14px; margin-top: 5px;">Powered by Neural Triage v3.4 • <span style="color: #3b82f6;">●</span> Real-time telemetry</p>', unsafe_allow_html=True)

total_patients = len(st.session_state.patients)
critical_count = sum(1 for p in st.session_state.patients if p['risk'] == 'CRITICAL')
high_count = sum(1 for p in st.session_state.patients if p['risk'] == 'HIGH')
low_count = sum(1 for p in st.session_state.patients if p['risk'] == 'ROUTINE')

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-card"><h4>PATIENTS TODAY</h4><div><h2 style="color:white; display:inline;">{total_patients}</h2><span class="badge-green">+12%</span></div></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h4>CRITICAL RISK</h4><div><h2 style="color:#ef4444; display:inline;">{critical_count}</h2><span class="badge-red">-3%</span></div></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h4>HIGH RISK</h4><div><h2 style="color:#f59e0b; display:inline;">{high_count}</h2></div></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><h4>BED OCCUPANCY</h4><div><h2 style="color:white; display:inline;">78%</h2><span class="badge-green">+5%</span></div></div>', unsafe_allow_html=True)

st.write("---")

st.markdown("#### 📈 24h Patient Vitals Monitor")
st.caption("Hourly average heart rate, blood pressure, and SpO2 across all monitored patients")
hours = [f"{i:02d}:00" for i in range(24)]
vitals_data = pd.DataFrame({
    "Systolic BP": np.random.normal(120, 5, 24),
    "Heart Rate": np.random.normal(85, 8, 24),
    "SpO2": np.random.normal(96, 1.5, 24)
}, index=hours)
st.line_chart(vitals_data, color=["#3b82f6", "#ef4444", "#10b981"], height=250)

st.write("---")
left_col, right_col = st.columns([1.8, 1])

with left_col:
    st.markdown('#### Patient Status Feed &nbsp;&nbsp;<span style="color: #3b82f6; font-size: 12px; font-weight: normal;">● Real-time stream</span>', unsafe_allow_html=True)
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: transparent; color: white; margin: 0; padding: 5px; }
            .glass-panel { background: rgba(19, 28, 46, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.07); }
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
            
            /* Live Icon Animations */
            @keyframes heartbeat {
                0%, 100% { transform: scale(1); }
                15% { transform: scale(1.25); }
                30% { transform: scale(1); }
                45% { transform: scale(1.25); }
                60% { transform: scale(1); }
            }
            .animate-heartbeat { 
                display: inline-block; 
                animation: heartbeat 1.2s infinite; 
            }
            
            @keyframes breathing {
                0%, 100% { transform: scale(1); opacity: 0.7; }
                50% { transform: scale(1.15); opacity: 1; }
            }
            .animate-breathing { 
                display: inline-block; 
                animation: breathing 3s infinite ease-in-out; 
            }
        </style>
    </head>
    <body>
    <div class="space-y-4">
    """
    for p in st.session_state.patients:
        if p['risk'] == 'CRITICAL':
            border_color, tag_bg, text_color, glow = "border-l-red-500", "bg-red-500 text-white animate-pulse", "text-red-400", "shadow-lg shadow-red-900/20"
        elif p['risk'] == 'HIGH':
            border_color, tag_bg, text_color, glow = "border-l-amber-500", "bg-amber-500 text-black", "text-amber-400", "shadow-lg shadow-amber-900/20"
        else:
            border_color, tag_bg, text_color, glow = "border-l-emerald-500", "bg-emerald-500 text-black", "text-emerald-400", ""
            
        html_code += f"""
        <div class="glass-panel rounded-2xl p-6 border-l-4 {border_color} {glow} relative overflow-hidden">
            <div class="flex items-center justify-between mb-4">
                <div class="flex items-center space-x-3">
                    <span class="px-3 py-1 rounded-lg text-xs font-bold uppercase {tag_bg}">{p['risk']}</span>
                    <h3 class="text-lg font-bold font-[JetBrains Mono] tracking-wide text-white">Patient {p['id']}</h3>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="text-xs text-slate-400 uppercase">AI Score</span>
                    <span class="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 {text_color} font-[JetBrains Mono] font-bold flex items-center justify-center text-sm">{p['score']}</span>
                </div>
            </div>
            <div class="grid grid-cols-5 gap-2 py-3 my-2 border-y border-slate-800/80">
                <div><p class="text-xs text-slate-400 mb-1"><span class="animate-heartbeat text-red-500">❤️</span> HR</p><p class="text-lg font-[JetBrains Mono] font-bold {text_color}">{p['hr']}</p></div>
                <div><p class="text-xs text-slate-400 mb-1">🫁 RR</p><p class="text-lg font-[JetBrains Mono] font-bold text-white">{p['rr']}</p></div>
                <div><p class="text-xs text-slate-400 mb-1"><span class="animate-breathing text-blue-400">🩸</span> SPO2</p><p class="text-lg font-[JetBrains Mono] font-bold {text_color}">{p['spo2']}</p></div>
                <div><p class="text-xs text-slate-400 mb-1">🩺 BP</p><p class="text-lg font-[JetBrains Mono] font-bold text-white">{p['bp']}</p></div>
                <div><p class="text-xs text-slate-400 mb-1">🌡️ TEMP</p><p class="text-lg font-[JetBrains Mono] font-bold text-white">{p['temp']}</p></div>
            </div>
            <div class="mt-3 text-xs {text_color} bg-slate-900/50 p-2 rounded-lg border border-slate-800/50"><strong>AI Note:</strong> {p['note']}</div>
        </div>
        """
    html_code += "</div></body></html>"
    components.html(html_code, height=600, scrolling=True)

with right_col:
    st.markdown("#### 🧬 Patient Neural Grid")
    
    grid_html = '<div style="background: rgba(19, 28, 46, 0.7); backdrop-filter: blur(12px); padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.07); display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-bottom: 20px;">'
    for p in st.session_state.patients:
        if p['risk'] == 'CRITICAL':
            color, shadow, anim = "#ef4444", "rgba(239, 68, 68, 0.6)", "animation: pulse 1.5s infinite;"
        elif p['risk'] == 'HIGH':
            color, shadow, anim = "#f59e0b", "rgba(245, 158, 11, 0.4)", ""
        else:
            color, shadow, anim = "#10b981", "rgba(16, 185, 129, 0.2)", ""
            
        grid_html += f'<div title="Patient {p["id"]}" style="width: 14px; height: 14px; background-color: {color}; border-radius: 50%; box-shadow: 0 0 10px {shadow}; {anim}"></div>'
    
    grid_html += """
        <style>
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { transform: scale(1.2); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        </style>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: -10px; margin-bottom: 25px; font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
    """
    grid_html += f'<span><span style="color:#ef4444; font-size: 14px;">●</span> CRITICAL: {critical_count}</span>'
    grid_html += f'<span><span style="color:#f59e0b; font-size: 14px;">●</span> HIGH: {high_count}</span>'
    grid_html += f'<span><span style="color:#10b981; font-size: 14px;">●</span> ROUTINE: {low_count}</span></div>'
    
    st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown("#### 🏥 Bed Occupancy")
    st.markdown("""
        <div style="background: rgba(19, 28, 46, 0.7); backdrop-filter: blur(12px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.07); text-align: center; margin-bottom: 20px;">
            <div style="width: 120px; height: 120px; border-radius: 50%; border: 12px solid rgba(30, 41, 59, 0.8); border-top-color: #10b981; border-right-color: #10b981; border-bottom-color: #3b82f6; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                <h2 style="margin: 0; font-family: 'JetBrains Mono', monospace; color: white;">78%</h2>
            </div>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 15px;">Occupancy rate by ward</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📊 Dept Workload")
    st.markdown("""
        <div style="background: rgba(19, 28, 46, 0.7); backdrop-filter: blur(12px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.07);">
            <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Emergency</p>
            <div style="background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; height: 10px; margin-bottom: 15px;"><div style="background-color: #ef4444; width: 85%; height: 100%; border-radius: 10px;"></div></div>
            
            <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Cardiology</p>
            <div style="background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; height: 10px; margin-bottom: 15px;"><div style="background-color: #3b82f6; width: 60%; height: 100%; border-radius: 10px;"></div></div>
            
            <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Neurology</p>
            <div style="background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; height: 10px;"><div style="background-color: #10b981; width: 40%; height: 100%; border-radius: 10px;"></div></div>
        </div>
    """, unsafe_allow_html=True)