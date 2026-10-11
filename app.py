import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import random
import os
import plotly.graph_objects as go

# --- 1. Page Configuration & Advanced CSS Injection ---
st.set_page_config(page_title="TriageNet - ICU Command Center", layout="wide", initial_sidebar_state="expanded")

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
        background-color: rgba(11, 15, 25, 0.95) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    /* Radio Button Custom Styling */
    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        background-color: rgba(19, 28, 46, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 10px 14px !important;
        border-radius: 12px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease-in-out !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        border-color: rgba(239, 68, 68, 0.5) !important;
        background-color: rgba(239, 68, 68, 0.08) !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"]:checked + div {
        background-color: #ef4444 !important;
        border-color: #ef4444 !important;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.8) !important;
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
        min-height: 120px;
    }
    .metric-card h4 { margin: 0; font-size: 13px; color: #94a3b8; font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-card h2 { margin: 10px 0 0 0; font-size: 36px; font-weight: 800; font-family: 'JetBrains Mono', monospace;}
    
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; vertical-align: middle;}
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; vertical-align: middle;}
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; vertical-align: middle;}
    
    header {background-color: transparent !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Interactive Cursor Injection ---
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

# --- 2. Database & Persistent Storage Functions ---
DB_FILE = 'patients_db.csv'
BASE_START_ID = 2717270

def load_database():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_csv(DB_FILE, dtype=str).fillna("").to_dict('records')
        except Exception:
            return []
    return []

def save_database(patients_list):
    try:
        pd.DataFrame(patients_list).to_csv(DB_FILE, index=False)
    except Exception as e:
        st.error(f"Failed to save to database: {e}")

def clean_and_persist_patients(patients_list):
    updated = False
    current_id = BASE_START_ID
    doctors = ["Dr. Rajesh Sharma (Cardiologist)", "Dr. Priya Nair (Endocrinologist)", "Dr. Vikram Mehta (Emergency)"]
    
    for p in patients_list:
        raw_id = str(p.get('id', '')).strip()
        try:
            val = int(raw_id)
            if val < BASE_START_ID or val > 2719999 or len(raw_id) > 7:
                p['id'] = str(current_id)
                updated = True
                current_id += 1
            else:
                if val >= current_id:
                    current_id = val + 1
        except ValueError:
            p['id'] = str(current_id)
            updated = True
            current_id += 1
            
        if not p.get('doctor'):
            p['doctor'] = random.choice(doctors)
            updated = True
            
    if updated or not os.path.exists(DB_FILE):
        save_database(patients_list)
        
    return patients_list

def get_next_sequential_id(patients_list):
    existing_ids = []
    for p in patients_list:
        try:
            p_id = int(str(p.get('id', '')).strip())
            if p_id >= BASE_START_ID:
                existing_ids.append(p_id)
        except ValueError:
            pass
    if existing_ids:
        return str(max(existing_ids) + 1)
    return str(BASE_START_ID)

# Reload and clean database from CSV to keep state persistent across refreshes
st.session_state.patients = clean_and_persist_patients(load_database())

# --- 3. Load ML Models ---
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

# --- 4. Sidebar Assessment Panel ---
st.sidebar.markdown('## 🩺 AI Assessment Panel')
st.sidebar.caption("Select Diagnostic Tool:")
assessment_type = st.sidebar.radio("", ["Track A: Heart Disease", "Track B: Diabetes", "Track C: Text Triage"], label_visibility="collapsed")
st.sidebar.write("---")

next_auto_id = get_next_sequential_id(st.session_state.patients)
patient_id = st.sidebar.text_input("Patient ID Number", next_auto_id)

col_room, col_bed = st.sidebar.columns(2)
with col_room:
    room_no = st.selectbox("Select ICU Room", ["ICU 1", "ICU 2", "ICU 3", "ICU 4"])
with col_bed:
    bed_no = st.selectbox("Select Bed No.", ["Bed 1", "Bed 2", "Bed 3", "Bed 4", "Bed 5"])

occupied_beds_set = {(p.get('room'), p.get('bed')) for p in st.session_state.patients}
is_bed_taken = (room_no, bed_no) in occupied_beds_set

if is_bed_taken:
    st.sidebar.error(f"⚠️ {room_no} - {bed_no} is ALREADY OCCUPIED!")

if assessment_type == "Track A: Heart Disease":
    assigned_doc = "Dr. Rajesh Sharma (Cardiologist)"
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (1=M, 0=F)", [1, 0])
    
    # Replaced numerical selection with medical term dropdowns
    cp_label_map = {
        "typical angina": 1,
        "atypical angina": 2,
        "non-anginal": 3,
        "asymptomatic": 4
    }
    cp_label = st.sidebar.selectbox("Chest Pain Type", list(cp_label_map.keys()))
    cp = cp_label_map[cp_label]

    trestbps = st.sidebar.slider("Resting BP", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Sugar >120?", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.sidebar.slider("Max HR", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Angina", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope (1-3)", [1, 2, 3])
    ca = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
    
    # Replaced numerical selection with medical term dropdowns
    thal_label_map = {
        "normal": 3,
        "fixed defect": 6,
        "reversible defect": 7
    }
    thal_label = st.sidebar.selectbox("Thalassemia", list(thal_label_map.keys()))
    thal = thal_label_map[thal_label]
    
    if st.sidebar.button("Run AI & Add Patient", disabled=is_bed_taken):
        if models['heart']:
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            pred = models['heart'].predict(input_data)[0]
            prob = models['heart'].predict_proba(input_data)[0]
            risk_level = "CRITICAL" if pred == 1 and prob[1] > 0.8 else ("HIGH" if pred == 1 else "ROUTINE")
            
            new_patient = {
                "id": patient_id.strip() if patient_id.strip() else next_auto_id,
                "room": room_no, "bed": bed_no, "risk": risk_level, "score": str(int((prob[1] if pred == 1 else prob[0]) * 10)),
                "hr": str(thalach), "rr": str(random.randint(12, 25)), "spo2": str(random.randint(90, 99)), 
                "bp": f"{trestbps}/80", "temp": "37.0", "note": f"Heart AI - Conf: {prob[1]*100:.1f}%",
                "doctor": assigned_doc
            }
            st.session_state.patients.append(new_patient)
            save_database(st.session_state.patients)
            st.rerun()

elif assessment_type == "Track B: Diabetes":
    assigned_doc = "Dr. Priya Nair (Endocrinologist)"
    preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
    gluc = st.sidebar.slider("Glucose", 0, 250, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    ins = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree", 0.0, 3.0, 0.5)
    age_d = st.sidebar.slider("Age", 21, 100, 33)
    
    if st.sidebar.button("Run AI & Add Patient", disabled=is_bed_taken):
        if models['diabetes']:
            input_data = pd.DataFrame([[preg, gluc, bp, skin, ins, bmi, dpf, age_d]], 
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            pred = models['diabetes'].predict(input_data)[0]
            prob = models['diabetes'].predict_proba(input_data)[0]
            risk_level = "HIGH" if pred == 1 else "ROUTINE"
            
            new_patient = {
                "id": patient_id.strip() if patient_id.strip() else next_auto_id,
                "room": room_no, "bed": bed_no, "risk": risk_level, "score": str(int((prob[1] if pred == 1 else prob[0]) * 10)),
                "hr": "75", "rr": "16", "spo2": "98", "bp": f"{bp}/60", "temp": "36.8", "note": f"Diabetes AI - Conf: {prob[1]*100:.1f}%",
                "doctor": assigned_doc
            }
            st.session_state.patients.append(new_patient)
            save_database(st.session_state.patients)
            st.rerun()

elif assessment_type == "Track C: Text Triage":
    assigned_doc = "Dr. Vikram Mehta (Emergency)"
    clinical_text = st.sidebar.text_area("Patient Notes / Symptoms:")
    if st.sidebar.button("Run AI & Add Patient", disabled=is_bed_taken):
        if models['text'] and clinical_text:
            pred = models['text'].predict([clinical_text])[0]
            prob = models['text'].predict_proba([clinical_text])[0]
            risk_level = "CRITICAL" if pred == 1 else "ROUTINE"
            
            new_patient = {
                "id": patient_id.strip() if patient_id.strip() else next_auto_id,
                "room": room_no, "bed": bed_no, "risk": risk_level, "score": str(int((prob[1] if pred == 1 else prob[0]) * 10)),
                "hr": "N/A", "rr": "N/A", "spo2": "N/A", "bp": "N/A", "temp": "N/A", "note": "NLP Text Triage Used",
                "doctor": assigned_doc
            }
            st.session_state.patients.append(new_patient)
            save_database(st.session_state.patients)
            st.rerun()

# --- Sort Patients by Risk Priority (CRITICAL -> HIGH -> ROUTINE) ---
risk_priority = {"CRITICAL": 0, "HIGH": 1, "ROUTINE": 2}
sorted_patients = sorted(
    st.session_state.patients, 
    key=lambda x: (risk_priority.get(x.get('risk', 'ROUTINE'), 2), -int(x.get('score', 0)))
)

TOTAL_ICU_BEDS = 20
occupied_beds = len(sorted_patients)
occupancy_rate = min(int((occupied_beds / TOTAL_ICU_BEDS) * 100), 100)

# --- 5. Dashboard View ---
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>TriageNet</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-bottom: 0px;'>❤️ ICU Command Center</h2>", unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; font-size: 14px; margin-top: 5px;">Powered by Neural Triage v3.4 • <span style="color: #3b82f6;">●</span> Real-time telemetry</p>', unsafe_allow_html=True)

total_patients = len(sorted_patients)
critical_count = sum(1 for p in sorted_patients if p.get('risk') == 'CRITICAL')
high_count = sum(1 for p in sorted_patients if p.get('risk') == 'HIGH')
low_count = sum(1 for p in sorted_patients if p.get('risk') == 'ROUTINE')

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-card"><h4>PATIENTS TODAY</h4><div><h2 style="color:white; display:inline;">{total_patients}</h2><span class="badge-green">+12%</span></div></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h4>CRITICAL RISK</h4><div><h2 style="color:#ef4444; display:inline;">{critical_count}</h2><span class="badge-red">-3%</span></div></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h4>HIGH RISK</h4><div><h2 style="color:#f59e0b; display:inline;">{high_count}</h2><span class="badge-amber">MONITORED</span></div></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><h4>BED OCCUPANCY</h4><div><h2 style="color:white; display:inline;">{occupancy_rate}%</h2><span class="badge-green">{occupied_beds}/{TOTAL_ICU_BEDS} BEDS</span></div></div>', unsafe_allow_html=True)

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
    st.markdown('#### Patient Status Feed &nbsp;&nbsp;<span style="color: #3b82f6; font-size: 12px; font-weight: normal;">● Prioritized by Severity</span>', unsafe_allow_html=True)
    
    with st.expander("🗑️ Patient Discharge / Management Panel"):
        if sorted_patients:
            patient_options = [f"ID: {p.get('id')} ({p.get('risk', 'ROUTINE')} - {p.get('room', 'ICU 1')}/{p.get('bed', 'Bed 1')})" for p in sorted_patients]
            selected_to_delete = st.selectbox("Select Patient to Discharge:", patient_options)
            if st.button("Discharge Selected Patient"):
                selected_id = selected_to_delete.split(" ")[1]
                st.session_state.patients = [p for p in st.session_state.patients if str(p.get('id')) != str(selected_id)]
                save_database(st.session_state.patients)
                st.success(f"Patient {selected_id} discharged successfully.")
                st.rerun()
        else:
            st.info("No active patients to manage.")

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
            
            @keyframes heartbeat {
                0%, 100% { transform: scale(1); }
                15% { transform: scale(1.25); }
                30% { transform: scale(1); }
                45% { transform: scale(1.25); }
                60% { transform: scale(1); }
            }
            .animate-heartbeat { display: inline-block; animation: heartbeat 1.2s infinite; }
            
            @keyframes breathing {
                0%, 100% { transform: scale(1); opacity: 0.7; }
                50% { transform: scale(1.15); opacity: 1; }
            }
            .animate-breathing { display: inline-block; animation: breathing 3s infinite ease-in-out; }
        </style>
    </head>
    <body>
    <div class="space-y-4">
    """
    for idx, p in enumerate(sorted_patients):
        risk = p.get('risk', 'ROUTINE')
        room = p.get('room', f"ICU {(idx // 5) + 1}")
        bed = p.get('bed', f"Bed {(idx % 5) + 1}")
        doctor_name = p.get('doctor', 'Dr. Rajesh Sharma (Cardiologist)')
        
        if risk == 'CRITICAL':
            border_color, tag_bg, text_color, glow = "border-l-red-500", "bg-red-500 text-white animate-pulse", "text-red-400", "shadow-lg shadow-red-900/20"
        elif risk == 'HIGH':
            border_color, tag_bg, text_color, glow = "border-l-amber-500", "bg-amber-500 text-black", "text-amber-400", "shadow-lg shadow-amber-900/20"
        else:
            border_color, tag_bg, text_color, glow = "border-l-emerald-500", "bg-emerald-500 text-black", "text-emerald-400", ""
            
        html_code += f"""
        <div class="glass-panel rounded-2xl p-6 border-l-4 {border_color} {glow} relative overflow-hidden">
            <div class="flex items-center justify-between mb-4">
                <div class="flex items-center space-x-3">
                    <span class="px-3 py-1 rounded-lg text-xs font-bold uppercase {tag_bg}">{risk}</span>
                    <h3 class="text-lg font-bold font-[JetBrains Mono] tracking-wide text-white">Patient #{p.get('id', 'N/A')}</h3>
                    <span class="text-xs bg-slate-800/90 text-blue-400 px-2.5 py-1 rounded-md border border-slate-700 font-medium">🚪 {room}</span>
                    <span class="text-xs bg-slate-800/90 text-emerald-400 px-2.5 py-1 rounded-md border border-slate-700 font-medium">🛌 {bed}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="text-xs text-slate-400 uppercase">AI Score</span>
                    <span class="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 {text_color} font-[JetBrains Mono] font-bold flex items-center justify-center text-sm">{p.get('score', '0')}</span>
                </div>
            </div>
            <div class="grid grid-cols-5 gap-2 py-3 my-2 border-y border-slate-800/80">
                <div><p class="text-xs text-slate-400 mb-1"><span class="animate-heartbeat text-red-500">❤️</span> HR</p><p class="text-lg font-[JetBrains Mono] font-bold {text_color}">{p.get('hr', 'N/A')}</p></div>
                <div><p class="text-xs text-slate-400 mb-1">🫁 RR</p><p class="text-lg font-[JetBrains Mono] font-bold text-white">{p.get('rr', 'N/A')}</p></div>
                <div><p class="text-xs text-slate-400 mb-1"><span class="animate-breathing text-blue-400">🩸</span> SPO2</p><p class="text-lg font-[JetBrains Mono] font-bold {text_color}">{p.get('spo2', 'N/A')}</p></div>
                <div><p class="text-xs text-slate-400 mb-1">🩺 BP</p><p class="text-lg font-[JetBrains Mono] font-bold text-white">{p.get('bp', 'N/A')}</p></div>
                <div><p class="text-xs text-slate-400 mb-1">🌡️ TEMP</p><p class="text-lg font-[JetBrains Mono] font-bold text-white">{p.get('temp', 'N/A')}</p></div>
            </div>
            <div class="mt-3 flex items-center justify-between gap-2">
                <div class="text-xs {text_color} bg-slate-900/50 p-2 rounded-lg border border-slate-800/50 flex-1"><strong>AI Note:</strong> {p.get('note', '')}</div>
                <div class="text-xs text-indigo-300 bg-indigo-950/60 p-2 rounded-lg border border-indigo-800/50 font-medium">👨‍⚕️ <strong>Assigned:</strong> {doctor_name}</div>
            </div>
        </div>
        """
    html_code += "</div></body></html>"
    components.html(html_code, height=600, scrolling=True)

with right_col:
    st.markdown("#### 🧬 Patient Neural Grid")
    
    grid_html = '<div style="background: rgba(19, 28, 46, 0.7); backdrop-filter: blur(12px); padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.07); display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-bottom: 20px;">'
    for idx, p in enumerate(sorted_patients):
        risk = p.get('risk', 'ROUTINE')
        room = p.get('room', f"ICU {(idx // 5) + 1}")
        bed = p.get('bed', f"Bed {(idx % 5) + 1}")
        
        if risk == 'CRITICAL':
            color, shadow, anim = "#ef4444", "rgba(239, 68, 68, 0.6)", "animation: pulse 1.5s infinite;"
        elif risk == 'HIGH':
            color, shadow, anim = "#f59e0b", "rgba(245, 158, 11, 0.4)", ""
        else:
            color, shadow, anim = "#10b981", "rgba(16, 185, 129, 0.2)", ""
            
        grid_html += f'<div title="Patient #{p.get("id")} ({room} - {bed})" style="width: 14px; height: 14px; background-color: {color}; border-radius: 50%; box-shadow: 0 0 10px {shadow}; {anim}"></div>'
    
    grid_html += f"""
        <style>
        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
            70% {{ transform: scale(1.2); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        </style>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: -10px; margin-bottom: 25px; font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
    <span><span style="color:#ef4444; font-size: 14px;">●</span> CRITICAL: {critical_count}</span>
    <span><span style="color:#f59e0b; font-size: 14px;">●</span> HIGH: {high_count}</span>
    <span><span style="color:#10b981; font-size: 14px;">●</span> ROUTINE: {low_count}</span></div>
    """
    
    st.markdown(grid_html, unsafe_allow_html=True)

    # --- Interactive Plotly Bed Occupancy Chart ---
    st.markdown("#### 🏥 Room Bed Occupancy")

    color_map = {
        "CRITICAL": "#ef4444",  # Red
        "HIGH": "#f59e0b",      # Yellow / Amber
        "ROUTINE": "#3b82f6",   # Blue
        "Available": "#10b981"  # Green
    }

    fig = go.Figure()

    for bed_idx in range(1, 6):
        bed_name = f"Bed {bed_idx}"
        
        x_rooms = []
        y_vals = []
        colors = []
        hover_texts = []

        for room_idx in range(1, 5):
            room_name = f"ICU {room_idx}"
            matched_patient = next(
                (p for p in sorted_patients if p.get('room') == room_name and p.get('bed') == bed_name), 
                None
            )
            
            x_rooms.append(room_name)
            y_vals.append(1)
            
            if matched_patient:
                risk = matched_patient.get('risk', 'ROUTINE')
                p_id = matched_patient.get('id', 'N/A')
                doc = matched_patient.get('doctor', 'Unassigned')
                colors.append(color_map.get(risk, "#3b82f6"))
                hover_texts.append(
                    f"<b>{room_name}</b> - <b>{bed_name}</b><br>"
                    f"Patient ID: #{p_id}<br>"
                    f"Condition: <b>{risk}</b><br>"
                    f"Doctor: {doc}"
                )
            else:
                colors.append(color_map["Available"])
                hover_texts.append(
                    f"<b>{room_name}</b> - <b>{bed_name}</b><br>"
                    f"Status: <b>Available</b>"
                )

        fig.add_trace(go.Bar(
            x=x_rooms,
            y=y_vals,
            marker_color=colors,
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False
        ))

    for status, color in color_map.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker_color=color,
            name=status,
            showlegend=True
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=False, color="#94a3b8")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 5], dtick=1, title="Total Beds")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- Department Workload HTML Section ---
    st.markdown("#### 📊 Dept Workload")
    components.html("""
        <div style="background: rgba(19, 28, 46, 0.7); backdrop-filter: blur(12px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.07); font-family: sans-serif;">
            <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px; margin-top: 0;">Emergency</p>
            <div style="background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; height: 10px; margin-bottom: 15px;">
                <div style="background-color: #ef4444; width: 85%; height: 100%; border-radius: 10px;"></div>
            </div>
            
            <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px; margin-top: 0;">Cardiology</p>
            <div style="background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; height: 10px; margin-bottom: 15px;">
                <div style="background-color: #3b82f6; width: 60%; height: 100%; border-radius: 10px;"></div>
            </div>
            
            <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px; margin-top: 0;">Neurology</p>
            <div style="background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; height: 10px;">
                <div style="background-color: #10b981; width: 40%; height: 100%; border-radius: 10px;"></div>
            </div>
        </div>
    """, height=180)