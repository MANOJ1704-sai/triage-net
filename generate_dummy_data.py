import pandas as pd

# 1. Define a realistic set of dummy patients for the hackathon evaluation
dummy_patients = [
    {"id": "30049356", "risk": "CRITICAL", "score": "9", "hr": "135", "rr": "28", "spo2": "88", "bp": "85/50", "temp": "39.1", "note": "Heart AI - Conf: 92.4% (Severe Tachycardia)"},
    {"id": "30006565", "risk": "HIGH", "score": "7", "hr": "110", "rr": "22", "spo2": "94", "bp": "145/90", "temp": "38.2", "note": "Diabetes AI - Conf: 76.5% (Elevated Glucose)"},
    {"id": "30224466", "risk": "CRITICAL", "score": "8", "hr": "45", "rr": "10", "spo2": "85", "bp": "90/60", "temp": "35.5", "note": "NLP Text Triage - Conf: 88.0% (Respiratory Distress)"},
    {"id": "30998877", "risk": "HIGH", "score": "6", "hr": "95", "rr": "20", "spo2": "95", "bp": "130/85", "temp": "37.8", "note": "Heart AI - Conf: 65.0%"},
    {"id": "30776655", "risk": "HIGH", "score": "5", "hr": "88", "rr": "18", "spo2": "96", "bp": "135/88", "temp": "37.5", "note": "NLP Text Triage - Conf: 55.4% (Mild Angina)"},
    {"id": "30112233", "risk": "LOW", "score": "2", "hr": "72", "rr": "16", "spo2": "98", "bp": "118/78", "temp": "36.6", "note": "Heart AI - Conf: 15.2% (Routine)"},
    {"id": "30881122", "risk": "LOW", "score": "3", "hr": "78", "rr": "15", "spo2": "97", "bp": "122/80", "temp": "37.0", "note": "Heart AI - Conf: 25.1%"},
    {"id": "30554433", "risk": "LOW", "score": "1", "hr": "68", "rr": "14", "spo2": "99", "bp": "110/70", "temp": "36.8", "note": "Diabetes AI - Conf: 12.0%"}
]

# 2. Save directly to the CSV file your app uses
df = pd.DataFrame(dummy_patients)
df.to_csv("patients_db.csv", index=False)

print("✅ patients_db.csv successfully overwritten with 8 evaluation patients!")