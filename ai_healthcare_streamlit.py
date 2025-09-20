"""
AI Healthcare Chatbot - Streamlit LLM Upgrade
Single-file Streamlit app to run a lightweight AI/ML-powered symptom-checker chatbot.

Updates:
- Uses LLM (flan-t5-small) for symptom -> condition prediction
- Offline CPU-friendly
- Keeps existing triage, risk scoring, and doctor dashboard
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

# LLM imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Optional TTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ---------------------------
# Symptom keywords / self-care
# ---------------------------
SYMPTOM_KEYWORDS = {
    'fever': ['fever', 'temperature', 'hot', 'chills'],
    'cough': ['cough', 'coughing'],
    'sore_throat': ['sore throat', 'throat pain', 'throat'],
    'runny_nose': ['runny nose', 'nasal', 'congestion'],
    'sneezing': ['sneezing', 'sneeze'],
    'headache': ['headache', 'head pain', 'migraine'],
    'vomiting': ['vomit', 'vomiting', 'throw up', 'nausea'],
    'diarrhea': ['diarrhea', 'watery stool', 'loose stool'],
    'chest_pain': ['chest pain', 'tightness', 'pressure in chest'],
    'shortness_breath': ['shortness of breath', 'breathless', 'dyspnea', 'difficulty breathing'],
    'fatigue': ['tired', 'fatigue', 'exhausted'],
    'loss_smell': ['loss of smell', 'no smell', 'anosmia'],
    'urinary': ['painful urination', 'burning pee', 'urinary'],
}

SEVERITY_KEYWORDS = {
    'emergency': ['unconscious', 'severe', 'very bad', 'collapse', 'faint', 'bleeding', 'chest pain', 'shortness of breath'],
    'moderate': ['high fever', 'fever', 'severe pain', 'persistent', 'worsening'],
    'mild': ['mild', 'slight', 'runny', 'little']
}

SELF_CARE_TIPS = {
    'Common Cold': "Rest, fluids, steam inhalation, paracetamol if feverish. If symptoms worsen > 3 days, consult a doctor.",
    'Flu': "Rest, lots of fluids, paracetamol for fever, isolate if possible. Seek medical care for breathing difficulty or persistent high fever.",
    'Allergy': "Avoid allergens, antihistamines can help. If breathing difficulties occur, seek emergency care.",
    'Gastroenteritis': "Oral rehydration (ORS), small frequent sips, avoid oily/spicy food. If severe dehydration or bloody stools, see a doctor.",
    'Migraine': "Rest in a dark quiet room, prescribed migraine meds if you have them. If first-time severe headache, consult a clinician.",
    'Cardiac Issue': "Chest pain or shortness of breath can be serious — seek emergency care immediately.",
    'Viral Infection': "Rest, fluids, symptomatic care. If breathing difficulty or confusion, seek urgent care.",
    'Urinary Tract Infection': "Drink plenty of fluids, see a doctor for antibiotics if symptoms are persistent or severe.",
}

# ---------------------------
# Load LLM
# ---------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

TOKENIZER, MODEL_LLM = load_llm()

# ---------------------------
# LLM prediction function
# ---------------------------
def llm_predict(text, max_new_tokens=50):
    if not text.strip():
        return []
    input_text = f"Given these symptoms: {text}. Predict possible conditions:"
    inputs = TOKENIZER(input_text, return_tensors="pt")
    outputs = MODEL_LLM.generate(**inputs, max_new_tokens=max_new_tokens)
    preds = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    # Split multiple conditions by semicolon
    return [x.strip() for x in preds.split(';') if x.strip()]

# ---------------------------
# Utilities
# ---------------------------
def extract_symptoms_from_text(text):
    text_low = text.lower()
    found = set()
    for key, variants in SYMPTOM_KEYWORDS.items():
        for v in variants:
            if v in text_low:
                found.add(key)
    return sorted(list(found))

def compute_risk_score(text, top_prob=0.8):  # top_prob default as heuristic
    base = int(top_prob * 100)
    text_low = text.lower()
    for kw in SEVERITY_KEYWORDS['emergency']:
        if kw in text_low:
            base += 40
    for kw in SEVERITY_KEYWORDS['moderate']:
        if kw in text_low:
            base += 15
    for kw in SEVERITY_KEYWORDS['mild']:
        if kw in text_low:
            base -= 10
    return max(0, min(100, base))

def triage_recommendation(score):
    if score >= 75:
        return 'EMERGENCY - Go to nearest emergency department now'
    elif score >= 40:
        return 'Consult a doctor soon (within 24-48 hours)'
    else:
        return 'Self-care at home; monitor symptoms. See doctor if worsens.'

def save_query(record, filename='queries.csv'):
    df_new = pd.DataFrame([record])
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(filename, index=False)

def speak_text(text):
    if pyttsx3 is None:
        return False
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title='AI Healthcare Chatbot (LLM)', layout='wide')
st.title('AI-assisted Symptom Checker — LLM Upgrade')
st.markdown("""
LLM-based chatbot to predict conditions from symptoms. Offline CPU-compatible.
Triage and self-care tips remain heuristic-based.
""")

with st.sidebar:
    st.header('Settings')
    language = st.selectbox('Language', ['English'])
    DOC_DEFAULT = 'doctor123'
    st.write('Doctor dashboard password is hidden for security (demo).')
    doc_password = st.text_input('Set doctor password (demo) — leave blank to use default', value='', type='password')
    if not doc_password:
        doc_password = DOC_DEFAULT
    st.markdown('---')
    st.write('Quick tips:')
    st.write('- Keep inputs concise for best results.')

col1, col2 = st.columns([2,1])

with col1:
    st.subheader('Chat / Symptom Input')
    user_input = st.text_area(
        'Describe symptoms', 
        placeholder='Write your symptoms here... (e.g., fever, cough, headache)', 
        height=140
    )

    if st.button('Analyze'):
        if not user_input.strip():
            st.warning('Please enter symptoms.')
        else:
            with st.spinner('Analyzing...'):
                preds = llm_predict(user_input)
                top_condition = preds[0] if preds else "Unknown"
                other_conditions = preds[1:] if len(preds) > 1 else []
                symptoms_found = extract_symptoms_from_text(user_input)
                score = compute_risk_score(user_input)
                triage = triage_recommendation(score)

            st.markdown('### Prediction')
            st.write(f'**Likely condition:** {top_condition}')
            if other_conditions:
                st.write('Other possible conditions:')
                for c in other_conditions:
                    st.write(f'- {c}')

            st.markdown('### Symptoms detected')
            st.write(', '.join(symptoms_found) if symptoms_found else 'No keywords detected.')

            st.markdown('### Risk & Triage')
            st.write(f'Risk score: **{score}/100**')
            st.write(f'Triage recommendation: **{triage}**')

            st.markdown('### Suggested next steps')
            tip = SELF_CARE_TIPS.get(top_condition, 'Monitor symptoms and consult a clinician if you are worried.')
            st.write(tip)

            record = {
                'timestamp': datetime.now().isoformat(),
                'input_text': user_input,
                'predicted': top_condition,
                'confidence': 0.8,  # heuristic for CPU model
                'risk_score': int(score),
                'triage': triage
            }
            save_query(record)
            st.success('Saved session to local queries.csv')

            if pyttsx3 is not None:
                if st.button('Play spoken reply'):
                    spoken = f'Likely condition {top_condition}. Recommendation: {triage}.'
                    ok = speak_text(spoken)
                    if not ok:
                        st.warning('TTS not available on this machine.')

            report_text = f"Timestamp: {record['timestamp']}\nInput: {record['input_text']}\nPrediction: {record['predicted']}\nRisk score: {record['risk_score']}\nTriage: {record['triage']}\nAdvice: {tip}\n"
            st.download_button('Download report (txt)', report_text, file_name='report.txt')

with col2:
    st.subheader('Quick help & demo prompts')
    st.write('Sample inputs:')
    st.write('- "fever cough body ache"')
    st.write('- "runny nose sneezing itchy eyes"')
    st.write('- "abdominal pain vomiting"')
    st.write('- "chest pain and shortness of breath"')
    st.markdown('---')
    st.subheader('Notes for judges/demo')
    st.write('- LLM-based predictions; triage and doctor dashboard are still heuristic.')

# ---------------------------
# Doctor dashboard (demo)
st.markdown('---')
st.header('Doctor Dashboard (demo)')
st.info('Password-protected. Ask admin for credentials.')
entered = st.text_input('Doctor login — enter password', type='password')

if entered:
    if entered == DOC_DEFAULT:
        st.success('Authenticated')
        if os.path.exists('queries.csv'):
            df = pd.read_csv('queries.csv')
            st.write('Saved sessions')
            st.dataframe(df.sort_values('timestamp', ascending=False).reset_index(drop=True))

            st.markdown('### Analytics')
            counts = df['predicted'].value_counts().reset_index()
            counts.columns = ['condition', 'count']
            st.write(counts)

            st.markdown('### Filter & export')
            sel_cond = st.selectbox('Filter by condition', options=['All'] + list(df['predicted'].unique()))
            if sel_cond != 'All':
                st.dataframe(df[df['predicted'] == sel_cond])
            if st.button('Export saved sessions (CSV)'):
                st.download_button('Download CSV', df.to_csv(index=False), file_name='queries_export.csv')
        else:
            st.info('No sessions saved yet. Interact with the chatbot and press Analyze.')
    else:
        st.error('Wrong password.')
else:
    st.info('Enter password to unlock doctor dashboard.')
