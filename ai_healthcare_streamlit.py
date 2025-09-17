"""
AI Healthcare Chatbot - Streamlit Starter
Single-file Streamlit app to run a lightweight AI/ML-powered symptom-checker chatbot.

Features included (starter):
- Simple training dataset embedded (small) and a MultinomialNB text classifier
- Keyword-based symptom extraction (lightweight NER)
- Heuristic risk scoring and triage (Self-care / See doctor / Emergency)
- Text input only (audio upload removed for simplicity)
- Offline TTS reply via pyttsx3 (if available)
- Doctor dashboard (password protected) that shows saved queries and simple analytics
- Save queries locally to `queries.csv`
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

# Machine Learning imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Optional features (import lazily)
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ---------------------------
# Embedded training data (small toy dataset)
# ---------------------------
TRAIN_DATA = [
    ("fever cough body ache sore throat", "Flu"),
    ("high fever chills cough and body pain", "Flu"),
    ("runny nose sneezing itchy eyes", "Allergy"),
    ("sneezing nasal congestion itchy throat", "Allergy"),
    ("abdominal pain vomiting diarrhea", "Gastroenteritis"),
    ("vomiting stomach cramps watery stools", "Gastroenteritis"),
    ("headache sensitivity to light nausea", "Migraine"),
    ("throbbing headache with nausea", "Migraine"),
    ("chest pain shortness of breath sweating", "Cardiac Issue"),
    ("severe chest tightness and short breath", "Cardiac Issue"),
    ("mild cough sore throat no fever", "Common Cold"),
    ("sore throat sneezing mild cough", "Common Cold"),
    ("fever dry cough loss of smell taste", "Viral Infection"),
    ("fever body ache tiredness", "Viral Infection"),
    ("painful urination lower abdominal pain", "Urinary Tract Infection"),
    ("burning pee frequency urgency", "Urinary Tract Infection"),
]

# Expand dataset slightly by simple paraphrases
EXTRA = []
for text, label in TRAIN_DATA:
    EXTRA.append((text + " " + label.lower(), label))
TRAIN_DATA.extend(EXTRA)

# ---------------------------
# Light NER / keyword lists
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
# Model training (small, in-memory)
# ---------------------------
def build_and_train_model(train_data):
    X = [t for t, l in train_data]
    y = [l for t, l in train_data]
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,2), max_features=2000)),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(X, y)
    return pipeline

MODEL = build_and_train_model(TRAIN_DATA)

# Quick cross-validation score on tiny dataset (for display only)
try:
    cv_scores = cross_val_score(MODEL, [t for t,l in TRAIN_DATA], [l for t,l in TRAIN_DATA], cv=3)
    CV_SCORE = float(cv_scores.mean())
except Exception:
    CV_SCORE = None

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

def compute_risk_score(text, top_prob):
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

def predict_conditions(text, top_n=3):
    if not text or text.strip() == "":
        return []
    probs = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    paired = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return paired[:top_n]

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
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title='AI Healthcare Chatbot (Starter)', layout='wide')
st.title('AI-assisted Symptom Checker — Starter')
st.markdown("""
A lightweight prototype to demo: symptom extraction, small ML-based condition prediction, risk scoring and triage.
This is a starter template for hackathon/demo use. Expand the dataset and replace heuristics with better models for production.
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
    st.write('Model info:')
    if CV_SCORE:
        st.write(f'Cross-val score (toy data): {CV_SCORE:.2f}')
    st.write('Model type: Naive Bayes (text)')
    st.markdown('---')
    st.write('Quick tips:')
    st.write('- Keep inputs short: list of symptoms works best.')

# Two columns
col1, col2 = st.columns([2,1])

with col1:
    st.subheader('Chat / Symptom Input')
    user_input = st.text_area(
    'Describe symptoms', 
    placeholder='Write your symptoms here... (e.g., fever, cough, headache)', 
    height=140
)


    if st.button('Analyze'):
        if not user_input or user_input.strip() == '':
            st.warning('Please enter symptoms.')
        else:
            with st.spinner('Analyzing...'):
                preds = predict_conditions(user_input, top_n=3)
                top_condition, top_prob = preds[0]
                symptoms_found = extract_symptoms_from_text(user_input)
                score = compute_risk_score(user_input, top_prob)
                triage = triage_recommendation(score)

            st.markdown('### Prediction')
            st.write(f'**Likely condition:** {top_condition} ({top_prob*100:.1f}% confidence)')
            if len(preds) > 1:
                st.write('Other possible conditions:')
                for c, p in preds[1:]:
                    st.write(f'- {c} ({p*100:.1f}%)')

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
                'confidence': float(top_prob),
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

            report_text = f"Timestamp: {record['timestamp']}\nInput: {record['input_text']}\nPrediction: {record['predicted']} ({record['confidence']*100:.1f}%)\nRisk score: {record['risk_score']}\nTriage: {record['triage']}\nAdvice: {tip}\n"
            st.download_button('Download report (txt)', report_text, file_name='report.txt')

with col2:
    st.subheader('Quick help & demo prompts')
    st.write('Sample inputs to try:')
    st.write('- "fever cough body ache"')
    st.write('- "runny nose sneezing itchy eyes"')
    st.write('- "abdominal pain vomiting"')
    st.write('- "chest pain and shortness of breath"')
    st.markdown('---')
    st.subheader('Notes for judges/demo')
    st.write('- Lightweight prototype — emphasize the triage and doctor dashboard during demo.')

# ---------------------------
# Doctor dashboard (secured demo)
st.markdown('---')
st.header('Doctor Dashboard (demo)')
st.info('Dashboard is password-protected. Ask admin for the credentials.')
DOC_DEFAULT = 'doctor123'
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
            st.info('No sessions saved yet. Interact with the chatbot and press Analyze to create sample records.')
    else:
        st.error('Wrong password.')
else:
    st.info('Enter password to unlock doctor dashboard.')

