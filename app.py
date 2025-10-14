# ==============================
# Streamlit App – CKD Prediction (Youssef El Habraj)
# ==============================

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Système Intelligent de Détection Précoce des Maladies Rénales",
    page_icon="🧬",
    layout="wide"  # important
)

# --- CUSTOM PAGE WIDTH ---
# This expands the app layout to 100rem (≈1600px)
st.markdown(
    """
    <style>
        .main {
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- PAGE HEADER ---
st.title("🧬 Système Intelligent de Détection Précoce des Maladies Rénales (CKD)")
st.markdown("""
Cette application permet de **prédire automatiquement la présence d'une Maladie Rénale Chronique (MRC)** 
à partir des données biologiques d’un patient issues du **Système d’Information de Laboratoire (SIL)**.

**Modèle utilisé : Random Forest (Optimisé par GridSearchCV)**  
**Déployé avec : Streamlit + Docker**
""")

st.divider()

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_model():
    model = joblib.load('model/best_random_forest_model.pkl')
    return model

model = load_model()

# --- INPUT FIELDS ---
st.header("Données biologiques du patient")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Âge (ans)", 0, 100, 50)
    sexe = st.selectbox("Sexe", ["Homme", "Femme"])
    hemoglobine = st.number_input("Hémoglobine (g/dL)", 2.0, 22.0, 13.0)
    hematocrite = st.number_input("Hématocrite (%)", 10.0, 60.0, 40.0)
    hematies = st.number_input("Hématies (M/mm³)", 1.0, 8.0, 4.5)

with col2:
    leucocytes = st.number_input("Leucocytes (/mm³)", 1000, 20000, 7000)
    uree = st.number_input("Urée (g/L)", 0.05, 3.0, 0.4)
    creatinine = st.number_input("Créatinine (mg/L)", 2.0, 220.0, 10.0)
    potassium = st.number_input("Potassium (mmol/L)", 2.0, 8.0, 4.5)
    sodium = st.number_input("Sodium (mmol/L)", 120.0, 160.0, 140.0)

with col3:
    glycemie = st.number_input("Glycémie (g/L)", 0.4, 6.0, 1.0)
    diabetique = st.selectbox("Diabétique", ["Non", "Oui"])
    anemie = st.selectbox("Anémie", ["Non", "Oui"])

# --- PREPARE INPUT ---
sexe = 1 if sexe == "Homme" else 0
diabetique = 1 if diabetique == "Oui" else 0
anemie = 1 if anemie == "Oui" else 0

# DataFrame for prediction
input_data = pd.DataFrame({
    "age": [age],
    "sexe": [sexe],
    "hemoglobine": [hemoglobine],
    "hematocrite": [hematocrite],
    "hematies": [hematies],
    "leucocytes": [leucocytes],
    "uree": [uree],
    "creatinine": [creatinine],
    "potassium": [potassium],
    "sodium": [sodium],
    "glycemie": [glycemie],
    "diabetique": [diabetique],
    "anemie": [anemie]
})

st.divider()

# --- PREDICTION ---
if st.button("🔍 Prédire le statut rénal"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("🎯 Résultat de la prédiction :")
    if prediction == 1:
        st.error(f"⚠️ Le patient est probablement atteint de **Maladie Rénale Chronique (MRC)**.\n\nProbabilité : {proba:.2%}")
    else:
        st.success(f"✅ Le patient est probablement **Normal**.\n\nProbabilité d’atteinte : {proba:.2%}")

    st.caption("Modèle : Random Forest Classifier optimisé (AUC = 0.966, Accuracy = 0.92)")
