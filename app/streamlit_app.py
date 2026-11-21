import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="AgriPredict", page_icon="", layout="centered")

st.title(" AgriPredict")
st.caption("Prédiction du rendement agricole au Cameroun à partir du climat et des intrants.")

try:
    meta = requests.get(f"{API}/meta", timeout=5).json()
except requests.exceptions.RequestException:
    st.error("L'API ne répond pas. Lance-la avec : uvicorn api.main:app --reload")
    st.stop()

with st.form("prediction_form"):
    culture = st.selectbox("Culture", meta["crops"])

    col1, col2 = st.columns(2)
    with col1:
        annee = st.number_input("Année", min_value=2000, max_value=2060, value=2025, step=1)
        temperature = st.number_input("Anomalie de température (°C)", value=1.0, step=0.1, format="%.1f")
        pluie = st.number_input("Pluviométrie annuelle (mm)", value=1600.0, step=10.0, format="%.0f")
    with col2:
        engrais = st.number_input("Engrais (tonnes)", value=50000.0, step=1000.0, format="%.0f")
        pesticides = st.number_input("Pesticides (tonnes)", value=3000.0, step=100.0, format="%.0f")

    valider = st.form_submit_button("Prédire", type="primary", use_container_width=True)

if valider:
    try:
        reponse = requests.post(
            f"{API}/predict",
            json={
                "crop": culture,
                "year": int(annee),
                "temp_anomaly": temperature,
                "rainfall": pluie,
                "fertilizer": engrais,
                "pesticides": pesticides,
            },
            timeout=5,
        )
        reponse.raise_for_status()
        rendement = reponse.json()["yield_t_ha"]
        st.metric(f"Rendement prédit — {culture}", f"{rendement} t/ha")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la prédiction : {e}")

with st.expander("À propos du modèle"):
    st.write(f"**Modèle utilisé :** {meta['best_model']}")
    st.write(f"**Période d'entraînement :** {meta['year_min']}–{meta['year_max']}")
    st.write("**Précision par modèle testé (RMSE, en t/ha — plus bas = meilleur) :**")
    st.dataframe(meta["metrics"], use_container_width=True)
