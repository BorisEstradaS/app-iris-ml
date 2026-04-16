import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2

# ======================================================
# CONFIGURACIÓN STREAMLIT
# ======================================================
st.set_page_config(page_title="Predictor Iris", page_icon="🌸")

# ======================================================
# DATABASE CONFIG
# (RECOMENDADO: mover luego a st.secrets)
# ======================================================
USER = "postgres.amceyyieuxlyqdcdsihx"
PASSWORD = "Google72025601:3"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# ======================================================
# CONEXIÓN DATABASE
# ======================================================
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

# ======================================================
# GUARDAR PREDICCIÓN
# ======================================================
def save_prediction(l_s, a_s, l_p, a_p, prediccion):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO ml.tb_iris
        (l_p, l_s, a_s, a_p, prediccion)
        VALUES (%s,%s,%s,%s,%s);
    """

    cursor.execute(
        query,
        (l_p, l_s, a_s, a_p, prediccion)
    )

    conn.commit()
    cursor.close()

# ======================================================
# CARGAR HISTORIAL
# ======================================================
def load_history():

    conn = get_connection()

    query = """
        SELECT
            id,
            created_at,
            l_s AS sepal_length,
            a_s AS sepal_width,
            l_p AS petal_length,
            a_p AS petal_width,
            prediccion
        FROM ml.tb_iris
        ORDER BY created_at DESC
        LIMIT 20;
    """

    df = pd.read_sql(query, conn)
    return df

# ======================================================
# CARGAR MODELO ML
# ======================================================
@st.cache_resource
def load_models():

    model = joblib.load("components/iris_model.pkl")
    scaler = joblib.load("components/iris_scaler.pkl")

    with open("components/model_info.pkl", "rb") as f:
        model_info = pickle.load(f)

    return model, scaler, model_info


model, scaler, model_info = load_models()

# ======================================================
# INTERFAZ
# ======================================================
st.title("🌸 Predictor de Especies de Iris")

st.header("Ingresa las características de la flor")

sepal_length = st.number_input(
    "Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1
)

sepal_width = st.number_input(
    "Ancho del Sépalo (cm)", 0.0, 10.0, 3.0, 0.1
)

petal_length = st.number_input(
    "Longitud del Pétalo (cm)", 0.0, 10.0, 4.0, 0.1
)

petal_width = st.number_input(
    "Ancho del Pétalo (cm)", 0.0, 10.0, 1.0, 0.1
)

# ======================================================
# BOTÓN PREDICCIÓN
# ======================================================
if st.button("Predecir Especie"):

    features = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    predicted_species = model_info["target_names"][prediction]
    confidence = float(max(probabilities))

    # RESULTADO
    st.success(f"🌼 Especie predicha: **{predicted_species}**")
    st.write(f"Confianza: **{confidence:.1%}**")

    st.write("Probabilidades:")
    for species, prob in zip(model_info["target_names"], probabilities):
        st.write(f"- {species}: {prob:.1%}")

    # GUARDAR EN SUPABASE
    save_prediction(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        predicted_species
    )

    st.toast("✅ Predicción guardada")

    # REFRESCAR APP
    st.rerun()

# ======================================================
# HISTORIAL
# ======================================================
st.divider()
st.subheader("📊 Historial de Predicciones")

history = load_history()

if not history.empty:
    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Aún no existen predicciones.")
