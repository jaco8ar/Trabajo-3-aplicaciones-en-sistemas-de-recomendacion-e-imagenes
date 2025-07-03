import streamlit as st

st.markdown("""
    <style>
        .stApp {
            background-color: #040317;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Welcome", layout="centered")

st.title("Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en la Empresa de Transporte")
st.write("Seleccione uno de nuestros servicios:")

st.page_link("pages/1Prediccion_de_demanda_de_transporte.py", label="**Predicción de Demanda de Transporte**", icon = "🚚")

st.page_link("pages/2Clasificación_de_conducción_distractiva.py", label="**Clasificación de Conducción Distractiva**", icon = "⚠️")
st.page_link("pages/3Recomendación_de_destinos.py", label="**Clasificación de Conducción Distractiva**", icon = "💯")
# st.page_link("pages/text_classifier.py", label="📝 Text Classifier")
st.page_link("pages/Sobre_Nosotros.py", label="Sobre Nosotros", icon = "❔")
