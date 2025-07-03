import streamlit as st

st.set_page_config(page_title="Welcome", layout="centered")

st.title("AI Services Hub")
st.write("Select a service from the options below:")

st.page_link("pages/image_recognition.py", label="Clasificación de Conducción Distractiva")
st.page_link("pages/prediccion_demanda.py", label="Predicción de Demanda de Transporte")

# st.page_link("pages/text_classifier.py", label="📝 Text Classifier")
st.page_link("pages/about.py", label="About This App")
