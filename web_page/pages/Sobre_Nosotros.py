import streamlit as st

st.set_page_config(page_title="Sobre Nosotros", page_icon="💼")

st.title("👥 Sobre Nosotros")

# Sección de enlace al reporte
st.header("🌐 Enlace al Reporte de Resultados en Quarto")
st.markdown("🔗 [Ver el informe completo](https://jochoara.quarto.pub/trabajo-3)", unsafe_allow_html=True)


st.divider()

# Sección de repartición de tareas
st.header("🛠️ Repartición de Tareas")

st.subheader("👤 Jacobo Ochoa Ramírez")
st.markdown("""
- Construcción de modelo de identificación de comportamientos peligrosos 
- Diseño y construcción de la interfaz web con Streamlit.
- Elaboración del informe técnico en Quarto.
""")

st.subheader("👤 Juan José Correa Hurtado")
st.markdown("""
- Entrenamiento del modelo de predicción de demanda.
- Desarrollo del sistema de recomendación híbrido.
- Elaboración del informe técnico en Quarto.
""")
