import streamlit as st
from prophet.plot import plot_plotly
import joblib
import pandas as pd

st.title("Predicción de Demanda Turística")

# Lista de ciudades según modelos guardados
import os
modelos_disponibles = [f.replace(".pkl", "") for f in os.listdir("modelos_prophet")]

ciudad = st.selectbox("Selecciona una ciudad", modelos_disponibles)

# Cargar el modelo entrenado
modelo = joblib.load(f"modelos_prophet/{ciudad}.pkl")

# (Opcional) volver a cargar los datos de esa ciudad solo para graficar la serie real
df_demand = pd.read_csv("data/df_demand.csv")
df_ciudad = df_demand[df_demand['Name_x'] == ciudad].copy()
df_ciudad = df_ciudad.rename(columns={'VisitDate': 'ds', 'num_viajes': 'y'})
df_ciudad['ds'] = pd.to_datetime(df_ciudad['ds'])

# Predecir
future = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(future)


# Mostrar predicción interactiva
st.subheader("Predicción para los próximos 30 días")
fig = plot_plotly(modelo, forecast)
st.plotly_chart(fig)


# Mostrar serie real
st.subheader(f"Demanda histórica en {ciudad}")
st.line_chart(df_ciudad.set_index("ds")["y"])

