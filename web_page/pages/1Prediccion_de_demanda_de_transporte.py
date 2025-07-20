import streamlit as st
from prophet.plot import plot_plotly
import joblib
import pandas as pd
import plotly.express as px

st.title("Predicción de Demanda Turística")

# Lista de ciudades según modelos guardados
import os
modelos_disponibles = [f.replace(".pkl", "") for f in os.listdir("Recursos/modelos_prophet")]

ciudad = st.selectbox("Selecciona una ciudad", modelos_disponibles)

# Cargar el modelo entrenado
modelo = joblib.load(f"Recursos/modelos_prophet/{ciudad}.pkl")

# Volver a cargar los datos de esa ciudad solo para graficar la serie real
df_demand = pd.read_csv("Recursos/data/df_demand.csv")
df_ciudad = df_demand[df_demand['Name_x'] == ciudad].copy()
df_ciudad = df_ciudad.rename(columns={'VisitDate': 'ds', 'num_viajes': 'y'})
df_ciudad['ds'] = pd.to_datetime(df_ciudad['ds'])

# Predecir
future = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(future)


# Mostrar predicción interactiva
st.subheader("Predicción para los próximos 30 días")
fig = plot_plotly(modelo, forecast)

fig.update_xaxes(title_text="Fecha")
fig.update_yaxes(title_text="Número de viajes predichos")

st.plotly_chart(fig)


# Mostrar serie real

st.subheader(f"Demanda histórica en {ciudad}")

# Calcular los límites del eje Y con un 10% de margen adicional
y_min = df_ciudad["y"].min()
y_max = df_ciudad["y"].max()
margen = (y_max - y_min) * 0.1  # 10% de espacio adicional

fig_hist = px.line(
    df_ciudad,
    x="ds",
    y="y",
    labels={"ds": "Fecha", "y": "Número de viajes"},
    title=f"Demanda turística histórica en {ciudad}"
)

# Ajustar los márgenes verticales del eje Y
fig_hist.update_yaxes(range=[y_min - margen, y_max + margen])

st.plotly_chart(fig_hist)