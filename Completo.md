## Punto 1
 ### 1. Código del modelo con preprocesamiento de datos (por ejemplo, limpieza de datos, transformación de series de tiempo). 
 Primero importamos la librerias utilizadas
 ```python
import os
import json
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
```
Luego descargamos los csv al colab y los leemos con la librería pandas
```python
#ES NECESARIO SUBIR LOS CSV AL COLAB
path = kagglehub.dataset_download("amanmehra23/travel-recommendation-dataset")

print("Path to dataset files:", path)

destinations_df = pd.read_csv("/kaggle/input/travel-recommendation-dataset/Expanded_Destinations.csv")
reviews_df = pd.read_csv("/kaggle/input/travel-recommendation-dataset/Final_Updated_Expanded_Reviews.csv")
userhistory_df = pd.read_csv("/kaggle/input/travel-recommendation-dataset/Final_Updated_Expanded_UserHistory.csv")
users_df = pd.read_csv("/kaggle/input/travel-recommendation-dataset/Final_Updated_Expanded_Users.csv")
```
```python
display(userhistory_df.head())
userhistory_df.info()
```


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 999 entries, 0 to 998
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   HistoryID         999 non-null    int64 
     1   UserID            999 non-null    int64 
     2   DestinationID     999 non-null    int64 
     3   VisitDate         999 non-null    object
     4   ExperienceRating  999 non-null    int64 
    dtypes: int64(4), object(1)
    memory usage: 39.2+ KB



```python
# Verificar valores únicos por columna
print("\nValores únicos por columna:")
print(userhistory_df.nunique())

# Verificar valores nulos
print("\nValores nulos por columna:")
print(userhistory_df.isnull().sum())
```

    
    Valores únicos por columna:
    HistoryID           999
    UserID              642
    DestinationID       638
    VisitDate             3
    ExperienceRating      5
    dtype: int64
    
    Valores nulos por columna:
    HistoryID           0
    UserID              0
    DestinationID       0
    VisitDate           0
    ExperienceRating    0
    dtype: int64



```python
# Combinar datasets

reviews_destinations = pd.merge(reviews_df, destinations_df, on='DestinationID', how='inner')

reviews_destinations_userhistory = pd.merge(reviews_destinations, userhistory_df, on='UserID', how='inner')

df = pd.merge(reviews_destinations_userhistory, users_df, on='UserID', how='inner')

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 993 entries, 0 to 992
    Data columns (total 20 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   ReviewID          993 non-null    int64  
     1   DestinationID_x   993 non-null    int64  
     2   UserID            993 non-null    int64  
     3   Rating            993 non-null    int64  
     4   ReviewText        993 non-null    object 
     5   Name_x            993 non-null    object 
     6   State             993 non-null    object 
     7   Type              993 non-null    object 
     8   Popularity        993 non-null    float64
     9   BestTimeToVisit   993 non-null    object 
     10  HistoryID         993 non-null    int64  
     11  DestinationID_y   993 non-null    int64  
     12  VisitDate         993 non-null    object 
     13  ExperienceRating  993 non-null    int64  
     14  Name_y            993 non-null    object 
     15  Email             993 non-null    object 
     16  Preferences       993 non-null    object 
     17  Gender            993 non-null    object 
     18  NumberOfAdults    993 non-null    int64  
     19  NumberOfChildren  993 non-null    int64  
    dtypes: float64(1), int64(9), object(10)
    memory usage: 155.3+ KB



```python
#Revisión de datos nulos o duplicados
df.shape
df.duplicated().sum()
df.isnull().sum()
```
Ahora que tenemos los csv combinados en un solo dataset podemos hacer una exploración inicial de los datos.

```python
plt.figure(figsize=(10, 6))
sns.barplot(y='Name', x='Popularity', data=destinations_df.sort_values(by='Popularity', ascending=True), palette='coolwarm', hue='Name')
plt.title('Most Popular Destinations')
plt.xlabel('Popularity Score')
plt.ylabel('Destination')
plt.show()
```
De esta gráfica inicial podemos ver cuales destinos son los más populares, siendo los que tengan colores más cálidos los más populares. Ahora veamos información sobre las columnas del dataset.
```python
#Ver cuantos datos hay por cada usuario
userhistory_df.groupby('UserID').size().sort_values(ascending=False)
```

```python
#Convertir la fecha a datetime name_x => ciudad name_y => nombre del usuario
df_copia = df.copy()
df_copia['VisitDate'] = pd.to_datetime(df_copia['VisitDate'], errors='coerce')
df_copia.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 993 entries, 0 to 992
    Data columns (total 20 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   ReviewID          993 non-null    int64         
     1   DestinationID_x   993 non-null    int64         
     2   UserID            993 non-null    int64         
     3   Rating            993 non-null    int64         
     4   ReviewText        993 non-null    object        
     5   Name_x            993 non-null    object        
     6   State             993 non-null    object        
     7   Type              993 non-null    object        
     8   Popularity        993 non-null    float64       
     9   BestTimeToVisit   993 non-null    object        
     10  HistoryID         993 non-null    int64         
     11  DestinationID_y   993 non-null    int64         
     12  VisitDate         993 non-null    datetime64[ns]
     13  ExperienceRating  993 non-null    int64         
     14  Name_y            993 non-null    object        
     15  Email             993 non-null    object        
     16  Preferences       993 non-null    object        
     17  Gender            993 non-null    object        
     18  NumberOfAdults    993 non-null    int64         
     19  NumberOfChildren  993 non-null    int64         
    dtypes: datetime64[ns](1), float64(1), int64(9), object(9)
    memory usage: 155.3+ KB
Ahora, para ver cual es el destino más visitado se crea la serie de tiempo agregada a partir del dataset.

```python
#Crear la serie de tiempo agregada: Agrupar por día y destino para contar número de viajes
df_demand = df_copia.groupby(['VisitDate', 'Name_x']).size().reset_index(name='num_viajes')

# Ordenar por fecha
df_demand = df_demand.sort_values('VisitDate')

# Ver los destinos más populares
top_destinos = df_demand.groupby('Name_x')['num_viajes'].sum().sort_values(ascending=False)
print("Destinos más visitados:")
print(top_destinos.head(10))

```

    Destinos más visitados:
    Name_x
    Leh Ladakh           222
    Taj Mahal            206
    Jaipur City          201
    Kerala Backwaters    192
    Goa Beaches          172
    Name: num_viajes, dtype: int64
Pasamos a hacer la serie de tiempo del destino más visitado para hacer una primera modelación.
```python

# Seleccionar la ciudad más visitada
ciudad_top = df_demand.groupby('Name_x')['num_viajes'].sum().sort_values(ascending=False).index[0]

# Filtrar solo esa ciudad
df_ciudad = df_demand[df_demand['Name_x'] == ciudad_top].copy()

# Renombrar columnas como lo requiere Prophet
df_ciudad = df_ciudad.rename(columns={'VisitDate': 'ds', 'num_viajes': 'y'})

# Asegurarnos de que estén ordenadas por fecha
df_ciudad = df_ciudad.sort_values('ds')

# Mostrar datos
print(f"Serie de tiempo para la ciudad: {ciudad_top}")
display(df_ciudad)

# Crear el modelo
modelo = Prophet(daily_seasonality=True)

# Entrenar el modelo
modelo.fit(df_ciudad)

# Crear un dataframe para predicción de los próximos 30 días
future = modelo.make_future_dataframe(periods=30)

# Predecir
forecast = modelo.predict(future)

# Graficar los resultados
modelo.plot(forecast)
plt.title(f"Predicción de demanda para {ciudad_top}")
plt.xlabel("Fecha")
plt.ylabel("Número de viajes")
plt.show()
```
Ahora veamos las predicciones para todas las ciudades
```python
# Unir todos los forecast en un solo DataFrame
df_predicciones_total = pd.concat(predicciones_por_ciudad.values())

# Filtrar solo las fechas futuras (los 30 días siguientes)
ultima_fecha_real = df_demand['VisitDate'].max()
df_futuro = df_predicciones_total[df_predicciones_total['ds'] > ultima_fecha_real]

# Gráfico combinado
plt.figure(figsize=(12, 6))
for ciudad in top_5_ciudades:
    datos = df_futuro[df_futuro['ciudad'] == ciudad]
    plt.plot(datos['ds'], datos['yhat'], label=ciudad)

plt.title("Proyección de demanda para los próximos 30 días")
plt.xlabel("Fecha")
plt.ylabel("Número de viajes")
plt.legend()
plt.grid(True)
plt.show()

```
### 2. Métricas de evaluación
```python
#Métricas de evaluación (RMSE, MAE)
print("Métricas de evaluación:\n")

for ciudad in top_5_ciudades:
    df_real = df_demand[df_demand['Name_x'] == ciudad][['VisitDate', 'num_viajes']].copy()
    df_real = df_real.rename(columns={'VisitDate': 'ds', 'num_viajes': 'y'}).sort_values('ds')

    forecast = predicciones_por_ciudad[ciudad]
    forecast_real = forecast.merge(df_real, on='ds', how='inner')  # Solo fechas reales

    rmse = np.sqrt(mean_squared_error(forecast_real['y'], forecast_real['yhat']))
    mae = mean_absolute_error(forecast_real['y'], forecast_real['yhat'])

    print(f"{ciudad} - RMSE: {rmse:.2f} | MAE: {mae:.2f}")
```

    Métricas de evaluación:
    
    Leh Ladakh - RMSE: 2.95 | MAE: 2.77
    Taj Mahal - RMSE: 3.68 | MAE: 3.46
    Jaipur City - RMSE: 2.56 | MAE: 2.40
    Kerala Backwaters - RMSE: 3.90 | MAE: 3.67
    Goa Beaches - RMSE: 2.22 | MAE: 2.09
### 3. -   Gráficas de predicción vs. demanda real para cada ruta de transporte.
```python
# Almacenar modelos y predicciones
predicciones_por_ciudad = {}
modelos_por_ciudad = {}

for ciudad in top_5_ciudades:
    df_ciudad = df_demand[df_demand['Name_x'] == ciudad][['VisitDate', 'num_viajes']].copy()
    df_ciudad = df_ciudad.rename(columns={'VisitDate': 'ds', 'num_viajes': 'y'}).sort_values('ds')

    modelo = Prophet(daily_seasonality=True)
    modelo.fit(df_ciudad)

    future = modelo.make_future_dataframe(periods=30)
    forecast = modelo.predict(future)

    predicciones_por_ciudad[ciudad] = forecast
    modelos_por_ciudad[ciudad] = modelo

# Mostrar componentes para cada ciudad
for ciudad in top_5_ciudades:
    print(f"\nComponentes del modelo para {ciudad}:")
    modelos_por_ciudad[ciudad].plot_components(predicciones_por_ciudad[ciudad])
    plt.show()
```
### 4. Análisis de la estacionalidad y tendencias de la demanda.
```python
# Almacenar modelos y predicciones
predicciones_por_ciudad = {}
modelos_por_ciudad = {}

for ciudad in top_5_ciudades:
    df_ciudad = df_demand[df_demand['Name_x'] == ciudad][['VisitDate', 'num_viajes']].copy()
    df_ciudad = df_ciudad.rename(columns={'VisitDate': 'ds', 'num_viajes': 'y'}).sort_values('ds')

    modelo = Prophet(daily_seasonality=True)
    modelo.fit(df_ciudad)

    future = modelo.make_future_dataframe(periods=30)
    forecast = modelo.predict(future)

    predicciones_por_ciudad[ciudad] = forecast
    modelos_por_ciudad[ciudad] = modelo

# Mostrar componentes para cada ciudad
for ciudad in top_5_ciudades:
    print(f"\nComponentes del modelo para {ciudad}:")
    modelos_por_ciudad[ciudad].plot_components(predicciones_por_ciudad[ciudad])
    plt.show()
```
Aunque el dataset tiene un número limitado de fechas, se observan algunos indicios de variaciones diarias que podrían interpretarse como patrones estacionales si tuviéramos más datos. Prophet permite visualizar las componentes de tendencia, estacionalidad y efectos semanales o anuales. En este caso, no se observan componentes significativos por la escasez de datos, pero el modelo está correctamente estructurado para captarlas si se amplía la serie temporal.

## Punto 3
#Importar librerías necesarias
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
```
### Carga de datos
```python
#ES NECESARIO SUBIR LOS CSV AL COLAB
destinations_df = pd.read_csv("Expanded_Destinations.csv")
reviews_df = pd.read_csv("Final_Updated_Expanded_Reviews.csv")
userhistory_df = pd.read_csv("Final_Updated_Expanded_UserHistory.csv")
users_df = pd.read_csv("Final_Updated_Expanded_Users.csv")
```
```python
# Tipado uniforme
for df in [users_df, userhistory_df, reviews_df]:
    df['UserID'] = df['UserID'].astype(str)
for df in [destinations_df, userhistory_df, reviews_df]:
    df['DestinationID'] = df['DestinationID'].astype(str)

# Combinar datasets

merged_df = pd.merge(userhistory_df, users_df, on='UserID', how='left')

merged_df = pd.merge(merged_df, destinations_df, on='DestinationID', how='left')

df = merged_df

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 999 entries, 0 to 998
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   HistoryID         999 non-null    int64  
     1   UserID            999 non-null    object 
     2   DestinationID     999 non-null    object 
     3   VisitDate         999 non-null    object 
     4   ExperienceRating  999 non-null    int64  
     5   Name_x            999 non-null    object 
     6   Email             999 non-null    object 
     7   Preferences       999 non-null    object 
     8   Gender            999 non-null    object 
     9   NumberOfAdults    999 non-null    int64  
     10  NumberOfChildren  999 non-null    int64  
     11  Name_y            999 non-null    object 
     12  State             999 non-null    object 
     13  Type              999 non-null    object 
     14  Popularity        999 non-null    float64
     15  BestTimeToVisit   999 non-null    object 
    dtypes: float64(1), int64(4), object(11)
    memory usage: 125.0+ KB

```python
#Convertir la fecha a datetime 
df['VisitDate'] = pd.to_datetime(df['VisitDate'])
#Eliminar columnas irrelevantes
df_copia = df.drop(columns=['HistoryID','Name_x','Email'])
#Normalizando columnas categóricas
df_copia['PreferencesList'] = df_copia['Preferences'].str.split(',')
df_copia['Type'] = df_copia['Type'].str.lower()
df_copia['BestTimeToVisit'] = df_copia['BestTimeToVisit'].str.lower()
df_copia['PopularityNormalizado'] = (df_copia['Popularity'] - df_copia['Popularity'].min()) / (df_copia['Popularity'].max() - df_copia['Popularity'].min())
df_copia['ExperienceRatingNormalizado'] = (df_copia['ExperienceRating'] - df_copia['ExperienceRating'].min()) / (df_copia['ExperienceRating'].max() - df_copia['ExperienceRating'].min())
df_copia.head()
```





  <div id="df-f5309b41-e607-448b-8b18-e18f507b4227" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UserID</th>
      <th>DestinationID</th>
      <th>VisitDate</th>
      <th>ExperienceRating</th>
      <th>Preferences</th>
      <th>Gender</th>
      <th>NumberOfAdults</th>
      <th>NumberOfChildren</th>
      <th>Name_y</th>
      <th>State</th>
      <th>Type</th>
      <th>Popularity</th>
      <th>BestTimeToVisit</th>
      <th>PreferencesList</th>
      <th>PopularityNormalizado</th>
      <th>ExperienceRatingNormalizado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>525</td>
      <td>760</td>
      <td>2024-01-01</td>
      <td>3</td>
      <td>City, Historical</td>
      <td>Female</td>
      <td>2</td>
      <td>2</td>
      <td>Leh Ladakh</td>
      <td>Jammu and Kashmir</td>
      <td>adventure</td>
      <td>8.352180</td>
      <td>apr-jun</td>
      <td>[City,  Historical]</td>
      <td>0.424836</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>184</td>
      <td>532</td>
      <td>2024-02-15</td>
      <td>5</td>
      <td>Beaches, Historical</td>
      <td>Male</td>
      <td>1</td>
      <td>2</td>
      <td>Goa Beaches</td>
      <td>Goa</td>
      <td>beach</td>
      <td>8.988127</td>
      <td>nov-mar</td>
      <td>[Beaches,  Historical]</td>
      <td>0.743557</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>897</td>
      <td>786</td>
      <td>2024-03-20</td>
      <td>2</td>
      <td>City, Historical</td>
      <td>Female</td>
      <td>1</td>
      <td>2</td>
      <td>Taj Mahal</td>
      <td>Uttar Pradesh</td>
      <td>historical</td>
      <td>8.389206</td>
      <td>nov-feb</td>
      <td>[City,  Historical]</td>
      <td>0.443392</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>470</td>
      <td>660</td>
      <td>2024-01-01</td>
      <td>1</td>
      <td>Nature, Adventure</td>
      <td>Male</td>
      <td>2</td>
      <td>1</td>
      <td>Leh Ladakh</td>
      <td>Jammu and Kashmir</td>
      <td>adventure</td>
      <td>7.923388</td>
      <td>apr-jun</td>
      <td>[Nature,  Adventure]</td>
      <td>0.209936</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>989</td>
      <td>389</td>
      <td>2024-02-15</td>
      <td>4</td>
      <td>Nature, Adventure</td>
      <td>Male</td>
      <td>2</td>
      <td>1</td>
      <td>Kerala Backwaters</td>
      <td>Kerala</td>
      <td>nature</td>
      <td>9.409146</td>
      <td>sep-mar</td>
      <td>[Nature,  Adventure]</td>
      <td>0.954561</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f5309b41-e607-448b-8b18-e18f507b4227')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>

Creación de la matriz de usuarios para hacer la recomendación de destinos de viaje.
```python
user_item_matrix = df_copia.pivot_table(index='UserID', columns='DestinationID', values='ExperienceRatingNormalizado', fill_value=0)
svd = TruncatedSVD(n_components=20, random_state=42)
item_embeddings = svd.fit_transform(user_item_matrix.T)
destination_similarity = cosine_similarity(item_embeddings)

# Índices de destinos
destination_indices = {
    dest: idx for idx, dest in enumerate(user_item_matrix.columns)
}

# Base para contenido
destination_content = df_copia[[
    "DestinationID",
    "Name_y",
    "PopularityNormalizado",
    "PreferencesList",
    "Type",
    "BestTimeToVisit"
]].drop_duplicates(subset=["DestinationID"])

# MultiLabelBinarizer para PreferencesList (lista de preferencias de los usuarios)
mlb = MultiLabelBinarizer()
prefs_encoded = pd.DataFrame(
    mlb.fit_transform(destination_content["PreferencesList"]),
    columns=[f"Pref_{c}" for c in mlb.classes_]
)
prefs_encoded.index = destination_content.index

# One-hot encode Type y BestTimeToVisit
dummies = pd.get_dummies(destination_content[["Type", "BestTimeToVisit"]])

# Concatenar
destination_content_encoded = pd.concat([
    destination_content[["DestinationID", "Name_y", "PopularityNormalizado"]],
    dummies,
    prefs_encoded
], axis=1)

# Asegurarse que todo sea float
for col in destination_content_encoded.columns:
    if destination_content_encoded[col].dtype in [bool, int]:
        destination_content_encoded[col] = destination_content_encoded[col].astype(float)


user_similarity = cosine_similarity(destination_content_encoded.drop(columns=["DestinationID", "Name_y"]) )
destinations_names = df_copia[['DestinationID', 'Name_y']].drop_duplicates()
```
### Función para recomendación de los top 5 destinos con un alpha de 0.5 (uno con base en destinos similares que el usuario ya visitó collab_scores y otro con base en contenido de los destinos ya visitados content_scores) el alpha determinar que ambos parámetros tienen la misma importancia.
```python
def recommend_destinations(user_id,top_n=5,alpha=0.5):
  if user_id not in user_item_matrix.index:
    print('Usuario no válido')
    return 0
  visited_destinations = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
  collab_scores = np.zeros(len(user_item_matrix.columns))
  content_scores = np.zeros(len(user_item_matrix.columns))

  for d in visited_destinations:
    idx = destination_indices[d]
    collab_scores += destination_similarity[idx]
    content_scores += user_similarity[idx]

  if len(visited_destinations) > 0:
    collab_scores /= len(visited_destinations)
    content_scores /= len(visited_destinations)

  hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores

  recommendations = pd.DataFrame({
      "DestinationID": user_item_matrix.columns,
      "HybridScore": hybrid_scores
  })

  recommendations = recommendations[~recommendations["DestinationID"].isin(visited_destinations)]
  recommendations = recommendations.merge(destinations_names, on="DestinationID", how="left")
  recommendations = recommendations.groupby("Name_y").agg({"HybridScore": "mean"}).reset_index()
  recommendations = recommendations.sort_values("HybridScore", ascending=False).head(top_n)
  recommendations = recommendations.rename(columns={"Name_y": "DestinationName"})
  return recommendations
```
### Ahora pasamos a la función para recomendación por perfil del usuario. (ideal para usuarios nuevos)
```python
def recommend_by_profile(user_profile, top_n=5):
    # One-hot de Type (puedes cambiar la lógica si quieres)
    type_cols = [c for c in destination_content_encoded.columns if c.startswith("Type_")]
    type_vec = pd.Series(0.0, index=type_cols)

    # One-hot de BestTimeToVisit (puedes ajustar la lógica)
    time_cols = [c for c in destination_content_encoded.columns if c.startswith("BestTimeToVisit_")]
    time_vec = pd.Series(0.0, index=time_cols)

    # MultiLabel de Preferences
    pref_cols = [c for c in destination_content_encoded.columns if c.startswith("Pref_")]
    pref_vec = pd.Series(0.0, index=pref_cols)
    for pref in user_profile["Preferences"].split(", "):
        col = f"Pref_{pref.strip()}"
        if col in pref_vec.index:
            pref_vec[col] = 1.0

    # Popularidad media
    popularity_mean = destination_content_encoded["PopularityNormalizado"].mean()

    # Concatenar
    user_vector = pd.concat([
        pd.Series({"PopularityNormalizado": popularity_mean}),
        type_vec,
        time_vec,
        pref_vec
    ]).to_frame().T

    # Ordenar columnas igual que en destino
    user_vector = user_vector[destination_content_encoded.drop(columns=["DestinationID", "Name_y"]).columns]

    # Convertir a float
    user_vector = user_vector.astype(float)

    # Similitud
    sim = cosine_similarity(
        destination_content_encoded.drop(columns=["DestinationID", "Name_y"]),
        user_vector
    ).flatten()

    recommendations = destination_content_encoded[["DestinationID", "Name_y"]].copy()
    recommendations["Similarity"] = sim
    recommendations = recommendations.sort_values("Similarity", ascending=False).head(top_n)
    recommendations = recommendations.rename(columns={"Name_y": "DestinationName"})
    return recommendations
```
```python
# Recomendación híbrida basada en historial
sample_user = df_copia["UserID"].iloc[0]
recs_hybrid = recommend_destinations(sample_user, top_n=3, alpha=0.6)
print("\n=== Recomendaciones híbridas ===")
print(recs_hybrid)

# Recomendación basada en perfil
user_profile = {
    "Preferences": "Nature, Adventure",
    "Gender": "Female",
    "NumberOfAdults": 2,
    "NumberOfChildren": 1
}
recs_profile = recommend_by_profile(user_profile)
print("\n=== Recomendaciones basadas en perfil ===")
print(recs_profile)

```

    
    === Recomendaciones híbridas ===
      DestinationName  HybridScore
    1     Jaipur City     0.155404
    0     Goa Beaches     0.155140
    4       Taj Mahal     0.147836
    
    === Recomendaciones basadas en perfil ===
        DestinationID    DestinationName  Similarity
    860           987        Goa Beaches    0.601030
    517           468        Jaipur City    0.600173
    31            114  Kerala Backwaters    0.599672
    207           684  Kerala Backwaters    0.599624
    84            130         Leh Ladakh    0.599397

###  Evaluación de métricas

```python
import random

# -------------------------------
# Generador automático de perfiles
# -------------------------------
def generate_random_profiles(n_profiles=50):
    possible_preferences = [
        "Nature", "Adventure", "Culture", "Relaxation", "Beach", "Gastronomy"
    ]
    genders = ["Female", "Male"]
    num_adults_options = [1, 2, 3]
    num_children_options = [0, 1, 2]

    profiles = []
    for _ in range(n_profiles):
        selected_prefs = random.sample(possible_preferences, k=random.randint(1,2))
        profile = {
            "Preferences": ", ".join(selected_prefs),
            "Gender": random.choice(genders),
            "NumberOfAdults": random.choice(num_adults_options),
            "NumberOfChildren": random.choice(num_children_options)
        }
        profiles.append(profile)

    return profiles

# -------------------------------
# Evaluación contra múltiples usuarios
# -------------------------------
def evaluate_profile_recommendations_against_multiple_users(user_profile, recommended_df, user_ids):
    recommended_names = set(recommended_df["DestinationName"].values)

    precisions = []
    recalls = []

    for uid in user_ids:
        actual_destinations = set(df_copia[df_copia["UserID"] == uid]["Name_y"].unique())

        if not actual_destinations:
            continue  # Saltar usuarios sin historial

        hits = recommended_names.intersection(actual_destinations)

        precision = len(hits) / len(recommended_names) if recommended_names else 0
        recall = len(hits) / len(actual_destinations) if actual_destinations else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else None
    avg_recall = sum(recalls) / len(recalls) if recalls else None

    return avg_precision, avg_recall

# -------------------------------
# Generar perfiles de prueba
# -------------------------------
test_profiles = generate_random_profiles(n_profiles=50)

# -------------------------------
# Evaluar todos los perfiles
# -------------------------------
all_user_ids = df_copia["UserID"].unique()

all_precisions = []
all_recalls = []

for idx, profile in enumerate(test_profiles):
    # Generar recomendaciones para este perfil
    recs_profile = recommend_by_profile(profile, top_n=5)

    # Evaluar contra todos los usuarios reales
    avg_precision, avg_recall = evaluate_profile_recommendations_against_multiple_users(
        profile, recs_profile, all_user_ids
    )

    all_precisions.append(avg_precision)
    all_recalls.append(avg_recall)

    print(f"Perfil {idx+1}: Precision={avg_precision:.3f}, Recall={avg_recall:.3f}")

# -------------------------------
# Métricas promedio globales
# -------------------------------
overall_precision = sum(all_precisions) / len(all_precisions)
overall_recall = sum(all_recalls) / len(all_recalls)

print("\n=== Métricas de evaluación global (promedio sobre perfiles aleatorios) ===")
print(f"Precision promedio global: {overall_precision:.3f}")
print(f"Recall promedio global: {overall_recall:.3f}")

```

    Perfil 1: Precision=0.282, Recall=0.802
    Perfil 2: Precision=0.278, Recall=0.591
    Perfil 3: Precision=0.278, Recall=0.591
    Perfil 4: Precision=0.278, Recall=0.591
    Perfil 5: Precision=0.282, Recall=0.802
    Perfil 6: Precision=0.278, Recall=0.591
    Perfil 7: Precision=0.282, Recall=0.802
    Perfil 8: Precision=0.282, Recall=0.802
    Perfil 9: Precision=0.282, Recall=0.802
    Perfil 10: Precision=0.278, Recall=0.591
    Perfil 11: Precision=0.278, Recall=0.591
    Perfil 12: Precision=0.278, Recall=0.591
    Perfil 13: Precision=0.278, Recall=0.591
    Perfil 14: Precision=0.278, Recall=0.591
    Perfil 15: Precision=0.278, Recall=0.591
    Perfil 16: Precision=0.278, Recall=0.591
    Perfil 17: Precision=0.278, Recall=0.591
    Perfil 18: Precision=0.282, Recall=0.802
    Perfil 19: Precision=0.278, Recall=0.591
    Perfil 20: Precision=0.278, Recall=0.591
    Perfil 21: Precision=0.278, Recall=0.591
    Perfil 22: Precision=0.282, Recall=0.802
    Perfil 23: Precision=0.278, Recall=0.591
    Perfil 24: Precision=0.278, Recall=0.591
    Perfil 25: Precision=0.278, Recall=0.591
    Perfil 26: Precision=0.278, Recall=0.591
    Perfil 27: Precision=0.278, Recall=0.591
    Perfil 28: Precision=0.282, Recall=0.802
    Perfil 29: Precision=0.278, Recall=0.591
    Perfil 30: Precision=0.278, Recall=0.591
    Perfil 31: Precision=0.278, Recall=0.591
    Perfil 32: Precision=0.282, Recall=0.802
    Perfil 33: Precision=0.278, Recall=0.591
    Perfil 34: Precision=0.278, Recall=0.591
    Perfil 35: Precision=0.278, Recall=0.591
    Perfil 36: Precision=0.278, Recall=0.591
    Perfil 37: Precision=0.282, Recall=0.802
    Perfil 38: Precision=0.278, Recall=0.591
    Perfil 39: Precision=0.278, Recall=0.591
    Perfil 40: Precision=0.278, Recall=0.591
    Perfil 41: Precision=0.282, Recall=0.802
    Perfil 42: Precision=0.282, Recall=0.802
    Perfil 43: Precision=0.278, Recall=0.591
    Perfil 44: Precision=0.278, Recall=0.591
    Perfil 45: Precision=0.282, Recall=0.802
    Perfil 46: Precision=0.278, Recall=0.591
    Perfil 47: Precision=0.278, Recall=0.591
    Perfil 48: Precision=0.278, Recall=0.591
    Perfil 49: Precision=0.282, Recall=0.802
    Perfil 50: Precision=0.282, Recall=0.802
    
    === Métricas de evaluación global (promedio sobre perfiles aleatorios) ===
    Precision promedio global: 0.279
    Recall promedio global: 0.655
### Análisis de efectividad de las recomendaciones
El dataset utilizado no está muy probado y los datos a veces tienden a ser muy similares los unos a los otros a tal punto que al momento de recomendar nuevos destinos a usuarios ya en la base de datos esto podía causar resultados que no tenían mucho sentido. Por otro lado, la recomendación de destinos podría personalizarse aún más buscando un alpha adecuado dependiendo del usuario y que él determine a cual criterio le daría más importancia.
-   Usuarios con preferencias similares reciben recomendaciones alineadas.
    
-   Se puede observar un sesgo positivo hacia destinos populares, lo que puede reflejar una mayor demanda.
    
-   Las rutas más recomendadas podrían recibir mayor tráfico si estas recomendaciones se aplican en producción.
