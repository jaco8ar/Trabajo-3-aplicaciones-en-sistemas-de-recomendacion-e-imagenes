import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

#maps de traducción

tiempo_map = {
    'Nov-Mar': 'Nov-Mar',
    'Nov-Feb': 'Nov-Feb',
    'Apr-Jun': 'Abr-Jun',
    'Oct-Mar': 'Oct-Mar',
    "Sep-Mar": "Sep-Mar"
}

tipo_map = {
    'Beach': 'Playa',
    'Historical': 'Histórico',
    'Adventure': 'Aventura',
    "City": "Ciudad",
    "Nature": "Naturaleza"
}



# --------------------- Cargar modelos y datos ---------------------
st.title("🔍 Sistema de Recomendación de Destinos")

path = "recommender_matrices"

@st.cache_resource
def load_data():
    destination_features = pd.read_csv(f"{path}/destination_features.csv")
    destination_features['DestinationID'] = destination_features['DestinationID'].astype(int)
    user_item_matrix = pd.read_csv(f"{path}/user_item_matrix.csv", index_col=0)
    user_item_matrix.columns = user_item_matrix.columns.astype(int)
    feature_matrix = load_npz(f"{path}/feature_matrix.npz")
    user_similarity = np.load(f"{path}/user_similarity.npy")
    return destination_features, user_item_matrix, feature_matrix, user_similarity

destination_features, user_item_matrix, feature_matrix, user_similarity = load_data()
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(destination_features['features'].values.astype(str))  # asegúrate de tener esta columna

# --------------------- Función principal ---------------------
def hybrid_recommend(user_id, user_item_matrix, user_similarity, destinations_df, feature_matrix, vectorizer,
                     user_preferences=None, top_n=5, k_neighbors=5):

    destination_ids = destinations_df['DestinationID'].values

    destinations_df['Type'] = destinations_df['Type'].map(tipo_map).fillna(destinations_df['Type'])
    destinations_df['BestTimeToVisit'] = destinations_df['BestTimeToVisit'].map(tiempo_map).fillna(destinations_df['BestTimeToVisit'])

    if user_id in user_item_matrix.index:
        user_idx = user_item_matrix.index.get_loc(user_id)
        similarities = user_similarity[user_idx]
        similar_users_idx = np.argsort(similarities)[::-1][1:k_neighbors+1]
        similar_users = user_item_matrix.index[similar_users_idx]

        weighted_ratings = np.zeros(user_item_matrix.shape[1])
        similarity_sum = np.zeros(user_item_matrix.shape[1])

        for i, neighbor_id in enumerate(similar_users):
            sim = similarities[similar_users_idx[i]]
            neighbor_ratings = user_item_matrix.loc[neighbor_id].values
            weighted_ratings += sim * neighbor_ratings
            similarity_sum += (neighbor_ratings > 0) * sim

        with np.errstate(divide='ignore', invalid='ignore'):
            predicted_ratings = np.true_divide(weighted_ratings, similarity_sum)
            predicted_ratings[np.isnan(predicted_ratings)] = 0

        user_rated = user_item_matrix.loc[user_id]
        unrated_mask = user_rated == 0
        unrated_ratings = predicted_ratings[unrated_mask.values]
        unrated_destinations = user_item_matrix.columns[unrated_mask]

        # Crear DataFrame con predicciones completas antes del top_n
        full_predictions = pd.DataFrame({
            'DestinationID': unrated_destinations,
            'PredictedRating': unrated_ratings
        })

        # Unir con datos del destino
        merged = pd.merge(full_predictions, destinations_df, on='DestinationID', how='left')

        # Filtrar filas que no lograron emparejar un destino válido
        merged = merged[merged['Name'].notna()]

        merged = merged[merged['PredictedRating'] > 0]

        if merged.empty:
            st.warning("No hay recomendaciones con puntuación positiva para mostrar.")
            
        # Eliminar duplicados por nombre y quedarnos con el de mayor predicted rating
        merged = merged.sort_values('PredictedRating', ascending=False)
        merged = merged.drop_duplicates(subset='Name', keep='first')

        recommendations = merged.head(top_n).copy()
        

    else:
        if user_preferences is None:
            st.warning("Debes ingresar tus preferencias si eres un usuario nuevo.")
            return pd.DataFrame()

        pref_str = (
            user_preferences.get('Type', '') + ' ' +
            user_preferences.get('State', '') + ' ' +
            user_preferences.get('BestTimeToVisit', '') + ' ' +
            user_preferences.get('Preferences', '')
        ).lower()

        pref_vector = vectorizer.transform([pref_str])
        similarities = cosine_similarity(pref_vector, feature_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1]

        recommendations = destinations_df.iloc[top_indices].copy()
        recommendations['PredictedRating'] = similarities[top_indices]
        recommendations = recommendations[recommendations['PredictedRating'] > 0]
       

        # Eliminar duplicados por nombre antes de seleccionar el top_n
        recommendations = recommendations.sort_values("PredictedRating", ascending=False)
        recommendations = recommendations.drop_duplicates(subset="Name", keep="first")
        recommendations = recommendations.head(top_n).copy()

    return recommendations[['DestinationID', 'Name', 'Type', 'State', 'BestTimeToVisit', 'Popularity', 'PredictedRating']]

# --------------------- Interfaz de usuario ---------------------

st.sidebar.header("📋 Configuración de usuario")
user_id_input = st.sidebar.text_input("ID de usuario", "")
top_n = st.sidebar.slider("Número de recomendaciones", 1, 10, 5)
k_neighbors = st.sidebar.slider("Número de vecinos similares (si aplica)", 1, 20, 10)

try:
    user_id = int(user_id_input)
except ValueError:
    user_id = None

if user_id is not None and user_id in user_item_matrix.index:
    st.success("Usuario conocido. Generando recomendaciones basandonos en los gustos de otros usuarios con gustos parecidos a los tuyos")
    recomendaciones = hybrid_recommend(user_id, user_item_matrix, user_similarity, destination_features,
                                       feature_matrix, vectorizer, top_n=top_n, k_neighbors=k_neighbors)
    recomendaciones = recomendaciones.rename(columns={
            'Name': 'Nombre',
            'Type': 'Tipo',
            'State': 'Estado',
            'BestTimeToVisit': 'Mejor época para visitar',
            'Popularity': 'Popularidad',
            'PredictedRating': 'Puntuación prevista'
        })
    st.dataframe(recomendaciones[['Nombre', 'Tipo', 'Estado', 'Mejor época para visitar', 'Popularidad', 'Puntuación prevista']])

else:
    st.info("Usuario nuevo. Por favor ingresa tus preferencias. ")
    col1, col2 = st.columns(2)
    with col1:
        tipo = st.selectbox("Tipo de destino", destination_features['Type'].unique())
        estado = st.selectbox("Estado", destination_features['State'].unique())
    with col2:
        tiempo = st.selectbox("Mejor época para visitar", destination_features['BestTimeToVisit'].unique())
        preferencias = st.text_input("Palabras clave (separadas por coma)", "Playa,Cultura")

    if st.button("Obtener recomendaciones"):
        st.info("Estamos generando las mejores recomendaciones de acuerdo a tus preferencias")
        prefs = {
            'Type': tipo,
            'State': estado,
            'BestTimeToVisit': tiempo,
            'Preferences': preferencias
        }
        recomendaciones = hybrid_recommend(user_id or 999999, user_item_matrix, user_similarity,
                                           destination_features, feature_matrix, vectorizer,
                                           user_preferences=prefs, top_n=top_n, k_neighbors=k_neighbors)
        recomendaciones = recomendaciones.rename(columns={
            'Name': 'Nombre',
            'Type': 'Tipo',
            'State': 'Estado',
            'BestTimeToVisit': 'Mejor época para visitar',
            'Popularity': 'Popularidad',
            'PredictedRating': 'Puntuación prevista'
        })
        st.dataframe(recomendaciones[['Nombre', 'Tipo', 'Estado', 'Mejor época para visitar', 'Popularidad', 'Puntuación prevista']])
