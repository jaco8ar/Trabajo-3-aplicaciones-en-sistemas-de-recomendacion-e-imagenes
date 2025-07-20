import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# ------------------ Título ------------------
st.title("🌎 Recomendador de Destinos Turísticos")

# ------------------ Cargar datos ------------------
@st.cache_resource
def load_data():
    destinos = pd.read_csv("resources/recommender_matrices/Expanded_Destinations.csv")
    reviews = pd.read_csv("resources/recommender_matrices/Final_Updated_Expanded_Reviews.csv")
    historial = pd.read_csv("resources/recommender_matrices/Final_Updated_Expanded_UserHistory.csv")
    usuarios = pd.read_csv("resources/recommender_matrices/Final_Updated_Expanded_Users.csv")

    # Tipado
    for df in [usuarios, historial, reviews]:
        df["UserID"] = df["UserID"].astype(str)
    for df in [destinos, historial, reviews]:
        df["DestinationID"] = df["DestinationID"].astype(str)

    # Merge total
    df = historial.merge(usuarios, on="UserID", how="left").merge(destinos, on="DestinationID", how="left")
    df["VisitDate"] = pd.to_datetime(df["VisitDate"])
    df["PreferencesList"] = df["Preferences"].str.split(',')
    df["Type"] = df["Type"].str.lower()
    df["BestTimeToVisit"] = df["BestTimeToVisit"].str.lower()

    # Normalización
    df["PopularityNormalizado"] = (df["Popularity"] - df["Popularity"].min()) / (df["Popularity"].max() - df["Popularity"].min())
    df["ExperienceRatingNormalizado"] = (df["ExperienceRating"] - df["ExperienceRating"].min()) / (df["ExperienceRating"].max() - df["ExperienceRating"].min())

    return df

df = load_data()

# ------------------ Matriz usuario-item y SVD ------------------
user_item_matrix = df.pivot_table(index='UserID', columns='DestinationID', values='ExperienceRatingNormalizado', fill_value=0)

svd = TruncatedSVD(n_components=20, random_state=42)
item_embeddings = svd.fit_transform(user_item_matrix.T)
destination_similarity = cosine_similarity(item_embeddings)

destination_indices = {dest: idx for idx, dest in enumerate(user_item_matrix.columns)}
id_to_name = df.drop_duplicates("DestinationID").set_index("DestinationID")["Name_y"].to_dict()

# ------------------ Contenido destino ------------------
destination_content = df[[
    "DestinationID", "Name_y", "PopularityNormalizado", "PreferencesList", "Type", "BestTimeToVisit"
]].drop_duplicates(subset=["DestinationID"])

mlb = MultiLabelBinarizer()
prefs_encoded = pd.DataFrame(
    mlb.fit_transform(destination_content["PreferencesList"]),
    columns=[f"Pref_{c.strip().lower()}" for c in mlb.classes_]
)
prefs_encoded.index = destination_content.index

# One-hot encoding para 'Type' y 'BestTimeToVisit'
dummies = pd.get_dummies(destination_content[["Type", "BestTimeToVisit"]])
content_features = pd.concat([prefs_encoded, dummies, destination_content[["PopularityNormalizado"]]], axis=1)

# ------------------ Función de recomendación ------------------
def recomendar_destinos(user_id=None, preferencias=None, top_n=5):
    if user_id and user_id in user_item_matrix.index:
        st.success("✅ Usuario conocido. Recomendaciones basadas en destinos similares a los visitados.")
        user_ratings = user_item_matrix.loc[user_id]
        rated_destinations = user_ratings[user_ratings > 0].index.tolist()

        if not rated_destinations:
            st.warning("Este usuario no tiene destinos valorados. Usa recomendaciones basadas en contenido.")
            return pd.DataFrame()

        scores = np.zeros(len(user_item_matrix.columns))
        for dest_id in rated_destinations:
            idx = destination_indices[dest_id]
            scores += destination_similarity[idx] * user_ratings[dest_id]

        scores_df = pd.DataFrame({
            'DestinationID': user_item_matrix.columns,
            'Score': scores
        })

        scores_df = scores_df[~scores_df['DestinationID'].isin(rated_destinations)]
        scores_df['Nombre'] = scores_df['DestinationID'].map(id_to_name)
        scores_df = scores_df[scores_df['Score'] > 0]
        scores_df = scores_df.drop_duplicates(subset='Nombre', keep='first')
        scores_df = scores_df.sort_values(by='Score', ascending=False).head(top_n)

        return scores_df[['Nombre', 'Score']]

    elif preferencias:
        st.success("🧠 Usuario nuevo. Recomendaciones basadas en contenido.")
        tipo = preferencias['Type'].lower()
        epoca = preferencias['BestTimeToVisit'].lower()
        palabras = [p.strip().lower() for p in preferencias['Preferences'].split(",")]

        vector = np.zeros(content_features.shape[1])
        for i, col in enumerate(content_features.columns):
            if col == f"Type_{tipo}" or col == f"BestTimeToVisit_{epoca}" or col in [f"Pref_{p}" for p in palabras]:
                if col in content_features.columns:
                    vector[i] = 1

        sims = cosine_similarity([vector], content_features.values).flatten()
        top_idx = sims.argsort()[::-1]

        recomendados = destination_content.iloc[top_idx].copy()
        recomendados["Score"] = sims[top_idx]
        recomendados.rename(columns={"Name_y": "Nombre"}, inplace=True)
        recomendados = recomendados[recomendados["Score"] > 0]
        recomendados = recomendados.drop_duplicates(subset="Nombre", keep="first")
        recomendados = recomendados.head(top_n)

        return recomendados[["Nombre", "Score"]]

    else:
        st.warning("🚫 No se proporcionó información suficiente.")
        return pd.DataFrame()

# ------------------ Interfaz ------------------
st.sidebar.header("🎛️ Configuración")

user_id_input = st.sidebar.text_input("ID de usuario (opcional)", "")
try:
    user_id = str(int(user_id_input)) if user_id_input else None
except ValueError:
    user_id = None

top_n = st.sidebar.slider("Número de recomendaciones", 1, 10, 5)

if not user_id:
    st.sidebar.subheader("Preferencias del nuevo usuario")
    tipo = st.sidebar.selectbox("Tipo de destino", sorted(df["Type"].dropna().unique()))
    epoca = st.sidebar.selectbox("Mejor época para visitar", sorted(df["BestTimeToVisit"].dropna().unique()))
    preferencias = st.sidebar.text_input("Preferencias (coma separadas)", "playa,cultura,naturaleza")

    if st.sidebar.button("Recomendar"):
        prefs = {
            "Type": tipo,
            "BestTimeToVisit": epoca,
            "Preferences": preferencias
        }
        resultados = recomendar_destinos(user_id=None, preferencias=prefs, top_n=top_n)
        st.subheader("📌 Recomendaciones basadas en contenido")
        st.dataframe(resultados)
    else: 
        st.warning("Es posible que se tengan menos destinos recomendados que el número de destinos que usted solicite, estamos trabajando para ofrecerle más destinos en un futuro cercano.")

    
    
elif user_id:
    resultados = recomendar_destinos(user_id=user_id, top_n=top_n)
    st.subheader("📌 Recomendaciones personalizadas")
    st.dataframe(resultados)
