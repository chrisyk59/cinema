import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz, process
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Configuration initiale
SIMILARITY_THRESHOLD = 80
TEXT_WEIGHT = 0.95
NUM_RECOMMENDATIONS = 5

# Fonction pour charger les données
@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/SeaJayEm/projet_2/refs/heads/main/df_list.csv')
    df.drop_duplicates(subset='localised_title', keep='first', inplace=True)
    return df

# Téléchargement des ressources NLTK
@st.cache_resource
def download_nltk_resources():
    nltk.download(['stopwords', 'wordnet', 'punkt'])

# Fonction pour nettoyer le texte
def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-z\\s]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)

# Préparation du dataframe
def prepare_dataframe(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    df = df.copy()
    list_columns = ['genres', 'Actors', 'Directors', 'Writers', 'production_countries']
    
    for col in list_columns:
        df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['keywords'] = df['overview'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    
    return df

# Création de la "soupe de fonctionnalités"
def create_feature_soup(row):
    elements = []
    elements.extend(row['genres'] * 2)
    elements.extend(row['Actors'])
    elements.extend(row['Directors'] * 2)
    elements.extend(row['production_countries'])
    elements.append(str(row['keywords']))
    return ' '.join(map(str, elements))

# Calcul de la matrice de similarité
def compute_similarity_matrix(df, text_weight):
    df['text_features'] = df.apply(create_feature_soup, axis=1)
    vectorizer = CountVectorizer(stop_words='english')
    text_matrix = vectorizer.fit_transform(df['text_features'])

    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(df[['year']].copy())

    text_similarity = cosine_similarity(text_matrix)
    numeric_similarity = cosine_similarity(numeric_features)

    return (text_weight * text_similarity + (1 - text_weight) * numeric_similarity)

# Trouver un film avec fuzzy matching
def find_movie(title, titles, threshold):
    match = process.extractOne(title, titles, scorer=fuzz.ratio, score_cutoff=threshold)
    return match if match else (None, 0)

# Obtenir des recommandations
def get_recommendations(df, similarity_matrix, movie_title, n):
    closest_title, match_score = find_movie(movie_title, df['localised_title'].tolist(), SIMILARITY_THRESHOLD)

    if not closest_title:
        return None

    idx = df[df['localised_title'] == closest_title].index[0]
    sim_scores = sorted(list(enumerate(similarity_matrix[idx])), 
                       key=lambda x: x[1], reverse=True)[1:n+1]

    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices][
        ['localised_title', 'year', 'genres', 'Directors', 'Actors', 'overview_fr', 'averageRating', 'poster_path']
    ].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return closest_title, match_score, recommendations

# Fonction pour afficher un film aléatoire avec détails
def afficher_film_aleatoire(df):
    random_movie = df[['localised_title', 'year', 'averageRating', 'overview_fr', 'poster_path']].sample(n=1)
    
    col1, col2 = st.columns([1, 2])  # 1 pour l'image, 2 pour les informations
    with col1:
        if pd.notna(random_movie['poster_path'].values[0]):
            poster_url = f"https://image.tmdb.org/t/p/w200{random_movie['poster_path'].values[0]}"
            st.image(poster_url, caption=random_movie['localised_title'].values[0], use_container_width=True)
        else:
            st.write("Aucune image disponible.")
    
    with col2:
        st.markdown(f"**Film :** {random_movie['localised_title'].values[0]}")
        st.markdown(f"**Année :** {random_movie['year'].values[0]}")
        st.markdown(f"**Note moyenne :** {random_movie['averageRating'].values[0]}")
        
        if not random_movie['overview_fr'].isnull().values[0]:
            st.markdown("**Résumé :**")
            st.markdown(f"*{random_movie['overview_fr'].values[0]}*")
        else:
            st.markdown("**Résumé :** Aucune information disponible.")
    
    return random_movie

# Fonction pour afficher 3 affiches dans la page d'accueil
def afficher_3_affiches(df):
    random_movies = df[['localised_title', 'poster_path']].sample(n=3)
    
    cols = st.columns(3)
    for i, (_, movie) in enumerate(random_movies.iterrows()):
        poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if pd.notna(movie['poster_path']) else None
        with cols[i]:
            if poster_url:
                st.image(poster_url, caption=movie['localised_title'], use_container_width=True)
            else:
                st.write("Aucune image disponible.")

# Fonction pour afficher le contenu en fonction de la sélection
def afficher_contenu(selection, df):
    if selection == "Accueil":
        st.write("Bienvenue sur la page d'accueil de **Ciné Réunion** ! 🎥")
        st.markdown("### Explorez les classiques du cinéma à travers notre interface interactive.")
        afficher_3_affiches(df)
        
    elif selection == "La base de données":
        st.write("### Détails de la base de données")
        st.dataframe(df.head(10))
        
    elif selection == "Recommandations de films":
        st.title("Moteur de recommandations de films 🎥")
        st.markdown("Cette section vous aidera à découvrir de nouveaux films basés sur vos préférences.")
        
        download_nltk_resources()
        df_prepared = prepare_dataframe(df)
        
        film_input = st.text_input("Nom du film", placeholder="Entrez le titre d'un film")
        
        if st.button("Recommander") and film_input:
            similarity_matrix = compute_similarity_matrix(df_prepared, TEXT_WEIGHT)
            result = get_recommendations(df_prepared, similarity_matrix, film_input, NUM_RECOMMENDATIONS)
            
            if result is None:
                st.error(f"Aucun film trouvé correspondant à '{film_input}'")
            else:
                closest_title, match_score, recommendations = result
                
                if closest_title != film_input:
                    st.info(f"Film trouvé : '{closest_title}' (score de similarité : {match_score:.1f}%)")
                
                st.write(f"Si vous avez aimé **{closest_title}**, vous pourriez aimer :")
                
                for _, row in recommendations.iterrows():
                        genres = ', '.join(row['genres'])
                        directors = ', '.join(row['Directors'])
                        poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if pd.notna(row['poster_path']):
                                st.image(poster_url, caption=row['localised_title'], use_container_width=True)
                            else:
                                st.write("Aucune image disponible.")
                        with col2:
                            st.write(f"- **{row['localised_title']}** ({row['year']})  \n"
                                    f"  Genres: {genres}  \n"
                                    f"  Réalisateur(s): {directors}  \n"
                                    f"  Résumé: {row['overview_fr']}")
            
    elif selection == "Explorer les films":
        st.title("Explorer les Films Anciens 📽️")
        st.markdown("Cette collection regroupe des films anciens avec leurs informations principales.")
        
        graph_type = st.selectbox(
            "Choisir un graphique",
            ["Aucun", "Distribution des genres", "Films par année", "Top films par note", 
             "Distribution des durées", "Notes moyennes par région"]
        )

        if graph_type == "Distribution des genres":
            genres_count = df['genres'].value_counts()
            genres_df = pd.DataFrame({'Genre': genres_count.index, 'Nombre': genres_count.values})
            fig = px.bar(genres_df, x="Genre", y="Nombre", color="Genre", height=400)
            st.plotly_chart(fig)
            
        elif graph_type == "Films par année":
            year_count = df['year'].value_counts().sort_index()
            year_df = pd.DataFrame({'Année': year_count.index, 'Nombre de films': year_count.values})
            fig = px.line(year_df, x='Année', y='Nombre de films')
            st.plotly_chart(fig)
            
        elif graph_type == "Top films par note":
            top_movies = df.nlargest(10, 'averageRating')
            fig = px.bar(top_movies, x='localised_title', y='averageRating', color='averageRating')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
        elif graph_type == "Distribution des durées":
            fig = px.histogram(df, x='runtime', nbins=50, title="Distribution des durées")
            st.plotly_chart(fig)
            
        elif graph_type == "Notes moyennes par région":
            region_ratings = df.groupby('region')['averageRating'].mean().reset_index()
            fig = px.bar(region_ratings, x='region', y='averageRating', color='region', height=400)
            st.plotly_chart(fig)
            
    elif selection == "Random":
        st.markdown("### Film aléatoire 🎬")
        
        if st.button("Reset"):
            st.session_state.random_movie = afficher_film_aleatoire(df)
        elif 'random_movie' not in st.session_state:
            st.session_state.random_movie = afficher_film_aleatoire(df)

# Fonction principale
def main():
    df = load_data()

    with st.sidebar:
        selection = option_menu(
            "Menu",
            ["Accueil", "La base de données", "Recommandations de films", 
             "Explorer les films", "Random"],
            icons=["house", "database", "search", "film", "shuffle"],
            menu_icon="cast",
            default_index=0
        )

    if selection == "Accueil":
        st.title("L'Antre des Cinéphiles  🎬")
    
    afficher_contenu(selection, df)

if __name__ == "__main__":
    main()