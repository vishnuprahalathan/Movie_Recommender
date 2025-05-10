

import streamlit as st
st.set_page_config(page_title="Movie Recommendation System", layout="centered")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from textblob import TextBlob

@st.cache_data
def load_data():
    movies = pd.read_csv('C:\\Users\\Vishnu Prahalathan\\Desktop\\MR\\movies.csv')
    ratings = pd.read_csv('C:\\Users\\Vishnu Prahalathan\\Desktop\\MR\\ratings.csv')

 
    ratings = ratings.groupby('userId').filter(lambda x: len(x) > 50)
    ratings = ratings.groupby('movieId').filter(lambda x: len(x) > 100)
    return movies, ratings

movies, ratings = load_data()

@st.cache_resource
def prepare_content_model(movies, n_neighbors=11):
    movies = movies.copy()
    movies['genres'] = movies['genres'].fillna('')
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(movies['genres'])
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model.fit(count_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return model, count_matrix, indices

model_content, count_matrix, indices = prepare_content_model(movies)

def content_model(movie_title, movies, model, count_matrix, indices):
    if movie_title not in indices:
        return ["❌ Movie not found."]
    idx = indices[movie_title]
    distances, neighbors = model.kneighbors(count_matrix[idx], n_neighbors=11)
    return movies['title'].iloc[neighbors.flatten()[1:]].tolist()

@st.cache_resource
def prepare_collaborative_model_sparse(ratings):
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_map = {mid: idx for idx, mid in enumerate(movie_ids)}
    row = ratings['userId'].map(user_map)
    col = ratings['movieId'].map(movie_map)
    data = ratings['rating']
    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(movie_ids)))
    return sparse_matrix, user_map, movie_map, user_ids, movie_ids

user_movie_sparse, user_map, movie_map, user_ids, movie_ids = prepare_collaborative_model_sparse(ratings)

@st.cache_resource
def train_collab_model(_sparse_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
    model.fit(_sparse_matrix.T)
    return model

model_collab = train_collab_model(user_movie_sparse)

def collaborative_model(movie_title, movies, ratings, model, sparse_matrix, movie_map, movie_ids):
    movie_id_series = movies[movies['title'] == movie_title]['movieId']
    if movie_id_series.empty:
        return ["❌ Movie not found."]
    movie_id = movie_id_series.values[0]
    if movie_id not in movie_map:
        return ["⚠️ Not enough data for this movie."]
    movie_idx = movie_map[movie_id]
    distances, neighbors = model.kneighbors(sparse_matrix.T[movie_idx], n_neighbors=11)
    neighbor_ids = [movie_ids[i] for i in neighbors.flatten()[1:]]
    return movies[movies['movieId'].isin(neighbor_ids)]['title'].tolist()


def sentiment_filter(movie_list):
    filtered = []
    for movie in movie_list:
        try:
            polarity = TextBlob(movie).sentiment.polarity
            if polarity >= 0:
                filtered.append(movie)
        except:
            continue
    return filtered

st.title("🎬 Movie Recommendation System")
st.markdown("""
This app recommends top movies based on your input using content and collaborative filtering. You can also apply sentiment filtering to remove negatively perceived titles.
""")

sample_titles = movies['title'].dropna().unique()
selected_movie = st.selectbox("🎞️ Select a movie you like:", sorted(np.random.choice(sample_titles, 50, replace=False)))

rec_type = st.radio("🔍 Choose recommendation type:", ["Content-Based", "Collaborative Filtering"])
use_sentiment = st.checkbox("😊 Filter recommendations by sentiment (Optional)")

if st.button("🎯 Get Recommendations"):
    with st.spinner("Finding movies for you..."):
        if rec_type == "Content-Based":
            recs = content_model(selected_movie, movies, model_content, count_matrix, indices)
        else:
            recs = collaborative_model(selected_movie, movies, ratings, model_collab, user_movie_sparse, movie_map, movie_ids)

        if use_sentiment:
            recs = sentiment_filter(recs)

        if recs:
            st.success("Here are your top recommendations:")
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec}")
        else:
            st.warning("No recommendations found. Try a different movie.")
