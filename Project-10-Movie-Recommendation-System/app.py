import streamlit as st
import pandas as pd
import pickle

st.title("Movie Recommendation System")

# Load saved objects
movies = pickle.load(open(src/"movies.pkl","rb"))
movie_indices = pickle.load(open(src/"movie_indices.pkl","rb"))
cosine_sim = pickle.load(open(src/"cosine_sim.pkl","rb"))
knn_model = pickle.load(open(src/"knn_model.pkl","rb"))
movie_user_matrix = pickle.load(open(src/"movie_user_matrix.pkl","rb"))
movie_matrix = pickle.load(open(src/"movie_matrix.pkl","rb"))
corr_matrix = pickle.load(open(src/"corr_matrix.pkl","rb"))

# -------- Recommendation Functions --------

def recommend_content(movie_name, n=10):

    idx = movie_indices[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:n+1]

    movie_indices_sim = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices_sim]


def recommend_knn(movie_name, n=10):

    movie_list = movie_user_matrix.index

    movie_idx = movie_list.get_loc(movie_name)

    distances, indices = knn_model.kneighbors(
        movie_user_matrix.iloc[movie_idx,:].values.reshape(1,-1),
        n_neighbors=n+1
    )

    recommendations = []

    for i in range(1,len(indices.flatten())):

        recommendations.append(movie_list[indices.flatten()[i]])

    return recommendations


def recommend_svd(movie_name, n=10):

    movie_index = movie_matrix.index.get_loc(movie_name)

    corr_movie = corr_matrix[movie_index]

    similar_indices = corr_movie.argsort()[::-1][1:n+1]

    return movie_matrix.index[similar_indices]


def hybrid_recommend(movie_name, n=10):

    content_rec = list(recommend_content(movie_name, n))

    knn_rec = list(recommend_knn(movie_name, n))

    svd_rec = list(recommend_svd(movie_name, n))

    combined = content_rec + knn_rec + svd_rec

    final = list(set(combined))

    return final[:n]

# -------- UI --------

movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Select Movie",
    movie_list
)

if st.button("Recommend"):

    recommendations = hybrid_recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write(movie)