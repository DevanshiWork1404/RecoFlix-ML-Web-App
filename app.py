from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import pickle

app = Flask(__name__)
CORS(app)

# Load only the dataframe (small, fast)
df = pickle.load(open("movies.pkl", "rb"))

# Fit vectorizer once on startup
tfidf = TfidfVectorizer(stop_words='english')
vector = tfidf.fit_transform(df['combined'])

@lru_cache(maxsize=200)
def recommend(movie):
    try:
        index = df[df['title'] == movie].index[0]
        movie_vector = vector[index]
        scores = cosine_similarity(movie_vector, vector)[0]
        distances = list(enumerate(scores))
        movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:8]
        return [df.iloc[i[0]].title for i in movies_list]
    except IndexError:
        return []

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    movie = request.json.get("movie", "")
    results = recommend(movie)
    if not results:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify({"recommendations": list(results)})

@app.route("/movies", methods=["GET"])
def get_movies():
    return jsonify({"movies": df['title'].tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)