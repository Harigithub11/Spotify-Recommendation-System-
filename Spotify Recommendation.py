import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load dataset (ensure `spotify_dataset.csv` is in the same directory)
try:
    songs = pd.read_csv("spotify_dataset.csv")
except FileNotFoundError:
    songs = None
    print("Error: Dataset file not found. Ensure 'spotify_dataset.csv' is in the project directory.")

# Recommender Class
class ContentBasedRecommender:
    def __init__(self, data):
        self.data = data
        self.numeric_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 
                                 'liveness', 'valence', 'tempo', 'duration_ms']

        # Normalize numeric features
        self.normalized_data = self.data.copy()
        self.normalized_data[self.numeric_features] = self.normalized_data[self.numeric_features].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
        )

        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.normalized_data[self.numeric_features])

    def recommend(self, track_name, number_songs=5):
        # Find the index of the song in the dataset
        song_index = self.data[self.data['track_name'].str.lower() == track_name.lower()].index
        if len(song_index) == 0:
            return {"error": "Song not found in the dataset."}
        song_index = song_index[0]

        # Get similarity scores for the song and sort them in descending order
        song_similarities = self.similarity_matrix[song_index]
        similar_indices = np.argsort(-song_similarities)

        # Prepare the recommendation list
        recommendations = []
        for i in similar_indices:
            if len(recommendations) >= number_songs:
                break
            if i == song_index:  # Skip the song itself
                continue
            track_info = self.data.iloc[i]
            if track_info['track_name'] not in [rec['track_name'] for rec in recommendations]:  # Avoid duplicates
                recommendations.append({
                    'track_name': track_info['track_name'],
                    'artist': track_info['track_artist'],
                    'similarity_score': float(song_similarities[i]),  # Convert numpy float to regular float
                    'features': {feature: float(track_info[feature]) for feature in self.numeric_features}
                })

        return {"recommendations": recommendations}

# Initialize the recommender if data is available
if songs is not None:
    recommender = ContentBasedRecommender(songs)

# API Route
@app.route("/")
def home():
    return "Welcome to the Spotify Recommendation API!"

@app.route("/recommend", methods=["GET"])
def recommend():
    if songs is None:
        return jsonify({"error": "Dataset not found"}), 500

    track_name = request.args.get("song")
    number_songs = request.args.get("num", default=5, type=int)

    if not track_name:
        return jsonify({"error": "Please provide a song name using the 'song' parameter"}), 400

    recommendations = recommender.recommend(track_name, number_songs)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
