import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

    def recommend(self, track_name, number_songs):
        # Find the index of the song in the dataset
        song_index = self.data[self.data['track_name'] == track_name].index
        if len(song_index) == 0:
            print("Song not found in the dataset.")
            return
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
                    'similarity_score': song_similarities[i],
                    'features': {feature: track_info[feature] for feature in self.numeric_features}
                })

        # Print recommendations
        for i, rec in enumerate(recommendations, start=1):
            print(f"Number {i}:")
            print(f"{rec['track_name']} by {rec['artist']} with {rec['similarity_score']} similarity score calculated based on:")
            for feature, value in rec['features'].items():
                print(f"- {feature}: {value}")
            print("-" * 20)

if __name__ == "__main__":
    # Load the dataset
    songs = pd.read_csv('C:/Users/Sabar/Desktop/Mini Project/spotify dataset.csv')  # Update the path to your dataset

    # Create an instance of the recommender
    recommender = ContentBasedRecommender(songs)

    # Take input from the user
    track_name = input("Enter the track name: ")
    number_songs = int(input("Enter the number of songs to recommend: "))

    # Generate recommendations
    recommender.recommend(track_name, number_songs)
