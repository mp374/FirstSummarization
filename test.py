from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Sample short texts
short_texts = [
    "I love hiking in the mountains",
    "Basketball is my favorite sport",
    "I enjoy reading novels",
    "Traveling to new places is exciting",
    "Learning new things is a great experience",
    "Cooking delicious meals is my passion"
]

# Convert short texts to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
vector_representation = vectorizer.fit_transform(short_texts)

# Perform K-means clustering
num_clusters = 2  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(vector_representation)

# Display text for each cluster
for cluster_id in range(num_clusters):
    cluster_texts = [short_texts[i] for i in range(len(short_texts)) if clusters[i] == cluster_id]
    print(f"Cluster {cluster_id + 1} Texts:")
    for text in cluster_texts:
        print(text)
    print()

# Get cluster centroids (representative points)
cluster_centroids = kmeans.cluster_centers_

# Print cluster definitions based on centroid words
for cluster_id in range(num_clusters):
    centroid_indices = cluster_centroids[cluster_id].argsort()[::-1]
    top_words = [vectorizer.get_feature_names_out()[index] for index in centroid_indices[:5]]
    print(f"Cluster {cluster_id + 1} Definition: {' '.join(top_words)}")
