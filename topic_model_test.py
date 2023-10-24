import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Sample tweet data
tweets = [
    "I love the weather today",
    "I'm feeling great",
    "The concert was amazing",
    "I can't believe I lost my wallet",
    "Such a boring day",
    "The movie was fantastic",
    "I'm so excited about the game",
    "I hate Mondays"
]

# Convert the tweet text into feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets)

# Perform K-means clustering
num_clusters = 2  # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Extract topics for each cluster using Latent Dirichlet Allocation (LDA)
num_topics = 1  # Number of topics to extract per cluster
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
for cluster_num in range(num_clusters):
    cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
    cluster_tweets = [tweets[i] for i in cluster_indices]
    X_cluster = vectorizer.transform(cluster_tweets)
    lda.fit(X_cluster)
    cluster_topics = lda.components_

    print(f"Cluster {cluster_num + 1} Topics:")
    for topic_num, topic_weights in enumerate(cluster_topics):
        top_words_indices = topic_weights.argsort()[:-5:-1]  # Top 4 words per topic
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_indices]
        print(f"Topic {topic_num + 1}: {', '.join(top_words)}")
    print("Sample Tweets:")
    sample_tweets = np.random.choice(cluster_tweets, size=min(3, len(cluster_tweets)), replace=False)
    for tweet in sample_tweets:
        print(tweet)
    print()