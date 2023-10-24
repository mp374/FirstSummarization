import numpy as np
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler


def Latent_Dirichlet_Allocation_TM_TFIDF(num_clusters, clustering_model, vectorizer, stories):
    # Extract topics for each cluster using Latent Dirichlet Allocation (LDA)
    num_topics = 1  # Number of topics to extract per cluster
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    for cluster_num in range(num_clusters):
        cluster_indices = np.where(clustering_model.labels_ == cluster_num)[0]
        cluster_stories = [stories[i] for i in cluster_indices]
        vectorized_cluster = vectorizer.transform(cluster_stories)
        lda.fit(vectorized_cluster)
        cluster_topics = lda.components_

        print(f"Cluster {cluster_num + 1} Topics:")
        for topic_num, topic_weights in enumerate(cluster_topics):
            top_words_indices = topic_weights.argsort()[:-5:-1]  # Top 4 words per topic
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_indices]
            print(f"Topic {topic_num + 1}: {', '.join(top_words)}")
        print("Sample Stories:")
        sample_stories = np.random.choice(cluster_stories, size=min(3, len(cluster_stories)), replace=False)
        for story in sample_stories:
            print("\n", story)
        print()