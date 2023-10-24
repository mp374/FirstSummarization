from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from summrizers.rank_based_summarizer import sentence_summery


def cosine_based_summery(num_clusters, cluster_model, stories, vectorized_stories):
    # Summarize the content for each cluster
    for cluster_num in range(num_clusters):
        cluster_indices = np.where(cluster_model.labels_ == cluster_num)[0]
        cluster_stories = [stories[i] for i in cluster_indices]

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(vectorized_stories[cluster_indices], vectorized_stories[cluster_indices])

        # Find the most representative story
        representative_index = np.argmax(np.sum(similarity_matrix, axis=1))
        representative_story = cluster_stories[representative_index]

        print(f"\n#####Cluster {cluster_num + 1} Summary:")
        print(representative_story)
        sentence_summery(representative_story)
