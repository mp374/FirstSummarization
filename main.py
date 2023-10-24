import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

from data_set_managers.merge_dataset import read_file_content, get_list_all_stories
from pre_processing.pre_processing import *
from clustering_algorithms.kmeans_models import *

# Keep model file name as global variable so that changing this will change everything.
from summrizers.cosine_based_summerizer import cosine_based_summery
from vectorizers.word2vec_vectorizer import word2vec_vectorizer
import numpy as np

modelNameSuffix = "Lemma"

if __name__ == '__main__':

    story_file_path = '/Users/heshankavinda/Library/CloudStorage/OneDrive-UniversityofPlymouth/PROJ518/Project/First_Attempt/data_set/opinion/2022_01_01.json'
    stories = get_list_all_stories()
    #stories = ["I was diagnosed with a rare condition in 2018 and still don't have a treatment plan."]

    # Second argument should either be "Stem" or "Lemma" or "None"
    tokenized_stories = pre_process_tokenize(stories, modelNameSuffix)

    vectorizer, vector_representation = word2vec_vectorizer(tokenized_stories)

    # Retrieve vector for the target word
    """target_vector = vectorizer.wv["staff"]

    # Get similar words
    similar_words = vectorizer.wv.similar_by_vector(target_vector, topn=10)

    # Print similar words
    for word, score in similar_words:
        print(word, score)"""

    # Perform K-means clustering for different values of k
    distortions = []
    K = range(1, len(stories))
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(vector_representation)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    number_of_cluters = 2
    clusters = kmeans_clusters(vectorized_data=vector_representation, k=number_of_cluters)

    num_topics = 1  # Number of topics to extract per cluster
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    for cluster_num in range(number_of_cluters):
        cluster_indices = np.where(clusters.labels_ == cluster_num)[0]
        cluster_stories = [stories[i] for i in cluster_indices]
        tokenized_cluster_stories = [word_tokenize(story.lower()) for story in cluster_stories]
        cluster_story_vectors = [np.mean([vectorizer.wv[word] for word in story_tokens if word in vectorizer.wv], axis=0)
                                 for story_tokens in tokenized_cluster_stories]
        X_cluster = np.array(cluster_story_vectors)

        # Normalize the data using min-max scaling
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(X_cluster)

        lda.fit(normalized_data)
        cluster_topics = lda.components_

        print(f"Cluster {cluster_num + 1} Topics:")
        for topic_num, topic_weights in enumerate(cluster_topics):
            top_words_indices = topic_weights[:len(vectorizer.wv)].argsort()[:-5:-1]  # Top 4 words per topic
            top_words = [vectorizer.wv.index_to_key[i] for i in top_words_indices]
            print(f"Topic {topic_num + 1}: {', '.join(top_words)}")
        print("Sample Stories:")
        sample_stories = np.random.choice(cluster_stories, size=min(3, len(cluster_stories)), replace=False)
        for tweet in sample_stories:
            print(tweet)
        print()
