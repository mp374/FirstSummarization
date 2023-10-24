from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from clustering_algorithms.kmeans_models import kmeans_clusters
from data_set_managers.merge_dataset import read_file_content, get_list_all_stories
from pre_processing.pre_processing import *
from summrizers.cosine_based_summerizer import cosine_based_summery
from topic_extracters.lda_topic_modelling import Latent_Dirichlet_Allocation_TM_TFIDF
from vectorizers.tf_idf_vectorizer import tf_idf_vectorize
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from wordcloud import WordCloud

# Keep model file name as global variable so that changing this will change everything.
modelNameSuffix = "Lemma"

if __name__ == '__main__':
    story_file_path = '/Users/heshankavinda/Library/CloudStorage/OneDrive-UniversityofPlymouth/PROJ518/Project/First_Set_of_Algo/data_set/test_stories_1.json'
    stories = read_file_content(story_file_path)
    #stories = ["I felt the need to give a good review about this surgery. On the last three occasions I have needed this service within the last year, I have always been able to speak to the receptionist without waiting too long in a queue and they are always polite and efficient. A follow up triage phone call has usually been no more than one hour later and if required have been given an ap-pointment that same day. I have found the doctors very thorough and didn’t feel rushed given a plan of action that will help my health issues. Even better now that I can receive a text message regarding test re-sults and any further action. In the last 2 years the surgery seems to have improved in all areas. I think the staff deserve some recognition for all their hard work."]

    # Second argument should either be "Stem" or "Lemma" or "None"
    pre_processed_stories = preprocess(stories, modelNameSuffix)

    start_time = time.time()

    # vectorize the stories
    vectorizer, vector_representation = tf_idf_vectorize(pre_processed_stories)

    # List of words in the vocabulary
    """words = vectorizer.get_feature_names_out()

    # Word of interest
    target_word = "staff"

    # Find index of the target word in the vocabulary
    word_index = np.where(words == target_word)

    # Retrieve TF-IDF score for the word in the first document
    tfidf_score = vector_representation[0, word_index]
    tfidf_score2 = vector_representation[1, word_index]
    tfidf_score3 = vector_representation[2, word_index]
    tfidf_score4 = vector_representation[3, word_index]"""

    # Perform K-means clustering for different values of k
    distortions = []
    K = range(2, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(vector_representation)
        distortions.append(kmeanModel.inertia_)

    # Apply a moving average to smooth the curve
    window_size = 2  # Adjust the window size as needed
    smoothed_distortions = np.convolve(distortions, np.ones(window_size) / window_size, mode='same')

    # Create the Elbow Method plot with the smoothed curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, smoothed_distortions, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Smoothed Distortion (Inertia)')
    plt.title('Smoothed Elbow Method for Optimal K')
    plt.grid(True)

    sit = []
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vector_representation)
        silhouette_avg = silhouette_score(vector_representation, cluster_labels)
        sit.append(silhouette_avg)
        print(f"For {n_clusters} clusters, the average silhouette score is {silhouette_avg:.4f}")

    plt.figure(figsize=(8, 8))
    plt.plot(K, sit, 'bx-')
    plt.xlabel('K', fontsize=16)
    plt.ylabel('Silhouette score', fontsize=16)
    plt.title('The Silhouette score method showing the optimal k', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    number_of_cluters = 3

    kmeans = KMeans(n_clusters=number_of_cluters)
    clusters = kmeans.fit_predict(vector_representation)
    clusters_2 = {}

    # Display text for each cluster
    for cluster_id in range(number_of_cluters):
        cluster_texts = [pre_processed_stories[i] for i in range(len(pre_processed_stories)) if clusters[i] == cluster_id]
        print(f"Cluster {cluster_id + 1} Texts:")
        for text in cluster_texts:
            print(text, "\n")
        clusters_2[cluster_id] = []
        clusters_2[cluster_id].extend(cluster_texts)
        print()

    # Get cluster centroids (representative points)
    cluster_centroids = kmeans.cluster_centers_

    # Print cluster definitions based on centroid words
    for cluster_id in range(number_of_cluters):
        centroid_indices = cluster_centroids[cluster_id].argsort()[::-1]
        top_words = [vectorizer.get_feature_names_out()[index] for index in centroid_indices[:5]]
        print(f"Cluster {cluster_id + 1} Definition: {' '.join(top_words)}")

    print("--- %s seconds ---" % (time.time() - start_time))

    # Generate word clouds for each cluster
    stop_words = set(stopwords.words('english'))

    plt.figure(figsize=(12, 12))
    for cluster_id, cluster_docs in clusters_2.items():
        cluster_text = ' '.join(cluster_docs)
        wordcloud = WordCloud(width=200, height=200, background_color='white', stopwords=stop_words).generate(
            cluster_text)

        plt.subplot(1, number_of_cluters, cluster_id + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Cluster {cluster_id + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()