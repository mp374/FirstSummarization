import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample feedback from users about a service they received
from data_set_managers.merge_dataset import read_file_content
from pre_processing.pre_processing import preprocess, pre_process_tokenize
import time

story_file_path = '/Users/heshankavinda/Library/CloudStorage/OneDrive-UniversityofPlymouth/PROJ518/Project/First_Set_of_Algo/data_set/test_stories_2.json'
stories = read_file_content(story_file_path)

# Second argument should either be "Stem" or "Lemma" or "None"
pre_processed_stories = pre_process_tokenize(stories, "Lemma")

start_time = time.time()

# Create a dictionary and document-term matrix
dictionary = corpora.Dictionary(pre_processed_stories)
corpus = [dictionary.doc2bow(text) for text in pre_processed_stories]

# Perform LDA
num_topics = 3  # Number of topics to discover
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=5)

# Display the topics and associated keywords
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx + 1}: {topic}")

# Assign the feedback to their respective topics
topic_assignments = [max(lda_model[text], key=lambda x: x[1])[0] for text in corpus]

# Organize feedback into topic clusters
clusters = {}
for i, topic_id in enumerate(topic_assignments):
    if topic_id not in clusters:
        clusters[topic_id] = []
    clusters[topic_id].append(stories[i])

# Display the clustered story
for cluster_id, story_in_cluster in clusters.items():
    print(f"\nCluster {cluster_id + 1}:")
    for story in story_in_cluster:
        print(story, "\n")

print("--- %s seconds ---" % (time.time() - start_time))

# Generate word clouds for each cluster
stop_words = set(stopwords.words('english'))

plt.figure(figsize=(12, 6))
for cluster_id, story_in_cluster in clusters.items():
    cluster_text = ' '.join(story_in_cluster)
    wordcloud = WordCloud(width=200, height=200, background_color='white', stopwords=stop_words).generate(cluster_text)

    plt.subplot(1, num_topics, cluster_id + 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Cluster {cluster_id + 1}")
    plt.axis("off")

plt.tight_layout()
plt.show()