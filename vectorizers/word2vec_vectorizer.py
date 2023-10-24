import os
import random
import numpy as np
from gensim.models import Word2Vec
import pickle

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)


def word2vec_vectorizer(tokenized_stories):

    # file name of the model (the models will be saved by this name if not exists.
    model_file_name = "tf-idf_model_with.obj"
    if os.path.exists(model_file_name):
        fileObj = open(model_file_name, 'rb')
        vectorizer = pickle.load(fileObj)
        print("loading the model....")
    else:
        # trains the model using given set of tokenized stories.
        vectorizer = Word2Vec(tokenized_stories, min_count=1)
        fileObj = open(model_file_name, 'wb')
        pickle.dump(vectorizer, fileObj)
        print("saving the model....")

    # Convert stories to Word2Vec vectors
    story_vectors = [np.mean([vectorizer.wv[word] for word in story_tokens if word in vectorizer.wv], axis=0)
                     for story_tokens in tokenized_stories]
    vectorized_stories = np.array(story_vectors)

    return vectorizer, vectorized_stories
