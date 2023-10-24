from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_vectorize(stories):
    # Formulate the feature vector out of given stories
    vectorizer = TfidfVectorizer()
    vectorized_stories = vectorizer.fit_transform(stories)

    return vectorizer, vectorized_stories