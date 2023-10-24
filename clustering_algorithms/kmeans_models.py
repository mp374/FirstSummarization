from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


def kmeans_clusters(vectorized_data, k):
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(vectorized_data)

    return kmeanModel