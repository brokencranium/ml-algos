import numpy as np
from sklearn.cluster import AgglomerativeClustering

if __name__ == "__main__":
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5, 3]) * 10
    ac = AgglomerativeClustering(n_clusters=3,
                                 affinity='euclidean',
                                 linkage='complete')
    labels = ac.fit_predict(X)
    print('Cluster labels: %s' % labels)

    ac = AgglomerativeClustering(n_clusters=2,
                                 affinity='euclidean',
                                 linkage='complete')
    labels = ac.fit_predict(X)
    print('Cluster labels: %s' % labels)
