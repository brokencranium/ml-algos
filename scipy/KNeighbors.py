import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Util import plot_decision_regions, get_iris_data

if __name__ == "__main__":
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    X_train_std, X_test_std, y_train, y_test = get_iris_data()

    knn.fit(X_train_std, y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')cd ,,
    plt.legend(loc='upper left')
    plt.show()
