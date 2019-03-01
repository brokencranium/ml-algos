import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Util import get_iris_data, plot_decision_regions

if __name__ == "__main__":
    forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
    X_train, X_test, y_train, y_test = get_iris_data()
    forest.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.show()
