import matplotlib.pyplot as plt
import numpy as np
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from Util import get_iris_data, plot_decision_regions

if __name__ == "__main__":
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    X_train, X_test, y_train, y_test = get_iris_data()
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()

    dot_data = export_graphviz(tree, filled=True, rounded=True,
                               class_names=['Setosa', 'Versicolor', 'Virginica'],
                               feature_names=['petal length', 'petal width'], out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('tree.png')
