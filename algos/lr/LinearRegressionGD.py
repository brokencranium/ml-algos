import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shaper[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:] + self.w_[0])

    def predict(self, X):
        return self.net_input(X)


if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/' +
                     'python-machine-learning-book-2nd-edition' +
                     '/master/code/ch10/housing.data.txt',
                     header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                  'PTRATIO', 'B', 'LSTAT', 'MEDV']

    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols], height=2.5)
    plt.tight_layout()
    plt.show()

    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
                     yticklabels=cols, xticklabels=cols)
    plt.show()

    X = df[['RM']].values
    y = df['MEDV'].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    # Many transformers in sci-kit require a two dimensional array
    # Changing y to a 2d array and flatten it.
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    sns.reset_orig()  # reset mat plot lib style
    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000s [MEDV] (standardized)')
    plt.show()

    num_rooms_std = sc_x.transform([5.0])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000s: %.3f", sc_y.inverse_transform(price_std))

    print('Slope: %.3f' % lr.w_[1])
    # Intercept is zero for standardized variables
    print('Intercept: %.3f' % lr.w_[0])
