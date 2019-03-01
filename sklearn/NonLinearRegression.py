import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return


if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/' +
                     'python-machine-learning-book-2nd-edition' +
                     '/master/code/ch10/housing.data.txt',
                     header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                  'PTRATIO', 'B', 'LSTAT', 'MEDV']

    X = df[['LSTAT']].values
    y = df['MEDV'].values

    regr = LinearRegression()

    # create quadratic features
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # fit features
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))

    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))

    # plot results
    plt.scatter(X, y, label='training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2,
             linestyle=':')

    plt.plot(X_fit, y_quad_fit,
             label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
             color='red',
             lw=2,
             linestyle='-')

    plt.plot(X_fit, y_cubic_fit,
             label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
             color='green',
             lw=2,
             linestyle='--')

    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.legend(loc='upper right')
    plt.show()

    # transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)

    # fit features
    X_fit = np.arange(X_log.min() - 1,
                      X_log.max() + 1, 1)[:, np.newaxis]
    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

    # plot results
    plt.scatter(X_log, y_sqrt,
                label='training points',
                color='lightgray')
    plt.plot(X_fit, y_lin_fit,
             label='linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2)
    plt.xlabel('log(% lower status of the population [LSTAT])')
    plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
    plt.legend(loc='lower left')
    plt.show()

    X = df[['LSTAT']].values
    y = df['MEDV'].values
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)
    sort_idx = X.flatten().argsort()
    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.show()

    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.4,
                         random_state=1)

    forest = RandomForestRegressor(n_estimators=1000,
                                   criterion='mse',
                                   random_state=1,
                                   n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', edgecolor='white', marker='o',
                s=35, alpha=0.9, label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', edgecolor='white', marker='s',
                s=35, alpha=0.9, label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.show()

# if __name__ == "__main__":
#     X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:,
#         np.newaxis]
#     y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
#     lr = LinearRegression()
#     pr = LinearRegression()
#     quadratic = PolynomialFeatures(degree=2)
#     X_quad = quadratic.fit_transform(X)
#
#     lr.fit(X, y)
#     X_fit = np.arange(250, 600, 10)[:, np.newaxis]
#     y_lin_fit = lr.predict(X_fit)
#
#     pr.fit(X_quad, y)
#     y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
#
#     plt.scatter(X, y, label='training points')
#     plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
#     plt.plot(X_fit, y_quad_fit, label='quadratic fit')
#     plt.legend(loc='upper left')
#     plt.show()
#
#     y_lin_pred = lr.predict(X)
#     y_quad_pred = pr.predict(X_quad)
#     print('Training MSE linear: %.3f, quadratic: %.3f'
#           % (mean_squared_error(y, y_lin_pred), mean_squared_error(y, y_quad_pred)))
#     print('Training  R^2 linear: %.3f, quadratic: %.3f'
#           % (r2_score(y, y_lin_pred), r2_score(y, y_quad_pred)))
