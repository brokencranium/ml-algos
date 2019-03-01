import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/' +
                     'python-machine-learning-book-2nd-edition' +
                     '/master/code/ch10/housing.data.txt',
                     header=None, sep='\s+')

    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                  'PTRATIO', 'B', 'LSTAT', 'MEDV']

    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()

    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                           mean_squared_error(y_test, y_test_pred)))

    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                                           r2_score(y_test, y_test_pred)))

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0)
    elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
