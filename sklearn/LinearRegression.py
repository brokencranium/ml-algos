import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/' +
                     'python-machine-learning-book-2nd-edition' +
                     '/master/code/ch10/housing.data.txt',
                     header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                  'PTRATIO', 'B', 'LSTAT', 'MEDV']

    X = df[['RM']].values
    y = df['MEDV'].values

    # RANdom SAmple Consensus (RANSAC)
    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                             loss='absolute_loss', residual_threshold=5.0, random_state=0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o',
                label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s',
                label='Outliers')
    plt.plot(line_X, line_y_ransac, color='black', lw=2)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.legend(loc='upper left')
    plt.show()

    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)
