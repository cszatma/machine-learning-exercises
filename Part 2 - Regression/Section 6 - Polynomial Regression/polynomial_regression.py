# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Don't split into a training and test set since we do not have enough data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Regression to the dataset
# Transform the matrix of features X into a matrix of polynomial features
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)


def plot_results(x, y, predition, title, plotX=None):
    plt.scatter(x, y, color='red')
    plt.plot(x if plotX is None else plotX, predition, color='blue')
    plt.title(title)
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()


# Visualising the Linear Regression results
plot_results(X, y, linear_regressor.predict(X), "Truth or Bluff (Linear Regression)")

# Visualising the Polynomial Regression results
# Create matrix of values incremented by 0.1 for a smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plot_results(X, y, linear_regressor_2.predict(polynomial_regressor.fit_transform(X_grid)),
             "Truth or Bluff (Polynomial Regression)", X_grid)

# Predicting a new result with Linear Regression
linear_regressor.predict(6.5)

# Predicting a new result with Polynomial Regression
linear_regressor_2.predict(polynomial_regressor.fit_transform(6.5))
