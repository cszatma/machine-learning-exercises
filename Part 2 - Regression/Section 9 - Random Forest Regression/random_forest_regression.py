# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

# Fitting the Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)


def plot_results(x, y, predition, title, plot_x=None):
    plt.scatter(x, y, color='red')
    plt.plot(x if plot_x is None else plot_x, predition, color='blue')
    plt.title(title)
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()


# Visualising the Random Forest Regression results
# Create matrix of values incremented by 0.1 for a smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plot_results(X, y, regressor.predict(X_grid),
             "Truth or Bluff (Random Forest Regression)", plot_x=X_grid)

# Predicting a new result
y_pred = regressor.predict(6.5)
