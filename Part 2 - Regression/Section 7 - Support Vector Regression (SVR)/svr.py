# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Don't split into a training and test set since we do not have enough data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))
y = y.reshape(-1)

# Fitting the SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)


def plot_results(x, y, predition, title, plot_x=None):
    plt.scatter(x, y, color='red')
    plt.plot(x if plot_x is None else plot_x, predition, color='blue')
    plt.title(title)
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()


# Visualising the SVR results
# Create matrix of values incremented by 0.1 for a smoother curve
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
plot_results(X, y, regressor.predict(X),
             "Truth or Bluff (SVR)")

# Predicting a new result with Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
