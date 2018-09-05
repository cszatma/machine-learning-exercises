# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
# Avoid the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting XGBoost to the Training Set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
