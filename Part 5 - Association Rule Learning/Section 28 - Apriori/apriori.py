# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []

for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])

# Training Apriori on the dataset
# Support is items purchased at least 3 times a day - ie 3*7/7500
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
