# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
nltk.download('stopwords')


def clean_review(review):
    # Remove non alpha chars, convert charts to lowercase and split the string into a list
    review_list = re.sub('[^a-zA-Z]', ' ', review).lower().split()
    # Remove unnecessary words
    # Stemming - Only keep roots of words ex: loved -> love
    ps = PorterStemmer()
    stemmed_review = [ps.stem(word) for word in review_list if word not in set(stopwords.words('english'))]
    # Join words back into a single string
    return ' '.join(stemmed_review)


corpus = [clean_review(review) for review in dataset['Review']]
# corpus = []
#
# for i in range(0, dataset.shape[0]):
#     # Remove non alpha chars, convert charts to lowercase and split the string into a list
#     review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
#     # Remove unnecessary words
#     # Stemming - Only keep roots of words ex: loved -> love
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
#     # Join words back into a single string
#     review = ' '.join(review)
#     corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
