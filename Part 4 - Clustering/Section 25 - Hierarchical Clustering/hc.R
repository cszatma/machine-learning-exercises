# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Splitting the dataset into the Trainging set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')


# Applying Hierarchical Clustering to the dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
