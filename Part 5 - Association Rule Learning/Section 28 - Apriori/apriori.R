# Apriori

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training the Apriori on the dataset
# Support is items purchased at least 3 times a day - ie 4*7/7500
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.4))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
