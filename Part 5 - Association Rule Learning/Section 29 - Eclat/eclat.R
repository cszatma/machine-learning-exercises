# Eclat

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
# Support is items purchased at least 3 times a day - ie 4*7/7500
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])
