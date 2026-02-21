library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read.table("../data/berry_berry1990.txt", header = FALSE)
colnames(data) <- c("state", "year", "adopt", "fiscal_1", "party", "elect1", "elect2", "income_1", "neighbor", "nbrpercn", "religion")
data <- data[data$party != 9, ]

data_subset <- data %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adopt ~ fiscal_1 + party + elect1 + elect2 + income_1 + nbrpercn + religion, family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <-  glm(data = data_subset, adopt ~ fiscal_1*half + party*half + elect1*half + elect2*half + income_1*half + nbrpercn*half + religion*half, family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/berry_berry1990/berry_lacombe_lrtest_results.txt")