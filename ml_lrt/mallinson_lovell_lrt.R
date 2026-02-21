library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read.csv("../data/mallinson_lovell2022.csv")

data_subset <- data %>% select(state, year, adopt, republican, legprof_squire, exp_pupil10000_adj, mathscore4th, readscore4th, time) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adopt ~ republican + legprof_squire + exp_pupil10000_adj + mathscore4th + readscore4th + time,
                          family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adopt ~ republican*half + legprof_squire*half + exp_pupil10000_adj*half + mathscore4th*half + readscore4th*half + time*half,
                          family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/mallinson_lovell2022/mallinson_lovell_lrtest_results.txt")