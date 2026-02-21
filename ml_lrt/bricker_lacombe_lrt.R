library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../data/bricker_lacombe2021.dta")

data_subset <- data %>% select(state, year, adoption, std_score, initiative, init_sigs, std_population, std_citideology, unified, std_income, std_legp_squire, duration, durationsq, durationcb) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adoption ~ std_score + initiative + init_sigs + std_population + std_citideology + unified + std_income + std_legp_squire + 
                          duration + durationsq + durationcb + factor(year), family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adoption ~ std_score*half + initiative*half + init_sigs*half + std_population*half + std_citideology*half + unified*half + std_income*half + std_legp_squire*half + 
                          duration*half + durationsq*half + durationcb*half + factor(year), family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/bricker_lacombe2021/bricker_lacombe_lrtest_results.txt")