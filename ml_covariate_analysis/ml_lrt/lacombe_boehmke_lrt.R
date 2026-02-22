library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../../data/lacombe_boehmke2021.dta")

data_subset <- data %>% select(state, year, adoption, initiative, init_sigs, std_latnt_decay, std_nbrs_lag, std_population,
                               std_masssociallib_est, unified, duration, durationsq, durationcb, std_income,
                               std_bowen_1, std_bowen_2, change_pop, change_inc, party_change) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adoption ~ initiative + init_sigs + std_latnt_decay + std_nbrs_lag + std_population +
                          std_masssociallib_est + unified + duration + durationsq + durationcb + std_income +
                          std_bowen_1 + std_bowen_2 + change_pop + change_inc + party_change + factor(year),
                          family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adoption ~ initiative*half + init_sigs*half + std_latnt_decay*half + std_nbrs_lag*half + std_population*half +
                          std_masssociallib_est*half + unified*half + duration*half + durationsq*half + durationcb*half + std_income*half +
                          std_bowen_1*half + std_bowen_2*half + change_pop*half + change_inc*half + party_change*half + factor(year),
                          family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/lacombe_boehmke2021/lacombe_boehmke_lrtest_results.txt")