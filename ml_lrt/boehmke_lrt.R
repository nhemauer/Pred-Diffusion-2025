library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../data/boehmke2017.dta")

data_subset <- data %>% select(state, year, adopt, srcs_decay, nbrs_lag, rpcpinc, totpop, legp_squire, citi6010, unif_rep, unif_dem, time, time_sq, time_cube) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adopt ~ srcs_decay + nbrs_lag + rpcpinc + totpop + legp_squire + citi6010 + 
                          unif_rep + unif_dem + time + time_sq + time_cube + factor(state), family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adopt ~ srcs_decay*half + nbrs_lag*half + rpcpinc*half + totpop*half + legp_squire*half + citi6010*half + 
                          unif_rep*half + unif_dem*half + time*half + time_sq*half + time_cube*half + factor(state), family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/boehmke2017/boehmke_lrtest_results.txt")