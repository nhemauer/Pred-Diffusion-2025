library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../../data/kreitzer_boehmke2016.dta")

data_subset <- data %>% select(state, year, policy_num, adopt_policy, norrander_legality, religadhrate, initdif, dem_gov, 
                               uni_dem_leg, fem_dem, nbrspct, rescaledmedincome, rescaledpopsize, time, time2, webster) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adopt_policy ~ norrander_legality + religadhrate + initdif + dem_gov + uni_dem_leg + fem_dem + 
                          nbrspct + rescaledmedincome + rescaledpopsize + time + time2 + webster + factor(policy_num), family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adopt_policy ~ norrander_legality*half + religadhrate*half + initdif*half + dem_gov*half + uni_dem_leg*half + fem_dem*half + 
                          nbrspct*half + rescaledmedincome*half + rescaledpopsize*half + time*half + time2*half + webster*half + factor(policy_num),
                          family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/kreitzer_boehmke2016/kreitzer_boehmke_lrtest_results.txt")