library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../data/boushey2016.dta")

data_subset <- data %>% select(state, year, dvadopt, policycongruent, gub_election, elect2, hvd_4yr, fedcrime, leg_dem_per_2pty, dem_governor,
                                      insession, propneighpol, citidist, squire_prof86, citi6008, crimespendpc, crimespendpcsq, violentthousand,
                                      pctwhite, stateincpercap, logpop, counter, counter2, counter3) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, dvadopt ~ policycongruent + gub_election + elect2 + hvd_4yr + fedcrime + leg_dem_per_2pty + dem_governor + insession + propneighpol + citidist + squire_prof86 +
                   citi6008 + crimespendpc + crimespendpcsq + violentthousand + pctwhite + stateincpercap + logpop + counter + counter2 + counter3, family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, dvadopt ~ policycongruent*half + gub_election*half + elect2*half + hvd_4yr*half + fedcrime*half + leg_dem_per_2pty*half + dem_governor*half + insession*half +
                   propneighpol*half + citidist*half + squire_prof86*half + citi6008*half + crimespendpc*half + crimespendpcsq*half + violentthousand*half + pctwhite*half + stateincpercap*half + logpop*half
                   + counter*half + counter2*half + counter3*half, family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/boushey2016/boushey_lrtest_results.txt")