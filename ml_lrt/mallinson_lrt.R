library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read.csv("../data/mallinson2019.csv")

data_subset <- data %>% select(state, year, adopt, neighbor_prop, ideology_relative_hm, congress_majortopic,
                               init_avail, init_qual, divided_gov, legprof_squire,
                               percap_log, population_log, mip, complexity_topic,
                               mip_complexity_topic, nyt, year_count, time_log) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adopt ~ neighbor_prop + ideology_relative_hm + congress_majortopic +
                          init_avail + init_qual + divided_gov + legprof_squire +
                          percap_log + population_log + mip + complexity_topic +
                          mip_complexity_topic + nyt + year_count + time_log,
                          family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adopt ~ neighbor_prop*half + ideology_relative_hm*half + congress_majortopic*half +
                          init_avail*half + init_qual*half + divided_gov*half + legprof_squire*half +
                          percap_log*half + population_log*half + mip*half + complexity_topic*half +
                          mip_complexity_topic*half + nyt*half + year_count*half + time_log*half,
                          family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/mallinson2019/mallinson_lrtest_results.txt")