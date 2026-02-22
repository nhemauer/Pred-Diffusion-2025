library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../../data/parinandi2020.dta")

data_subset <- data %>% select(state, year, oneemulation, adagovideology, citizenideology, medianivoteshare, partydecline, squirescore,
                               incunemp, pctpercapincome, percenturban, ugovd, percentfossilprod, renergyprice11, deregulated,
                               geoneighborlag, ideoneighborlag, premulation1, featureyear) %>% na.omit()

# Find year 70th percentile
years <- sort(unique(data_subset$year))

split_year <- years[ceiling(length(years) * 0.7)]

# Create indicator variable for first 70% / last 30% of data
data_subset$half <- ifelse(data_subset$year >= split_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, oneemulation ~ adagovideology + citizenideology + medianivoteshare + partydecline + squirescore +
                          incunemp + pctpercapincome + percenturban + ugovd + percentfossilprod + renergyprice11 + deregulated +
                          geoneighborlag + ideoneighborlag + premulation1 + featureyear,
                          family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, oneemulation ~ adagovideology*half + citizenideology*half + medianivoteshare*half + partydecline*half + squirescore*half +
                          incunemp*half + pctpercapincome*half + percenturban*half + ugovd*half + percentfossilprod*half + renergyprice11*half + deregulated*half +
                          geoneighborlag*half + ideoneighborlag*half + premulation1*half + featureyear*half,
                          family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/parinandi2020/parinandi_lrtest_results.txt")