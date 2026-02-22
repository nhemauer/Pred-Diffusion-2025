library(haven)
library(tidyverse)
library(lmtest)

# Load data
data <- read_dta("../../data/karch2016.dta")

data_subset <- data %>% select(state, year, adopt, traditional, nborsstd, prevadoptstd, complexity, igrole,
                               regov, unified, perdemstd, incpcadjstd, exppcadjstd,
                               logpopstd, collegstd, perurbanstd, profstd,
                               traditional_nborsstd, traditional_prevadoptstd, traditional_complexity,
                               traditional_igrole, traditional_regov, traditional_unified,
                               traditional_perdemstd, traditional_incpcadjstd, traditional_exppcadjstd,
                               traditional_logpopstd, traditional_collegstd, traditional_perurbanstd, traditional_profstd) %>% na.omit()

# Find year midpoint
years <- sort(unique(data_subset$year))

mid_year <- years[ceiling(length(years) / 2)]

# Create indicator variable for first/last half of data
data_subset$half <- ifelse(data_subset$year >= mid_year, 1, 0)

# Create model from article
model_small <- glm(data = data_subset, adopt ~ traditional + nborsstd + prevadoptstd + complexity + igrole +
                          regov + unified + perdemstd + incpcadjstd + exppcadjstd +
                          logpopstd + collegstd + perurbanstd + profstd +
                          traditional_nborsstd + traditional_prevadoptstd + traditional_complexity +
                          traditional_igrole + traditional_regov + traditional_unified +
                          traditional_perdemstd + traditional_incpcadjstd + traditional_exppcadjstd +
                          traditional_logpopstd + traditional_collegstd + traditional_perurbanstd + traditional_profstd,
                          family = binomial(link = "logit"))

# Create model from article with interaction terms
model_large <- glm(data = data_subset, adopt ~ traditional*half + nborsstd*half + prevadoptstd*half + complexity*half + igrole*half +
                          regov*half + unified*half + perdemstd*half + incpcadjstd*half + exppcadjstd*half +
                          logpopstd*half + collegstd*half + perurbanstd*half + profstd*half +
                          traditional_nborsstd*half + traditional_prevadoptstd*half + traditional_complexity*half +
                          traditional_igrole*half + traditional_regov*half + traditional_unified*half +
                          traditional_perdemstd*half + traditional_incpcadjstd*half + traditional_exppcadjstd*half +
                          traditional_logpopstd*half + traditional_collegstd*half + traditional_perurbanstd*half + traditional_profstd*half,
                          family = binomial(link = "logit"))

# Save output of likelihood ratio test
capture.output(lrtest(model_small, model_large), file = "figures/karch2016/karch_lrtest_results.txt")