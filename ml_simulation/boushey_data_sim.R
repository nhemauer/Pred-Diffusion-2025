set.seed(1337)

if (!requireNamespace("neha", quietly = TRUE)){
  if (!requireNamespace("devtools", quietly = TRUE)){
    install.packages("devtools")
  }
  devtools::install_github("desmarais-lab/neha")
}

library(neha)
library(tidyverse)
library(haven)

boushey2016_full <- read_dta("data/boushey2016.dta")

# Covariates
covariates <- c("policycongruent","gub_election","elect2", "hvd_4yr", "fedcrime",
                "leg_dem_per_2pty","dem_governor","insession","propneighpol",
                "citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq",
                "violentthousand","pctwhite","stateincpercap","logpop",
                "counter","counter2","counter3")

# Subset and drop missing
boushey2016 <- boushey2016_full %>%
  select(state, year, billnum, dvadopt, all_of(covariates)) %>%
  na.omit()

# Define formula
formula <- as.formula(
  paste("dvadopt ~", paste(covariates, collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = boushey2016, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Drop intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])

sim_results <- data.frame()

for (bill in unique(boushey2016$billnum)){
  # Filter data per bill
  policy_data <- boushey2016 %>% filter(billnum == bill)
  policy_data <- as.data.frame(policy_data)
  
  # Select relevant data
  policy_data <- policy_data %>% select(state, year, all_of(covariates))

  oldest_year <- min(policy_data$year)
  newest_year <- max(policy_data$year)

  # Create complete panel data with all states and all years
  all_states <- unique(policy_data$state)
  all_years <- oldest_year:newest_year
  
  complete_panel <- expand.grid(
    state = all_states,
    year = all_years,
    stringsAsFactors = FALSE
  )
  
  # Merge with existing data
  policy_data_complete <- complete_panel %>%
    left_join(policy_data, by = c("state", "year"))

  # Fill missing covariate values by randomly sampling from each state's available data
  policy_data_complete <- policy_data_complete %>%
    group_by(state) %>%
    mutate(across(all_of(covariates), ~ {
      available_values <- .x[!is.na(.x)]
      ifelse(is.na(.x), 
            sample(available_values, length(.x), replace = TRUE)[is.na(.x)], 
            .x)
    })) %>%
    ungroup()
  
  policy_data_complete <- as.data.frame(policy_data_complete)
  
  # Create fake gamma
  tie_names <- c("california_montana", "minnesota_wisconsin", "iowa_minnesota")
  tie_values <- c(0.8, 0.5, 0.3)
  gamma <- matrix(tie_values, ncol = 1, dimnames = list(tie_names, "value"))
  
  sim_data <- simulate_neha_discrete(policy_data_complete, node = "state", time = "year", beta = coef_matrix, gamma = gamma, a = 0)
  
  # Add bill name column
  sim_data$billnum <- bill
  
  # Aggregate results
  sim_results <- bind_rows(sim_results, sim_data)
  
}

write.csv(sim_results, "ml_simulation/figures/boushey2016/boushey_sim_data.csv", row.names = FALSE)