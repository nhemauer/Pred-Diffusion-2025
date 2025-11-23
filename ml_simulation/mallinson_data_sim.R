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

# Data
mallinson2019_full <- read_csv("data/mallinson2019.csv")

# Covariates
covariates = c("neighbor_prop", "ideology_relative_hm", "congress_majortopic", "init_avail", "init_qual", "divided_gov",
              "legprof_squire", "percap_log", "population_log", "mip", "complexity_topic", "mip_complexity_topic", "nyt", "year_count", "time_log")

# Subset and drop missing
mallinson2019 <- mallinson2019_full %>%
  select(state, year, policy, adopt, all_of(covariates)) %>%
  na.omit()

# Define formula
formula <- as.formula(
  paste("adopt ~", paste(covariates, collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = mallinson2019, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Fix intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])
intercept <- coef(logistic)[1]
coef_matrix <- rbind(coef_matrix, intercept)

sim_results <- data.frame()

for (bill in unique(mallinson2019$policy)){
  # Filter data per bill
  policy_data <- mallinson2019 %>% filter(policy == bill)
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

  # Add intercept
  policy_data_complete$intercept = 1
  policy_data_complete$year <- policy_data_complete$year - 1
  
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

write.csv(sim_results, "ml_simulation/figures/mallinson2019/mallinson_sim_data.csv", row.names = FALSE)