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
library(fastDummies)

bricker_lacombe_2021_full <- read_dta("data/bricker_lacombe2021.dta")

# Covariates
covariates = c("std_score","initiative","init_sigs","std_population",
                "std_citideology","unified","std_income","std_legp_squire",
                "duration","durationsq","durationcb")

# Subset and drop missing
bricker_lacombe_2021 <- bricker_lacombe_2021_full %>%
  select(state, year, policy, adoption, all_of(covariates)) %>%
  fastDummies::dummy_cols(
    select_columns = "year",
    remove_first_dummy = TRUE,
    remove_selected_columns = FALSE
  ) %>%
  na.omit()

year_dummies <- grep("^year_", names(bricker_lacombe_2021), value = TRUE)

# Define formula
formula <- as.formula(
  paste("adoption ~", paste(c(covariates, year_dummies), collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = bricker_lacombe_2021, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Fix intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])
intercept <- coef(logistic)[1]
coef_matrix <- rbind(coef_matrix, intercept)

sim_results <- data.frame()

for (bill in unique(bricker_lacombe_2021$policy)){
  # Filter data per bill
  policy_data <- bricker_lacombe_2021 %>% filter(policy == bill)
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

  # Fill missing covariate values by randomly sampling from each state's available data
  policy_data_complete <- complete_panel %>%
    group_by(state) %>%
    group_modify(~ {
      df_panel <- .x     

      # Observed rows for state
      donors <- policy_data %>% filter(state == .y$state)

      # For each covariate, sample independently
      for (cv in covariates) {
        df_panel[[cv]] <- sample(donors[[cv]], size = nrow(df_panel), replace = TRUE)
      }

      df_panel
    }) %>%
    ungroup()

  # Recreate dummies
  policy_data_complete <- policy_data_complete %>%   
    fastDummies::dummy_cols(
    select_columns = "year",
    remove_first_dummy = TRUE,
    remove_selected_columns = FALSE
  )
  
  policy_data_complete <- as.data.frame(policy_data_complete)

  # Add intercept
  policy_data_complete$intercept = 1

  beta_names <- intersect(rownames(coef_matrix), names(policy_data_complete))
  beta_sim <- coef_matrix[beta_names, , drop = FALSE]

  # Create fake gamma
  tie_names <- c("california_montana", "minnesota_wisconsin", "iowa_minnesota")
  tie_values <- c(0, 0, 0)
  gamma <- matrix(tie_values, ncol = 1, dimnames = list(tie_names, "value"))
  
  sim_data <- simulate_neha_discrete(policy_data_complete, node = "state", time = "year", beta = beta_sim, gamma = gamma, a = 0)
  
  # Add bill name column
  sim_data$billnum <- bill
  
  # Aggregate results
  sim_results <- bind_rows(sim_results, sim_data)
  
}

# Will show missing values for dummies if those years arn't in the unique policy data, this fixes that
sim_results <- sim_results %>%
  mutate(across(where(is.numeric), ~ replace_na(.x, 0)))

write.csv(sim_results, "ml_simulation/figures/bricker_lacombe2021/bricker_lacombe_sim_data.csv", row.names = FALSE)