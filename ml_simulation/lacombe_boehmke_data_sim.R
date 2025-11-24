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

lacombe_boehmke2021_full <- read_dta("data/lacombe_boehmke2021.dta")

# Covariates
covariates = c("initiative", "init_sigs", "std_latnt_decay", "std_nbrs_lag", "std_population",
    "std_masssociallib_est", "unified", "duration", "durationsq", "durationcb", "std_income",
    "std_bowen_1", "std_bowen_2", "change_pop", "change_inc", "party_change")

# Subset and drop missing
lacombe_boehmke2021 <- lacombe_boehmke2021_full %>%
  select(state, year, policyno, adoption, all_of(covariates)) %>%
    fastDummies::dummy_cols(
    select_columns = "year",
    remove_first_dummy = TRUE,
    remove_selected_columns = FALSE
  ) %>%
  na.omit()

year_dummies <- grep("^year_", names(lacombe_boehmke2021), value = TRUE)

# Define formula
formula <- as.formula(
  paste("adoption ~", paste(c(covariates, year_dummies), collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = lacombe_boehmke2021, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Fix intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])
intercept <- coef(logistic)[1]
coef_matrix <- rbind(coef_matrix, intercept)

# Convert policies
policy_lookup <- data.frame(
  original_name = unique(lacombe_boehmke2021$policyno),
  policy_number = seq_along(unique(lacombe_boehmke2021$policyno))
)
lacombe_boehmke2021$policyno <- as.numeric(as.factor(lacombe_boehmke2021$policyno))

sim_results <- data.frame()

for (bill in unique(lacombe_boehmke2021$policyno)){
  # Filter data per bill
  original_policy_name <- policy_lookup$original_name[policy_lookup$policy_number == bill]
  print(paste("Bill", bill, ":", original_policy_name))
  policy_data <- lacombe_boehmke2021 %>% filter(policyno == bill)
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

  # Remove any duplicates
  policy_data_complete <- policy_data_complete %>%
    group_by(state, year) %>%
    slice_sample(n = 1) %>%
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
  policy_data_complete$year <- policy_data_complete$year - 1

  beta_names <- intersect(rownames(coef_matrix), names(policy_data_complete))
  beta_sim <- coef_matrix[beta_names, , drop = FALSE]
  print(policy_data_complete)

  # Create fake gamma
  tie_names <- c("california_montana", "minnesota_wisconsin", "iowa_minnesota")
  tie_values <- c(0.8, 0.5, 0.3)
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

write.csv(sim_results, "ml_simulation/figures/lacombe_boehmke2021/lacombe_boehmke_sim_data.csv", row.names = FALSE)