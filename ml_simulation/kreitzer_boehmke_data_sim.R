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

kreitzer_boehmke2016_full <- read_dta("data/kreitzer_boehmke2016.dta")

# Covariates
covariates = c("norrander_legality", "religadhrate", "initdif", "dem_gov", "uni_dem_leg",
    "fem_dem", "nbrspct", "rescaledmedincome", "rescaledpopsize", "time", 
    "time2", "webster")

# Subset and drop missing
kreitzer_boehmke2016 <- kreitzer_boehmke2016_full %>%
  select(state, year, policy_num, adopt_policy, all_of(covariates)) %>%
  fastDummies::dummy_cols(
    select_columns = "policy_num",
    remove_first_dummy = TRUE,
    remove_selected_columns = FALSE
  ) %>%
  na.omit()

# edit_data <- kreitzer_boehmke2016 %>% arrange(policy_num, year, state)

policy_dummies <- grep("^policy_num_", names(kreitzer_boehmke2016), value = TRUE)

# Define formula
formula <- as.formula(
  paste("adopt_policy ~", paste(c(covariates, policy_dummies), collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = kreitzer_boehmke2016, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Fix intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])
intercept <- coef(logistic)[1]
coef_matrix <- rbind(coef_matrix, intercept)

sim_results <- data.frame()

for (bill in unique(kreitzer_boehmke2016$policy_num)){
  # Filter data per bill
  print(bill)
  policy_data <- kreitzer_boehmke2016 %>% filter(policy_num == bill)
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
  # policy_data_complete <- policy_data_complete %>%
  #   group_by(state, year) %>%
  #   slice_sample(n = 1) %>%
  #   ungroup()
  
  policy_data_complete <- as.data.frame(policy_data_complete)

  # Add intercept
  policy_data_complete$intercept = 1

  beta_names <- intersect(rownames(coef_matrix), names(policy_data_complete))
  beta_sim <- coef_matrix[beta_names, , drop = FALSE]

  # Add policy-specific coefficient to intercept (a different way to dummy out the policy column)
  policy_coef_name <- paste0("policy_num_", bill)
  if (policy_coef_name %in% names(coef_vec)) {
    policy_intercept <- intercept + coef_vec[policy_coef_name]
    beta_sim["intercept", 1] <- policy_intercept
  }

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

write.csv(sim_results, "ml_simulation/figures/kreitzer_boehmke2016/kreitzer_boehmke_sim_data.csv", row.names = FALSE)