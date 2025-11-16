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

boehmke2017_full <- read_dta("data/boehmke2017.dta")

# Covariates
covariates = c("srcs_decay","nbrs_lag","rpcpinc","totpop","legp_squire",
                "citi6010","unif_rep","unif_dem","time","time_sq","time_cube")

# Subset and drop missing
boehmke2017 <- boehmke2017_full %>%
  select(state, year, policy, adopt, all_of(covariates)) %>%
  fastDummies::dummy_cols(
    select_columns = "state",
    remove_first_dummy = TRUE,
    remove_selected_columns = FALSE
  ) %>%
  na.omit()

state_dummies <- grep("^state_", names(boehmke2017), value = TRUE)

# Define formula
formula <- as.formula(
  paste("adopt ~", paste(c(covariates, state_dummies), collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = boehmke2017, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Drop intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])

sim_results <- data.frame()

for (bill in unique(boehmke2017$policy)){
  # Filter data per bill
  policy_data <- boehmke2017 %>% filter(policy == bill)
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

  # Recreate dummies
  policy_data_complete <- policy_data_complete %>%   
    fastDummies::dummy_cols(
    select_columns = "state",
    remove_first_dummy = TRUE,
    remove_selected_columns = FALSE
  )
  
  policy_data_complete <- as.data.frame(policy_data_complete)

  beta_names <- intersect(rownames(coef_matrix), names(policy_data_complete))
  beta_sim <- coef_matrix[beta_names, , drop = FALSE]

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

# Will show missing values for dummies if those states arn't in the specific policy data, this fixes that
sim_results <- sim_results %>%
  mutate(across(where(is.numeric), ~ replace_na(.x, 0)))

write.csv(sim_results, "ml_simulation/figures/boehmke2017/boehmke_sim_data.csv", row.names = FALSE)