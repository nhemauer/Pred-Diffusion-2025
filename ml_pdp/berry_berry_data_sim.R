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
berry_berry1990_full <- read.table("../data/berry_berry1990.txt", header = FALSE, sep = "")
colnames(berry_berry1990_full) <- c("state", "year", "adopt", "fiscal_1", "party", "elect1", "elect2", "income_1", "neighbor", "nbrpercn", "religion")
covariates <- c("fiscal_1", "party", "elect1", "elect2", "income_1", "nbrpercn", "religion")

berry_berry1990 <- berry_berry1990_full[berry_berry1990_full$party != 9, ] # 9 is the NA (For MN and NE)

# Subset and drop missing
berry_berry1990 <- berry_berry1990_full %>%
  select(state, year, adopt, all_of(covariates)) %>%
  na.omit()

# Define formula
formula <- as.formula(
  paste("adopt ~", paste(covariates, collapse = " + "))
)

# Fit logistic regression model
logistic <- glm(formula, data = berry_berry1990, family = binomial(link = "logit"))

# Extract coefficients
coef_vec <- coef(logistic)

# Fix intercept
coef_matrix <- as.matrix(coef_vec[-c(1), drop = FALSE])
intercept <- coef(logistic)[1]
coef_matrix <- rbind(coef_matrix, intercept)

# Filter data per bill
policy_data <- berry_berry1990
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

policy_data_complete <- as.data.frame(policy_data_complete)

# Add intercept
policy_data_complete$intercept = 1

# Create fake gamma
tie_names <- c("california_montana", "minnesota_wisconsin", "iowa_minnesota")
tie_values <- c(0, 0, 0)
gamma <- matrix(tie_values, ncol = 1, dimnames = list(tie_names, "value"))

sim_data <- simulate_neha_discrete(policy_data_complete, node = "state", time = "year", beta = coef_matrix, gamma = gamma, a = 0)

write.csv(sim_data, "ml_simulation/figures/berry_berry1990/berry_berry_sim_data.csv", row.names = FALSE)