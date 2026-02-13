import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import random
import os

random.seed(1337)

### Define Survival Analysis Function

def discrete_survival_probabilities(hazards, max_t):
    """
    Calculate PMF and CDF from discrete-time hazards.

    T = time until event (e.g., adoption)
    t is the discrete time period (e.g., year)
    pi = hazard at time t conditional on survival up to t-1
    pmf is different from hazard because its unconditional, and is the probability for everyone at the start
    cdf is the cumulative pmf up to time t, and is the probability of failure by time t
    survival is the probability of surviving past time t, and is 1 - cdf

    For example:
    PMF(5) = 0.08 → "8% of all subjects fail at time 5"
    π(5) = 0.15 → "15% of survivors to time 5 fail during time 5"
    
    Parameters:
    -----------
    hazards : array of π(t) for t = 1, 2, ..., max_t
    max_t   : maximum time period
    
    Returns:
    --------
    pmf : Pr(T = t) for each t
    cdf : Pr(T ≤ t) for each t  
    survival : Pr(T > t) for each t
    """
    
    pmf = np.zeros(max_t)
    cdf = np.zeros(max_t)
    survival = np.zeros(max_t)
    
    # Initialize
    survival_to_t = 1.0  # S(0) = 1, everyone "survives" to start
    cumulative_prob = 0.0
    
    for t in range(max_t):
        # PMF: probability of failure exactly at t
        # Pr(T = t) = π(t) × Pr(T ≥ t) = π(t) × S(t-1)
        pmf[t] = hazards[t] * survival_to_t
        
        # Update cumulative probability (CDF)
        cumulative_prob = cumulative_prob + pmf[t]
        cdf[t] = cumulative_prob
        
        # Update survival: S(t) = S(t-1) × (1 - π(t))
        survival_to_t = survival_to_t * (1 - hazards[t])
        survival[t] = survival_to_t
    
    return pmf, cdf, survival

### Import Data and Train Model

boushey_2016_full = pd.read_stata(r"data/boushey2016.dta")

# Covariates
covariates = ["policycongruent","gub_election","elect2","hvd_4yr","fedcrime",
                "leg_dem_per_2pty","dem_governor","insession","propneighpol",
                "citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq",
                "violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3"]
boushey_2016 = boushey_2016_full[["state", "year", "billname", "dvadopt"] + covariates].dropna()

# Define X and y
X = boushey_2016[covariates].copy()
y = boushey_2016['dvadopt'].copy()

# Split policies, not individual observations
unique_policies = boushey_2016['billname'].unique()
train_policies, test_policies = train_test_split(
    unique_policies, test_size = 0.2, random_state = 1337
)

# Create train and test masks based on policy membership
train_mask = boushey_2016['billname'].isin(train_policies)
test_mask = boushey_2016['billname'].isin(test_policies)

# Split X and y using the masks
X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use best hyperparameters from the random-split experiment
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0,
    class_weight = None,
    criterion = 'gini',
    max_depth = 10,
    min_samples_leaf = 3,
    n_estimators = 500,
    random_state = 1337
)

# Fit rf model
rf_model.fit(X_train_scaled, y_train)

os.chdir("ml_cdf")

### Calculate CDF for each state-policy combination until adoption (CDF > 0.5) with MICE imputation for missing covariates in extended years

results = []

for policy in test_policies[1:2]:
    # Get all observations for policy
    policy_data = boushey_2016[boushey_2016['billname'] == policy]
    
    # Loop through each state
    for state in policy_data['state'].unique():
        # Get state-specific data, sorted by year
        state_policy_data = policy_data[policy_data['state'] == state].sort_values('year')
        
        if len(state_policy_data) == 0:
            continue
        
        # Initialize variables for extending predictions
        extended_data = state_policy_data.copy()
        max_year = extended_data['year'].max()
        pred_adoption_time = None
        
        # Keep extending until CDF > 0.5 or max iterations are reached
        max_extensions = 20
        for extension in range(max_extensions):
            # Get covariates and scale
            X_state_policy = extended_data[covariates]
            X_state_policy_scaled = scaler.transform(X_state_policy)
            
            # Get hazard predictions
            hazards = rf_model.predict_proba(X_state_policy_scaled)[:, 1]
            
            # Calculate CDF for each time point
            max_t = len(hazards)
            pmf, cdf, survival = discrete_survival_probabilities(hazards, max_t)
            
            # Check if CDF exceeds threshold
            if cdf[-1] > 0.5:
                adoption_indices = np.where(cdf > 0.5)[0]
                pred_adoption_time = extended_data['year'].values[adoption_indices[0]]
                break
            
            # If not, extend by one year using MICE imputation
            last_row = extended_data.iloc[[-1]].copy()
            last_row['year'] = max_year + extension + 1
            
            # Impute covariates for the new year
            imputation_data = pd.concat([extended_data, last_row])
            
            imputer = IterativeImputer(random_state = 1337, max_iter = 10)
            imputed_values = imputer.fit_transform(imputation_data[covariates])
            
            # Update last row with imputed values
            last_row[covariates] = imputed_values[-1]
            last_row['dvadopt'] = 0  # Not yet adopted
            
            # Add to extended data
            extended_data = pd.concat([extended_data, last_row], ignore_index = True)
        
        # Find actual adoption time
        adoption_years = state_policy_data[state_policy_data['dvadopt'] == 1]['year'].values
        actual_adoption_time = adoption_years[0] if len(adoption_years) > 0 else None
        
        results.append({
            'policy': policy,
            'state': state,
            'years': extended_data['year'].values,
            'hazards': hazards,
            'cdf': cdf,
            'actual_adoption_time': actual_adoption_time,
            'pred_adoption_time': pred_adoption_time,
            'num_timepoints': len(extended_data)
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("figures/boushey2016/boushey_cdf_results_rf.csv", index = False)

# Filter for cases where both actual and predicted times are available
valid_predictions = results_df.dropna(subset = ['actual_adoption_time', 'pred_adoption_time'])

# Calculate the adoption year difference for each state-policy
valid_predictions['time_difference'] = valid_predictions['pred_adoption_time'] - valid_predictions['actual_adoption_time']

# Calculate mean absolute error
mean_absolute_difference = valid_predictions['time_difference'].abs().mean()

# Save results to a text file
with open('figures/boushey2016/boushey_mae_rf.txt', 'w') as f:
    f.write(f"Number of valid predictions: {len(valid_predictions)} / {len(results_df)}\n")
    f.write(f"Mean absolute difference: {mean_absolute_difference:.2f} years\n")