import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import random
import os

random.seed(1337)
np.random.seed(1337)

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

### Import Data

lacombe_boehmke2021_full = pd.read_stata(r"data/lacombe_boehmke2021.dta")

covariates = [
    "initiative", "init_sigs", "std_latnt_decay", "std_nbrs_lag", "std_population",
    "std_masssociallib_est", "unified", "duration", "durationsq", "durationcb", "std_income",
    "std_bowen_1", "std_bowen_2", "change_pop", "change_inc", "party_change"
]

lacombe_boehmke2021 = lacombe_boehmke2021_full[["adoption", "state", "year", "policyno"] + covariates].dropna()

# Define X and y
X_full = lacombe_boehmke2021.drop(columns = ['adoption', 'state', 'policyno']).copy()
X_full = pd.get_dummies(X_full, columns = ['year'], drop_first = True)
y_full = lacombe_boehmke2021['adoption']

os.chdir("ml_adoption_timing")

### Cross-validation on policies

unique_policies = lacombe_boehmke2021['policyno'].unique()
kfold = KFold(n_splits = 5, shuffle = True, random_state = 1337)

all_fold_results = []
fold_maes = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(unique_policies)):
    
    train_policies = unique_policies[train_idx]
    test_policies = unique_policies[test_idx]
    
    # Create train and test masks based on policy membership
    train_mask = lacombe_boehmke2021['policyno'].isin(train_policies)
    test_mask = lacombe_boehmke2021['policyno'].isin(test_policies)
    
    # Split X and y using the masks
    X_train = X_full[train_mask]
    X_test = X_full[test_mask]
    y_train = y_full[train_mask]
    y_test = y_full[test_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logit_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)
    logit_model.fit(X_train_scaled, y_train)
    
    ### Calculate CDF for each state-policy combination
    results = []
    
    for policy in test_policies:
        # Get all observations for policy
        policy_data = lacombe_boehmke2021[lacombe_boehmke2021['policyno'] == policy]
        
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
            hazards = None
            cdf = None
            
            for extension in range(max_extensions):
                # Prepare features
                X_state_policy = extended_data[['year'] + covariates].copy()
                X_state_policy = pd.get_dummies(X_state_policy, columns = ['year'], drop_first = True)
                
                # Align columns to match training data
                X_state_policy = X_state_policy.reindex(columns = X_train.columns, fill_value = 0)
                
                # Scale features
                X_state_policy_scaled = scaler.transform(X_state_policy)
                
                # Get hazard predictions
                hazards = logit_model.predict_proba(X_state_policy_scaled)[:, 1]
                
                # Calculate CDF for each time point
                max_t = len(hazards)
                pmf, cdf, survival = discrete_survival_probabilities(hazards, max_t)
                
                # Check if CDF exceeds threshold
                if cdf[-1] > 0.5:
                    adoption_indices = np.where(cdf > 0.5)[0]
                    pred_adoption_time = extended_data['year'].values[adoption_indices[0]]
                    break
                
                # Only extend if we haven't reached max iterations
                if extension < max_extensions - 1:
                    # If not, extend by one year using MICE imputation
                    last_row = extended_data.iloc[[-1]].copy()
                    last_row['year'] = max_year + extension + 1
                    
                    # Impute covariates for the new year
                    imputation_data = pd.concat([extended_data, last_row])
                    
                    imputer = IterativeImputer(random_state = 1337, max_iter = 10)
                    imputed_values = imputer.fit_transform(imputation_data[covariates])
                    
                    # Update last row with imputed values
                    last_row[covariates] = imputed_values[-1]
                    last_row['adoption'] = 0  # Not yet adopted
                    
                    # Add to extended data
                    extended_data = pd.concat([extended_data, last_row], ignore_index = True)
            
            # Find actual adoption time
            adoption_years = state_policy_data[state_policy_data['adoption'] == 1]['year'].values
            actual_adoption_time = adoption_years[0] if len(adoption_years) > 0 else None
            
            results.append({
                'fold': fold_idx + 1,
                'policy': policy,
                'state': state,
                'years': extended_data['year'].values,
                'hazards': hazards,
                'cdf': cdf,
                'actual_adoption_time': actual_adoption_time,
                'pred_adoption_time': pred_adoption_time,
                'num_timepoints': len(extended_data)
            })
    
    # Calculate MAE for this fold
    fold_results_df = pd.DataFrame(results)
    full_predictions = fold_results_df.dropna(subset = ['actual_adoption_time'])
    valid_predictions = full_predictions.dropna(subset = ['pred_adoption_time'])
    
    if len(valid_predictions) > 0:
        valid_predictions['time_difference'] = valid_predictions['pred_adoption_time'] - valid_predictions['actual_adoption_time']
        fold_mae = valid_predictions['time_difference'].abs().mean()
        fold_maes.append(fold_mae)
    
    all_fold_results.extend(results)

# Convert all results to DataFrame
results_df = pd.DataFrame(all_fold_results)
results_df.to_csv("figures/lacombe_boehmke2021/lacombe_boehmke_cdf_results_logit.csv", index = False)

# Calculate overall statistics
full_predictions = results_df.dropna(subset = ['actual_adoption_time'])
valid_predictions = full_predictions.dropna(subset = ['pred_adoption_time'])
valid_predictions['time_difference'] = valid_predictions['pred_adoption_time'] - valid_predictions['actual_adoption_time']
overall_mae = valid_predictions['time_difference'].abs().mean()

# Precision / Recall / F1 
predicted_adoption = results_df['pred_adoption_time'].notna()
actually_adopted = results_df['actual_adoption_time'].notna()

tp = (predicted_adoption & actually_adopted).sum()
fp = (predicted_adoption & ~actually_adopted).sum()
fn = (~predicted_adoption & actually_adopted).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# Save results to a text file
with open('figures/lacombe_boehmke2021/lacombe_boehmke_maef1_logit.txt', 'w') as f:
    for i, mae in enumerate(fold_maes):
        f.write(f"Fold {i+1} MAE: {mae:.2f} years\n")
    f.write("\n")
    f.write(f"Mean MAE across folds: {np.mean(fold_maes):.2f} years\n")
    f.write(f"Std MAE across folds: {np.std(fold_maes):.2f} years\n")
    f.write(f"Valid predictions: {len(valid_predictions)} / {len(full_predictions)}\n")
    f.write("\n")
    f.write("Adoption Prediction Coverage:\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1 Score:  {f1:.4f}\n")