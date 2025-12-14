import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import os
import sys

random.seed(1337)

# Data
mallinson_2019_full = pd.read_csv(r"data/mallinson2019.csv")

covariates = ["neighbor_prop", "ideology_relative_hm", "congress_majortopic", "init_avail", "init_qual", "divided_gov",
              "legprof_squire", "percap_log", "population_log", "mip", "complexity_topic", "mip_complexity_topic", "nyt", "year_count", "time_log"]
mallinson_2019 = mallinson_2019_full[["adopt", "policy", "state", "year"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'rf': {'ap_score': []},
}

os.chdir("ml_policy")

# Get all unique policies
unique_policies = mallinson_2019['policy'].unique()

# Get the specific policy for the sbatch job
policy_idx = int(sys.argv[1])
bill = unique_policies[policy_idx]

# Create datasets
train_data = mallinson_2019[mallinson_2019['policy'] != bill]
test_data = mallinson_2019[mallinson_2019['policy'] == bill]

# Define X and y for the current bill
X_train = train_data[covariates].copy()
y_train = train_data['adopt']
X_test = test_data[covariates].copy()
y_test = test_data['adopt']

# Create groups for CV
groups = train_data['policy']

# Remove current bill from groups for CV
groups = groups[groups != bill]

# Grab unique groups
unique_groups = np.unique(groups)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Processing bill: {bill}")

results['bill']['billname'].append(bill)

# Random Forest hyperparameters
rf_grid = {
        'n_estimators': (100, 500),
        'criterion': ['entropy'],
        'max_depth': (10, 25),
        'min_samples_leaf': (1, 4),
        'bootstrap': [True],
        'class_weight': [None, 'balanced'],
        'ccp_alpha': (0.0, 0.1),
}

# CV setup
n_splits = 5
n_repeats = 3

ap_logit, ap_rf, ap_xgb = [], [], []

for rep in range(n_repeats): # 3 CV repeats
    shuffled = unique_groups.copy() # Shuffle to ensure group randomness
    np.random.shuffle(shuffled)
    mapping = {g: i for i, g in enumerate(shuffled)}
    shuffled_groups = np.array([mapping[g] for g in groups])

    # Fit BayesSearchCV
    grid_search = BayesSearchCV(
        estimator = RandomForestClassifier(random_state = 1337),
        search_spaces = rf_grid,
        n_iter = 80,
        cv = GroupKFold(n_splits = n_splits),
        n_jobs = -1,
        verbose = 0,
        scoring = "average_precision",
        random_state = 1337
    )

    grid_search.fit(X_train_scaled, y_train, groups = shuffled_groups)

    # Use the refitted best model
    best_model = grid_search.best_estimator_
    
    # Get predicted probabilities for the positive class
    y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

    # Compute average precision (AUC PR)
    ap_score = average_precision_score(y_test, y_scores)

    # Append to list
    ap_rf.append(ap_score)

# Average AP over repeats
ap_score = np.mean(ap_rf)

# Save to results
results["rf"]["ap_score"].append(ap_score)

# Convert to dataframe
results_df = pd.DataFrame({
    'billname': results['bill']['billname'],
    'rf_ap_score': results['rf']['ap_score'],
})

# Save to CSV
results_df.to_csv(f'figures/mallinson2019/mallinson_policy_results_rf_{policy_idx}.csv', index = False)