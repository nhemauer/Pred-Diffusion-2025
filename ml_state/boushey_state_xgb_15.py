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

random.seed(1337)

# Data
boushey_2016_full = pd.read_stata(r"data/boushey2016.dta")

covariates = ["policycongruent","gub_election","elect2", "hvd_4yr", "fedcrime",
                "leg_dem_per_2pty","dem_governor","insession","propneighpol",
                "citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq",
                "violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3"]
boushey_2016 = boushey_2016_full[["state", "dvadopt"] + covariates].dropna()

# Initialize storage for results
results = {
    'state': {'state': []},
    'xgb': {'ap_score': []}
}

os.chdir("ml_state")

for state in boushey_2016['state'].unique()[10:15]:
    # Create datasets
    train_data = boushey_2016[boushey_2016['state'] != state]
    test_data = boushey_2016[boushey_2016['state'] == state]
    
    # Define X and y for the current state
    X_train = train_data[covariates].copy()
    y_train = train_data['dvadopt']
    X_test = test_data[covariates].copy()
    y_test = test_data['dvadopt']

    # Create groups for CV
    groups = train_data['state']
    
    # Remove current state from groups for CV
    groups = groups[groups != state]

    # Grab unique groups
    unique_groups = np.unique(groups)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Processing State: {state}")
    
    results['state']['state'].append(state)

    # XGBoost hyperparameters
    xgb_grid = {
        'n_estimators': (100, 300),
        'max_depth': (3, 6, 20),
        'max_bin': (32, 128, 256),
        'booster': ['dart'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'reg_lambda': (1, 2),
        'max_leaves': (16, 32),
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
            estimator = XGBClassifier(random_state = 1337, use_label_encoder = False),
            search_spaces = xgb_grid,
            n_iter = 150,
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
        ap_xgb.append(ap_score)

    # Average AP over repeats
    ap_score = np.mean(ap_xgb)

    # Save to results
    results["xgb"]["ap_score"].append(ap_score)

# Convert to dataframe
results_df = pd.DataFrame({
    'state': results['state']['state'],
    'xgb_ap_score': results['xgb']['ap_score']
})

# Save to CSV
results_df.to_csv('figures/boushey2016/boushey_state_results_xgb_15.csv', index = False)