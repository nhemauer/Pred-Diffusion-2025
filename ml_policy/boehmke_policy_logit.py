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
boehmke_2017_full = pd.read_stata(r"data/boehmke2017.dta")

covariates = ["year","srcs_decay","nbrs_lag","rpcpinc","totpop","legp_squire",
                "citi6010","unif_rep","unif_dem","time","time_sq","time_cube"]
boehmke_2017 = boehmke_2017_full[["state", "policy", "adopt"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'original': {'ap_score': []},
    'logit': {'ap_score': []},
}

os.chdir("ml_policy")

for bill in boehmke_2017['policy'].unique():
    # Create datasets
    train_data = boehmke_2017[boehmke_2017['policy'] != bill]
    test_data = boehmke_2017[boehmke_2017['policy'] == bill]
    
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

    # Create dummies for train set
    X_train = pd.get_dummies(X_train, columns = ['year'], drop_first = True)
    
    # Create dummies for test set
    X_test = pd.get_dummies(X_test, columns = ['year'], drop_first = True)
    
    # Ensure both have the same columns by reindexing
    all_columns = X_train.columns.union(X_test.columns)
    X_train = X_train.reindex(columns = all_columns, fill_value = 0)
    X_test = X_test.reindex(columns = all_columns, fill_value = 0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Processing bill: {bill}")

    # Original Logit
    original_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_scores = original_model.predict_proba(X_test_scaled)[:, 1]
    
    results['bill']['billname'].append(bill)
    results['original']['ap_score'].append(average_precision_score(y_test, original_scores))

    # Logistic Regression hyperparameters
    common_params = {
        "class_weight": [None, "balanced"],
        "fit_intercept": [True],
    }

    C_params = {"C": [0.001, 0.01, 0.1]}  # only for models with a penalty

    logit_grid = [
        # lbfgs: l2 with C
        {**common_params, **C_params, "solver": ["lbfgs"], "penalty": ["l2"]},

        # lbfgs: no penalty
        {**common_params, "solver": ["lbfgs"], "penalty": [None]},

        # newton-cholesky: l2 with C
        {**common_params, **C_params, "solver": ["newton-cholesky"], "penalty": ["l2"]},

        # newton-cholesky: no penalty
        {**common_params, "solver": ["newton-cholesky"], "penalty": [None]},

        # liblinear: l1 / l2 (no “none” allowed)
        {**common_params, **C_params, "solver": ["liblinear"], "penalty": ["l1", "l2"]},

        # saga: l1 / l2 with C
        {**common_params, **C_params, "solver": ["saga"], "penalty": ["l1", "l2"]},

        # saga: elasticnet with l1_ratio
        {
            **common_params,
            **C_params,
            "solver": ["saga"],
            "penalty": ["elasticnet"],
            "l1_ratio": [0, 0.5, 1],
        },

        # saga: no penalty
        {**common_params, "solver": ["saga"], "penalty": [None]},
    ]

    # CV setup
    n_splits = 5
    n_repeats = 3

    ap_logit, ap_rf, ap_xgb = [], [], []

    for rep in range(n_repeats): # 3 CV repeats
        shuffled = unique_groups.copy() # Shuffle to ensure group randomness
        np.random.shuffle(shuffled)
        mapping = {g: i for i, g in enumerate(shuffled)}
        shuffled_groups = np.array([mapping[g] for g in groups])

        # Fit GridSearch
        grid_search = GridSearchCV(
            estimator = linear_model.LogisticRegression(max_iter = 2000, random_state = 1337),
            param_grid = logit_grid,
            scoring = "average_precision",
            cv = GroupKFold(n_splits = n_splits),
            n_jobs = -1,
            verbose = 0
        )

        grid_search.fit(X_train_scaled, y_train, groups = shuffled_groups)

        # Use the refitted best model
        best_model = grid_search.best_estimator_
        
        # Get predicted probabilities for the positive class
        y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

        # Compute average precision (AUC PR)
        ap_score = average_precision_score(y_test, y_scores)

        # Append to list
        ap_logit.append(ap_score)

    # Average AP over repeats
    ap_score = np.mean(ap_logit)

    # Save to results
    results["logit"]["ap_score"].append(ap_score)

# Convert to dataframe
results_df = pd.DataFrame({
    'billname': results['bill']['billname'],
    'original_ap_score': results['original']['ap_score'],
    'logit_ap_score': results['logit']['ap_score'],
})

# Save to CSV
results_df.to_csv('figures/boehmke2017/boehmke_policy_results_logit.csv', index = False)