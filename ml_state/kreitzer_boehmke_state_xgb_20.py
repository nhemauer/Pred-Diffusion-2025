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
kreitzer_boehmke_2016_full = pd.read_stata(r"data/kreitzer_boehmke2016.dta")

covariates = [
    "norrander_legality", "religadhrate", "initdif", "dem_gov", "uni_dem_leg",
    "fem_dem", "nbrspct", "rescaledmedincome", "rescaledpopsize", "time", 
    "time2", "webster", "policy_num"
]
kreitzer_boehmke_2016 = kreitzer_boehmke_2016_full[["adopt_policy", "state", "year"] + covariates].dropna()

# Initialize storage for results
results = {
    'state': {'state': []},
    'xgb': {'ap_score': []}
}

os.chdir("ml_state")

for state in kreitzer_boehmke_2016['state'].unique()[15:20]:
    # Create datasets
    train_data = kreitzer_boehmke_2016[kreitzer_boehmke_2016['state'] != state]
    test_data = kreitzer_boehmke_2016[kreitzer_boehmke_2016['state'] == state]
    
    # Define X and y for the current state
    X_train = train_data[covariates].copy()
    y_train = train_data['adopt_policy']
    X_test = test_data[covariates].copy()
    y_test = test_data['adopt_policy']

    # Create groups for CV
    groups = train_data['state']
    
    # Remove current state from groups for CV
    groups = groups[groups != state]

    # Grab unique groups
    unique_groups = np.unique(groups)

    # Create dummies for train set
    X_train = pd.get_dummies(X_train, columns = ['policy_num'], drop_first = True)
    
    # Create dummies for test set
    X_test = pd.get_dummies(X_test, columns = ['policy_num'], drop_first = True)
    
    # Ensure both have the same columns by reindexing
    all_columns = X_train.columns.union(X_test.columns)
    X_train = X_train.reindex(columns = all_columns, fill_value = 0)
    X_test = X_test.reindex(columns = all_columns, fill_value = 0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Processing State: {state}")
    
    results['state']['state'].append(state)

    # XGBoost hyperparameters
    xgb_grid = {
        'max_depth': (6, 10),
        'booster': ['dart'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'reg_alpha': (0, 2),
        'min_child_weight': (1, 10),
        'scale_pos_weight': (1, 5)
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
results_df.to_csv('figures/kreitzer_boehmke2016/kreitzer_state_results_xgb_20.csv', index = False)