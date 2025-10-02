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
parinandi_2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]

parinandi_2020 = parinandi_2020_full[["oneemulation", "state", "featurenumber"] + covariates].dropna()

# Initialize storage for results
results = {
    'state': {'state': []},
    'original': {'ap_score': []},
    'logit': {'ap_score': []},
    'rf': {'ap_score': []},
    'xgb': {'ap_score': []}
}

os.chdir("ml_state")

for state in parinandi_2020['state'].unique():
    # Create datasets
    train_data = parinandi_2020[parinandi_2020['state'] != state]
    test_data = parinandi_2020[parinandi_2020['state'] == state]
    
    # Define X and y for the current state
    X_train = train_data[covariates].copy()
    y_train = train_data['oneemulation']
    X_test = test_data[covariates].copy()
    y_test = test_data['oneemulation']

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

    # Original Logit
    original_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_scores = original_model.predict_proba(X_test_scaled)[:, 1]
    
    results['state']['state'].append(state)
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

    # Random Forest hyperparameters
    rf_grid = {
            'n_estimators': (100, 500),
            'criterion': ['entropy', 'log_loss'],
            'max_depth': (10, 25, 50),
            'min_samples_leaf': (1, 4),
            'bootstrap': [True],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': (0.0, 0.1),
            'max_samples': (0.5, 0.75)
    }

    # XGBoost hyperparameters
    xgb_grid = {
        'n_estimators': (100, 500),
        'max_depth': (3, 10, 20),
        'max_bin': (32, 64, 256),
        'booster': ['gbtree'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'gamma': (0, 2),
        'reg_alpha': (0, 2),
        'reg_lambda': (1, 2),
        'min_child_weight': (1, 5, 10),
        'scale_pos_weight': (1, 5, 10)
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

    for rep in range(n_repeats): # 3 CV repeats
        shuffled = unique_groups.copy() # Shuffle to ensure group randomness
        np.random.shuffle(shuffled)
        mapping = {g: i for i, g in enumerate(shuffled)}
        shuffled_groups = np.array([mapping[g] for g in groups])

        # Fit BayesSearchCV
        grid_search = BayesSearchCV(
            estimator = RandomForestClassifier(random_state = 1337),
            search_spaces = rf_grid,
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
        ap_rf.append(ap_score)

    # Average AP over repeats
    ap_score = np.mean(ap_rf)

    # Save to results
    results["rf"]["ap_score"].append(ap_score)

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
    'original_ap_score': results['original']['ap_score'],
    'logit_ap_score': results['logit']['ap_score'],
    'rf_ap_score': results['rf']['ap_score'],
    'xgb_ap_score': results['xgb']['ap_score']
})

# Save to CSV
results_df.to_csv('figures/parinandi2020/parinandi_state_results.csv', index = False)