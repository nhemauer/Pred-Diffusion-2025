import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import os

random.seed(1337)

# Data
mallinson_2019_full = pd.read_csv(r"data/mallinson2019.csv")

covariates = ["neighbor_prop", "ideology_relative_hm", "congress_majortopic", "init_avail", "init_qual", "divided_gov",
              "legprof_squire", "percap_log", "population_log", "mip", "complexity_topic", "mip_complexity_topic", "nyt", "year_count", "time_log"]
mallinson_2019 = mallinson_2019_full[["adopt", "policy", "state", "year"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'original': {'ap_score': []},
    'logit': {'ap_score': []},
    'rf': {'ap_score': []},
    'xgb': {'ap_score': []}
}

os.chdir("ml_policy")

for bill in mallinson_2019['policy'].unique():
    # Create datasets
    train_data = mallinson_2019[mallinson_2019['policy'] != bill]
    test_data = mallinson_2019[mallinson_2019['policy'] == bill]
    
    # Define X and y for the current bill
    X_train = train_data[covariates].copy()
    y_train = train_data['adopt']
    X_test = test_data[covariates].copy()
    y_test = test_data['adopt']

    # Create groups for LeaveOneGroupOut
    groups = train_data['policy']

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

    # Logistic Regression
    common_params = {
        "class_weight": [None, "balanced"],
        "fit_intercept": [True],
    }

    C_params = {"C": [0.001, 0.01, 0.1]}  # only for models with a penalty

    param_grid = [
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

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337),
        param_grid = param_grid,
        cv = LeaveOneGroupOut(),
        scoring = 'average_precision',
        n_jobs = -1,
        verbose = 0,
        refit = True
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train, groups = groups)

    # Get the best model and score on test set
    best_model = grid_search.best_estimator_
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    ap_score = average_precision_score(y_test, test_scores)
    print(f"Logistic Regression AP Score: {ap_score}")
    
    results['logit']['ap_score'].append(ap_score)

    # Random Forest
    param_grid = {
            'n_estimators': (100, 500),
            'criterion': ['entropy'],
            'max_depth': (10, 25, 50),
            'min_samples_leaf': (1, 4),
            'bootstrap': [True],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': (0.0, 0.1),
    }

    # Set up GridSearchCV
    grid_search = BayesSearchCV(
        estimator = RandomForestClassifier(random_state = 1337),
        search_spaces = param_grid,
        n_iter = 150,
        cv = LeaveOneGroupOut(),
        n_jobs = -1,
        verbose = 0,
        scoring = "average_precision",
        random_state = 1337
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train, groups = groups)

    # Get the best model and score on test set
    best_model = grid_search.best_estimator_
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    ap_score = average_precision_score(y_test, test_scores)
    print(f"Random Forest AP Score: {ap_score}")
    
    results['rf']['ap_score'].append(ap_score)

    # XGBoost
    param_grid = {
        'n_estimators': (100, 300),
        'max_depth': (3, 6, 20),
        'max_bin': (32, 64, 256),
        'booster': ['gbtree'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'min_child_weight': (5, 10),
        'max_leaves': (16, 32),
    }

    # Set up GridSearchCV
    grid_search = BayesSearchCV(
        estimator = XGBClassifier(random_state = 1337, use_label_encoder = False),
        search_spaces = param_grid,
        n_iter = 150,
        cv = LeaveOneGroupOut(),
        n_jobs = -1,
        verbose = 0,
        scoring = "average_precision",
        random_state = 1337
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train, groups = groups)

    # Get the best model and score on test set
    best_model = grid_search.best_estimator_
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    ap_score = average_precision_score(y_test, test_scores)
    print(f"XGBoost AP Score: {ap_score}")
    
    results['xgb']['ap_score'].append(ap_score)

# Convert to dataframe
results_df = pd.DataFrame({
    'billname': results['bill']['billname'],
    'original_ap_score': results['original']['ap_score'],
    'logit_ap_score': results['logit']['ap_score'],
    'rf_ap_score': results['rf']['ap_score'],
    'xgb_ap_score': results['xgb']['ap_score']
})

# Save to CSV
results_df.to_csv('figures/mallinson2019/mallinson_policy_results.csv', index = False)