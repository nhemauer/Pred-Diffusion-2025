from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import warnings
import os

warnings.filterwarnings('ignore')
random.seed(1337)

# Data
bricker_lacombe_2021_full = pd.read_stata(r"data/bricker_lacombe2021.dta")

# Covariates
covariates = ["year","std_score","initiative","init_sigs","std_population",
                "std_citideology","unified","std_income","std_legp_squire",
                "duration","durationsq","durationcb"]
bricker_lacombe_2021 = bricker_lacombe_2021_full[["state", "policy", "adoption"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'original': {'ap_score': []},
    'logit': {'ap_score': []},
    'rf': {'ap_score': []},
    'xgb': {'ap_score': []}
}

os.chdir("ml_policy")

for bill in bricker_lacombe_2021['policy'].unique():
    # Create datasets
    train_data = bricker_lacombe_2021[bricker_lacombe_2021['policy'] != bill]
    test_data = bricker_lacombe_2021[bricker_lacombe_2021['policy'] == bill]
    
    # Define X and y for the current bill
    X_train = train_data[covariates].copy()
    y_train = train_data['adoption']
    X_test = test_data[covariates].copy()
    y_test = test_data['adoption']

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

    # Logistic Regression
    common_params = {
        'C': [0.001, 0.01, 0.1, 1, 2],
        'class_weight': [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}, {0: 1, 1: 8}, {0: 1, 1: 9}, {0: 1, 1: 10}],
        'fit_intercept': [True, False]
    }

    param_grid = [
        # lbfgs supports only l2 or none
        {
            **common_params,
            'solver': ['lbfgs'],
            'penalty': ['l2', None]
        },
        # newton-cholesky supports only l2 or none
        {
            **common_params,
            'solver': ['newton-cholesky'],
            'penalty': ['l2', None]
        },
        # liblinear supports l1 and l2 only (no elasticnet or none)
        {
            **common_params,
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2']
        },
        # saga supports l1, l2, elasticnet
        {
            **common_params,
            'solver': ['saga'],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # Only used if penalty = 'elasticnet', ignored otherwise
        }
    ]

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337),
        param_grid = param_grid,
        cv = 5,
        scoring = 'average_precision',
        n_jobs = -1,
        verbose = 0,
        refit = True
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and score on test set
    best_model = grid_search.best_estimator_
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    ap_score = average_precision_score(y_test, test_scores)
    print(f"Logistic Regression AP Score: {ap_score}")
    
    results['logit']['ap_score'].append(ap_score)

    # Random Forest
    param_grid = {
            'n_estimators': (100, 300, 500),
            'criterion': ['gini', 'entropy'],
            'max_depth': (10, 25, 50),
            'min_samples_split': (2, 10),
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
        cv = 5,
        n_jobs = -1,
        verbose = 0,
        scoring = "average_precision",
        random_state = 1337
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and score on test set
    best_model = grid_search.best_estimator_
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    ap_score = average_precision_score(y_test, test_scores)
    print(f"Random Forest AP Score: {ap_score}")
    
    results['rf']['ap_score'].append(ap_score)

    # XGBoost
    param_grid = {
        'n_estimators': (100, 500),
        'max_depth': (3, 6, 10),
        'max_bin': (32, 64, 128),
        'booster': ['gbtree'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'gamma': (0, 2),
        'min_child_weight': (5, 10),
        'scale_pos_weight': (1, 5)
    }

    # Set up GridSearchCV
    grid_search = BayesSearchCV(
        estimator = XGBClassifier(random_state = 1337, use_label_encoder = False),
        search_spaces = param_grid,
        n_iter = 150,
        cv = 5,
        n_jobs = -1,
        verbose = 0,
        scoring = "average_precision",
        random_state = 1337
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

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
results_df.to_csv('figures/bricker_lacombe2021/bricker_policy_results.csv', index = False)