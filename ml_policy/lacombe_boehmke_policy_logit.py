from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import warnings
import os
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category = ConvergenceWarning)

random.seed(1337)

# Data
lacombe_boehmke2021_full = pd.read_stata(r"data/lacombe_boehmke2021.dta")

covariates = [
    "initiative", "init_sigs", "std_latnt_decay", "std_nbrs_lag", "std_population",
    "std_masssociallib_est", "unified", "duration", "durationsq", "durationcb", "std_income",
    "std_bowen_1", "std_bowen_2", "change_pop", "change_inc", "party_change", "year"
]

lacombe_boehmke2021 = lacombe_boehmke2021_full[["adoption", "policyno", 'state'] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'original': {'ap_score': []},
    'logit': {'ap_score': []},
}

os.chdir("ml_policy")

for bill in lacombe_boehmke2021['policyno'].unique():
    # Create datasets
    train_data = lacombe_boehmke2021[lacombe_boehmke2021['policyno'] != bill]
    test_data = lacombe_boehmke2021[lacombe_boehmke2021['policyno'] == bill]
    
    # Define X and y for the current bill
    X_train = train_data[covariates].copy()
    y_train = train_data['adoption']
    X_test = test_data[covariates].copy()
    y_test = test_data['adoption']

    # Create groups for LeaveOneGroupOut
    groups = train_data['policyno']

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
        'C': [0.001, 0.01, 0.1],
        'class_weight': [None, 'balanced'],
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
            'l1_ratio': [0, 0.5, 1]  # Only used if penalty = 'elasticnet', ignored otherwise
        }
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

# Convert to dataframe
results_df = pd.DataFrame({
    'billname': results['bill']['billname'],
    'original_ap_score': results['original']['ap_score'],
    'logit_ap_score': results['logit']['ap_score'],
})

# Save to CSV
results_df.to_csv('figures/lacombe_boehmke2021/lacombe_policy_results_logit.csv', index = False)