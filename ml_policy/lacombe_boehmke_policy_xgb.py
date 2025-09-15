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
    'xgb': {'ap_score': []}
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

    # XGBoost
    param_grid = {
        'n_estimators': (100, 300),
        'max_depth': (3, 6, 20),
        'max_bin': (16, 32, 64),
        'booster': ['dart'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
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
    'xgb_ap_score': results['xgb']['ap_score']
})

# Save to CSV
results_df.to_csv('figures/lacombe_boehmke2021/lacombe_policy_results_xgb.csv', index = False)