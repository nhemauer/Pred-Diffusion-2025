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
bricker_lacombe_2021_full = pd.read_stata(r"data/bricker_lacombe2021.dta")

# Covariates
covariates = ["year","std_score","initiative","init_sigs","std_population",
                "std_citideology","unified","std_income","std_legp_squire",
                "duration","durationsq","durationcb"]
bricker_lacombe_2021 = bricker_lacombe_2021_full[["state", "policy", "adoption"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'rf': {'ap_score': []},
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

    # Create groups for LeaveOneGroupOut
    groups = train_data['policy']

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

# Convert to dataframe
results_df = pd.DataFrame({
    'billname': results['bill']['billname'],
    'rf_ap_score': results['rf']['ap_score'],
})

# Save to CSV
results_df.to_csv('figures/bricker_lacombe2021/bricker_policy_results_rf.csv', index = False)