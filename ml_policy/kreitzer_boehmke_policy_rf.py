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
kreitzer_boehmke_2016_full = pd.read_stata(r"data/kreitzer_boehmke2016.dta")

covariates = [
    "norrander_legality", "religadhrate", "initdif", "dem_gov", "uni_dem_leg",
    "fem_dem", "nbrspct", "rescaledmedincome", "rescaledpopsize", "time", 
    "time2", "webster"
]
kreitzer_boehmke_2016 = kreitzer_boehmke_2016_full[["adopt_policy", "state", "year", "policy_num"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'rf': {'ap_score': []},
}

os.chdir("ml_policy")

for bill in kreitzer_boehmke_2016['policy_num'].unique():
    # Create datasets
    train_data = kreitzer_boehmke_2016[kreitzer_boehmke_2016['policy_num'] != bill]
    test_data = kreitzer_boehmke_2016[kreitzer_boehmke_2016['policy_num'] == bill]
    
    # Define X and y for the current bill
    X_train = train_data[covariates].copy()
    y_train = train_data['adopt_policy']
    X_test = test_data[covariates].copy()
    y_test = test_data['adopt_policy']

    # Create groups for LeaveOneGroupOut
    groups = train_data['policy_num']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Processing bill: {bill}")

    # Random Forest
    param_grid = {
            'n_estimators': (100, 500),
            'criterion': ['gini', 'entropy'],
            'max_depth': (None, 10, 25, 50),
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
results_df.to_csv('figures/kreitzer_boehmke2016/kreitzer_policy_results_rf.csv', index = False)