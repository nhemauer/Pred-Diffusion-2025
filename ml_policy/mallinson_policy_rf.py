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
mallinson_2019_full = pd.read_csv(r"data/mallinson2019.csv")

covariates = ["neighbor_prop", "ideology_relative_hm", "congress_majortopic", "init_avail", "init_qual", "divided_gov",
              "legprof_squire", "percap_log", "population_log", "mip", "complexity_topic", "mip_complexity_topic", "nyt", "year_count", "time_log"]
mallinson_2019 = mallinson_2019_full[["adopt", "policy", "state", "year"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'rf': {'ap_score': []},
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

# Convert to dataframe
results_df = pd.DataFrame({
    'billname': results['bill']['billname'],
    'rf_ap_score': results['rf']['ap_score'],
})

# Save to CSV
results_df.to_csv('figures/mallinson2019/mallinson_policy_results_rf.csv', index = False)