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
parinandi_2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]

parinandi_2020 = parinandi_2020_full[["oneemulation", "state", "featurenumber"] + covariates].dropna()

# Initialize storage for results
results = {
    'bill': {'billname': []},
    'xgb': {'ap_score': []}
}

os.chdir("ml_policy")

for bill in parinandi_2020['featurenumber'].unique():
    # Create datasets
    train_data = parinandi_2020[parinandi_2020['featurenumber'] != bill]
    test_data = parinandi_2020[parinandi_2020['featurenumber'] == bill]
    
    # Define X and y for the current bill
    X_train = train_data[covariates].copy()
    y_train = train_data['oneemulation']
    X_test = test_data[covariates].copy()
    y_test = test_data['oneemulation']

    # Create groups for LeaveOneGroupOut
    groups = train_data['featurenumber']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Processing bill: {bill}")

    # XGBoost
    param_grid = {
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
results_df.to_csv('figures/parinandi2020/parinandi_policy_results_xgb.csv', index = False)