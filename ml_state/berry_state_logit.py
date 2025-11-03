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
berry_berry1990_full = pd.read_csv("data/berry_berry1990.txt", delim_whitespace = True, header = None)
berry_berry1990_full.columns = ["state", "year", "adopt", "fiscal_1", "party", "elect1", "elect2", "income_1", "neighbor", "nbrpercn", "religion"]
berry_berry1990 = berry_berry1990_full[berry_berry1990_full['party'] != 9].copy() # 9 is the NA (For MN and NE)

# Initialize storage for results - store all predictions and labels
results = {
    'original': {'all_predictions': [], 'all_true_labels': []},
    'logit': {'all_predictions': [], 'all_true_labels': []},
}

os.chdir("ml_state")

for state in berry_berry1990['state'].unique():
    # Create datasets
    train_data = berry_berry1990[berry_berry1990['state'] != state]
    test_data = berry_berry1990[berry_berry1990['state'] == state]
    
    # Define X and y for the current state
    X_train = train_data.drop(columns = ['adopt', 'neighbor', 'state', 'year']).copy()
    y_train = train_data['adopt']
    X_test = test_data.drop(columns = ['adopt', 'neighbor', 'state', 'year']).copy()
    y_test = test_data['adopt']

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
    
    # Store predictions and true labels for overall calculation
    results['original']['all_predictions'].extend(original_scores)
    results['original']['all_true_labels'].extend(y_test)

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

        # liblinear: l1 / l2 (no "none" allowed)
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

    # CV setup
    n_splits = 5
    n_repeats = 3

    ap_logit, ap_rf, ap_xgb = [], [], []

    # Store predictions for this state across repeats
    state_logit_preds, state_rf_preds, state_xgb_preds = [], [], []

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
        state_logit_preds.append(y_scores)

    # Average predictions across repeats for this state
    avg_logit_preds = np.mean(state_logit_preds, axis = 0)
    results['logit']['all_predictions'].extend(avg_logit_preds)
    results['logit']['all_true_labels'].extend(y_test)

# Calculate overall AUCPR scores
overall_ap_scores = {}
for model in ['original', 'logit']:
    overall_ap_scores[model] = average_precision_score(
        results[model]['all_true_labels'], 
        results[model]['all_predictions']
    )

# Save to CSV
results_df = pd.DataFrame({
    'model': ['original', 'logit'],
    'overall_ap_score': [overall_ap_scores[model] for model in ['original', 'logit']]
})

results_df.to_csv('figures/berry_berry1990/berry_state_results_logit.csv', index = False)