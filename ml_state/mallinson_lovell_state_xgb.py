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
mallinson_lovell2022_full = pd.read_csv(r"data/mallinson_lovell2022.csv")

# Covariates
covariates = ["republican","legprof_squire","exp_pupil10000_adj","mathscore4th","readscore4th",
              "time"]
mallinson_lovell2022 = mallinson_lovell2022_full[["adopt", "state"] + covariates].dropna()

# Initialize storage for results - store all predictions and labels
results = {
    'xgb': {'all_predictions': [], 'all_true_labels': []}
}

os.chdir("ml_state")

for state in mallinson_lovell2022['state'].unique():
    # Create datasets
    train_data = mallinson_lovell2022[mallinson_lovell2022['state'] != state]
    test_data = mallinson_lovell2022[mallinson_lovell2022['state'] == state]
    
    # Define X and y for the current state
    X_train = train_data[covariates].copy()
    y_train = train_data['adopt']
    X_test = test_data[covariates].copy()
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

    # XGBoost hyperparameters
    xgb_grid = {
        'max_depth': (3, 6, 10),
        'booster': ['gbtree'],
        'objective': ['binary:logistic'],
        'eval_metric': ['aucpr'],
        'tree_method': ['auto'],
        'grow_policy': ['depthwise'],
        'subsample': (0.5, 1.0),
        'gamma': (0, 2),
        'reg_alpha': (0, 1),
        'scale_pos_weight': (1, 5)
    }

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

        # Fit BayesSearchCV
        grid_search = BayesSearchCV(
            estimator = XGBClassifier(random_state = 1337, use_label_encoder = False),
            search_spaces = xgb_grid,
            n_iter = 100,
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
        state_xgb_preds.append(y_scores)

    # Average predictions across repeats for this state
    avg_xgb_preds = np.mean(state_xgb_preds, axis = 0)
    results['xgb']['all_predictions'].extend(avg_xgb_preds)
    results['xgb']['all_true_labels'].extend(y_test)

# Calculate overall AUCPR scores
overall_ap_scores = {}
for model in ['xgb']:
    overall_ap_scores[model] = average_precision_score(
        results[model]['all_true_labels'], 
        results[model]['all_predictions']
    )

# Save to CSV
results_df = pd.DataFrame({
    'model': ['xgb'],
    'overall_ap_score': [overall_ap_scores[model] for model in ['xgb']]
})

results_df.to_csv('figures/mallinson_lovell2022/mallinson_lovell_state_results_xgb.csv', index = False)