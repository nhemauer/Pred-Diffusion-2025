### Preprocessing and Rolling Window t+1
import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

random.seed(1337)

# Data
karch_2016_full = pd.read_stata(r"data/karch2016.dta")

covariates = [
    "traditional", "nborsstd", "prevadoptstd", "complexity", "igrole",
    "regov", "unified", "perdemstd", "incpcadjstd", "exppcadjstd",
    "logpopstd", "collegstd", "perurbanstd", "profstd",
    "traditional_nborsstd", "traditional_prevadoptstd", "traditional_complexity",
    "traditional_igrole", "traditional_regov", "traditional_unified",
    "traditional_perdemstd", "traditional_incpcadjstd", "traditional_exppcadjstd",
    "traditional_logpopstd", "traditional_collegstd", "traditional_perurbanstd",
    "traditional_profstd"
]

karch_2016 = karch_2016_full[["adopt", "state", "year"] + covariates].dropna()

# Ensure year column is an integer
karch_2016['year'] = karch_2016['year'].astype(int)

karch_2016 = karch_2016.sort_values(["state", "year"])

# Create count variable (0 for first year, 1 for second year, etc.)
karch_2016['count'] = karch_2016['year'] - karch_2016['year'].min()

# Get year range
min_year = karch_2016['year'].min()
max_year = karch_2016['year'].max()
mid_year = min_year + (max_year - min_year) // 2

os.chdir("ml_forecast")

#--------------------------------------------------------------------------------------------------------

### Rolling Window t+5

# Initialize storage for results
results = {
    'original': {'ap_score': []},
    'logit': {'ap_score': []},
}

# Rolling window forecasting
for train_end_year in range(mid_year, max_year - 4):
    val_year = train_end_year + 5
    test_year = train_end_year + 6

    print(f"Training on years {min_year}-{train_end_year}, validation year {val_year}, predicting year {test_year}")
    
    # Split data
    train_data = karch_2016[karch_2016['year'] <= train_end_year]
    val_data = karch_2016[karch_2016['year'] == val_year]
    test_data = karch_2016[karch_2016['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['adopt', 'year', 'state']) 
    y_train = train_data['adopt']
    X_val = val_data.drop(columns = ['adopt', 'year', 'state'])
    y_val = val_data['adopt']
    X_test = test_data.drop(columns = ['adopt', 'year', 'state'])
    y_test = test_data['adopt']
    
    # Combine train and validation for sklearn GridSearchCV
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Create custom validation split indices
    train_indices = list(range(len(X_train)))
    val_indices = list(range(len(X_train), len(X_train_val)))
    cv_split = [(train_indices, val_indices)]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_val_scaled = scaler.transform(X_train_val)

    # Original Logit
    original_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_scores = original_model.predict_proba(X_test_scaled)[:, 1]
    
    results['original']['ap_score'].append(average_precision_score(y_test, original_scores))
    
    # Logistic Regression
    common_params = {
        'C': [0.001, 0.01, 0.1],
        'class_weight': [None, 'balanced'],
        'fit_intercept': [True]
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
        estimator = linear_model.LogisticRegression(max_iter = 2000, random_state = 1337),
        param_grid = param_grid,
        cv = cv_split,
        scoring = 'average_precision',
        n_jobs = -1,
        verbose = 0,
        refit = True
    )

    # Fit grid search
    grid_search.fit(X_train_val_scaled, y_train_val)

    # Get the best model and score on test set
    best_model = grid_search.best_estimator_
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    ap_score = average_precision_score(y_test, test_scores)
    print(f"Logistic Regression AP Score: {ap_score}")
    
    results['logit']['ap_score'].append(ap_score)

# Save aggregated results
with open("figures/karch2016/t5_forecast_results_logit.txt", "w") as f:
    for model in ['original', 'logit']:
        f.write(f"\n{model.upper()} Results:\n")
        f.write(f"Average AP Score: {np.mean(results[model]['ap_score']):.4f} (±{np.std(results[model]['ap_score']):.4f})\n")

# Plot time series of results from t+5 rolling window
years = list(range(mid_year + 6, mid_year + 6 + len(results['original']['ap_score'])))

plt.figure(figsize = (8, 6))

# AP Score Over Time
plt.plot(years, results['original']['ap_score'], marker = 'o', label = 'Original Logit')
plt.plot(years, results['logit']['ap_score'], marker = 'o', label = 'Logit')
plt.title('Average Precision Score Over Time (t+5 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('AP Score')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('figures/karch2016/t5_forecast_timeseries_logit.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save CSV
time_series_results = pd.DataFrame({
    'year': years,
    'original_ap_score': results['original']['ap_score'],
    'logit_ap_score': results['logit']['ap_score'],
})

time_series_results.to_csv('figures/karch2016/t5_forecast_timeseries_logit.csv', index = False)