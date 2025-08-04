### Preprocessing and Rolling Window t+1

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import warnings
import os

warnings.filterwarnings('ignore')
random.seed(1337)

# Data
bricker_lacombe_2021_full = pd.read_stata(r"data/bricker_lacombe2021.dta")

# Covariates
covariates = ["std_score","initiative","init_sigs","std_population",
                "std_citideology","unified","std_income","std_legp_squire",
                "duration","durationsq","durationcb"]
bricker_lacombe_2021 = bricker_lacombe_2021_full[["state", "year", "policy", "adoption"] + covariates].dropna()

bricker_lacombe_2021 = bricker_lacombe_2021.sort_values(["state", "year"])

# Get year range
min_year = bricker_lacombe_2021['year'].min()
max_year = bricker_lacombe_2021['year'].max()
mid_year = min_year + (max_year - min_year) // 2

# Initialize storage for results
results = {
    'logit': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'rf': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'xgb': {'f1': [], 'balanced_acc': [], 'ap_score': []}
}

os.chdir("ml_forecast")

# Rolling window forecasting
for train_end_year in range(mid_year, max_year):
    test_year = train_end_year + 1
    
    print(f"Training on years {min_year}-{train_end_year}, predicting year {test_year}")
    
    # Split data
    train_data = bricker_lacombe_2021[bricker_lacombe_2021['year'] <= train_end_year]
    test_data = bricker_lacombe_2021[bricker_lacombe_2021['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['adoption', 'state', 'year', 'policy'])
    y_train = train_data['adoption']
    X_test = test_data.drop(columns = ['adoption', 'state', 'year', 'policy'])
    y_test = test_data['adoption']

    # Prepare features
    X_train = train_data.drop(columns = ['adoption', 'state', 'policy'])
    X_test = test_data.drop(columns = ['adoption', 'state', 'policy'])
    
    # Create dummy variables for ALL possible years in the dataset
    all_years = sorted(bricker_lacombe_2021['year'].unique())
    
    # Create dummies for train set
    X_train = pd.get_dummies(X_train, columns = ['year'], drop_first = True)
    
    # Create dummies for test set
    X_test = pd.get_dummies(X_test, columns = ['year'], drop_first = True)
    
    # Ensure both have the same columns by reindexing
    all_columns = X_train.columns.union(X_test.columns)
    X_train = X_train.reindex(columns = all_columns, fill_value = 0)
    X_test = X_test.reindex(columns = all_columns, fill_value = 0)
    
    y_train = train_data['adoption']
    y_test = test_data['adoption']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    logit_model = linear_model.LogisticRegression(
        C = 0.1, 
        class_weight = {0: 1, 1: 3}, 
        fit_intercept = False,
        penalty = 'elasticnet', 
        solver = 'saga',
        l1_ratio = 0.5, 
        max_iter = 2500, 
        random_state = 1337
    )

    logit_model.fit(X_train_scaled, y_train)
    logit_pred = logit_model.predict(X_test_scaled)
    logit_scores = logit_model.predict_proba(X_test_scaled)[:, 1]
    
    results['logit']['f1'].append(f1_score(y_test, logit_pred, average = "binary"))
    results['logit']['balanced_acc'].append(balanced_accuracy_score(y_test, logit_pred))
    results['logit']['ap_score'].append(average_precision_score(y_test, logit_scores))
    
    # Random Forest
    rf_model = RandomForestClassifier(
        bootstrap = True, 
        ccp_alpha = 0.0, 
        class_weight = None, 
        criterion = 'log_loss',
        max_depth = 25, 
        max_features = 'sqrt', 
        max_leaf_nodes = None, 
        max_samples = 0.5,
        min_samples_leaf = 3,
        min_samples_split = 10, 
        n_estimators = 300, 
        random_state = 1337
    )

    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    results['rf']['f1'].append(f1_score(y_test, rf_pred, average = "binary"))
    results['rf']['balanced_acc'].append(balanced_accuracy_score(y_test, rf_pred))
    results['rf']['ap_score'].append(average_precision_score(y_test, rf_scores))
    
    # XGBoost
    xgb_model = XGBClassifier(
        booster = 'gbtree', 
        colsample_bytree = 0.5, 
        eval_metric = 'aucpr', 
        gamma = 0,
        grow_policy = 'lossguide', 
        learning_rate = 0.1, 
        max_bin = 32, 
        max_depth = 10,
        max_leaves = 32, 
        min_child_weight = 1, 
        n_estimators = 300, 
        objective = 'binary:logistic',
        reg_alpha = 0, 
        reg_lambda = 1, 
        scale_pos_weight = 1, 
        subsample = 0.9769109003040041,
        tree_method = 'approx', 
        random_state = 1337
    )

    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
    results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
    results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Save aggregated results
with open("figures/bricker_lacombe2021/t1_forecast_results.txt", "w") as f:
    for model in ['logit', 'rf', 'xgb']:
        f.write(f"\n{model.upper()} Results:\n")
        f.write(f"Average F1: {np.mean(results[model]['f1']):.4f} (±{np.std(results[model]['f1']):.4f})\n")
        f.write(f"Average Balanced Acc: {np.mean(results[model]['balanced_acc']):.4f} (±{np.std(results[model]['balanced_acc']):.4f})\n")
        f.write(f"Average AP Score: {np.mean(results[model]['ap_score']):.4f} (±{np.std(results[model]['ap_score']):.4f})\n")

# Plot time series of results from t+1 rolling window
years = list(range(mid_year + 1, max_year + 1))

plt.figure(figsize = (15, 5))

# F1 Score Over Time
plt.subplot(1, 3, 1)
plt.plot(years, results['logit']['f1'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['f1'], marker = 's', label = 'Random Forest') 
plt.plot(years, results['xgb']['f1'], marker = '^', label = 'XGBoost')
plt.title('F1 Score Over Time (t+1 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, alpha = 0.3)

# Balanced Accuracy Over Time
plt.subplot(1, 3, 2)
plt.plot(years, results['logit']['balanced_acc'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['balanced_acc'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['balanced_acc'], marker = '^', label = 'XGBoost')
plt.title('Balanced Accuracy Over Time (t+1 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.grid(True, alpha = 0.3)

# AP Score Over Time
plt.subplot(1, 3, 3)
plt.plot(years, results['logit']['ap_score'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['ap_score'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['ap_score'], marker = '^', label = 'XGBoost')
plt.title('Average Precision Score Over Time (t+1 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('AP Score')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('figures/bricker_lacombe2021/t1_forecast_timeseries.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save CSV
time_series_results = pd.DataFrame({
    'year': years,
    'logit_f1': results['logit']['f1'],
    'logit_balanced_acc': results['logit']['balanced_acc'],
    'logit_ap_score': results['logit']['ap_score'],
    'rf_f1': results['rf']['f1'],
    'rf_balanced_acc': results['rf']['balanced_acc'],
    'rf_ap_score': results['rf']['ap_score'],
    'xgb_f1': results['xgb']['f1'],
    'xgb_balanced_acc': results['xgb']['balanced_acc'],
    'xgb_ap_score': results['xgb']['ap_score']
})

time_series_results.to_csv('figures/bricker_lacombe2021/t1_forecast_timeseries.csv', index = False)

#--------------------------------------------------------------------------------------------------------

### Rolling Window t+5

# Initialize storage for results
results = {
    'logit': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'rf': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'xgb': {'f1': [], 'balanced_acc': [], 'ap_score': []}
}

# Rolling window forecasting
for train_end_year in range(mid_year, max_year - 4):
    test_year = train_end_year + 5
    
    print(f"Training on years {min_year}-{train_end_year}, predicting year {test_year}")
    
    # Split data
    train_data = bricker_lacombe_2021[bricker_lacombe_2021['year'] <= train_end_year]
    test_data = bricker_lacombe_2021[bricker_lacombe_2021['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['dvadopt', 'state', 'year'])
    y_train = train_data['dvadopt']
    X_test = test_data.drop(columns = ['dvadopt', 'state', 'year'])
    y_test = test_data['dvadopt']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    logit_model = linear_model.LogisticRegression(
        C = 0.1, 
        class_weight = {0: 1, 1: 3}, 
        fit_intercept = False,
        penalty = 'elasticnet', 
        solver = 'saga',
        l1_ratio = 0.5, 
        max_iter = 2500, 
        random_state = 1337
    )

    logit_model.fit(X_train_scaled, y_train)
    logit_pred = logit_model.predict(X_test_scaled)
    logit_scores = logit_model.predict_proba(X_test_scaled)[:, 1]
    
    results['logit']['f1'].append(f1_score(y_test, logit_pred, average = "binary"))
    results['logit']['balanced_acc'].append(balanced_accuracy_score(y_test, logit_pred))
    results['logit']['ap_score'].append(average_precision_score(y_test, logit_scores))
    
    # Random Forest
    rf_model = RandomForestClassifier(
        bootstrap = True, 
        ccp_alpha = 0.0, 
        class_weight = None, 
        criterion = 'log_loss',
        max_depth = 25, 
        max_features = 'sqrt', 
        max_leaf_nodes = None, 
        max_samples = 0.5,
        min_samples_leaf = 3,
        min_samples_split = 10, 
        n_estimators = 300, 
        random_state = 1337
    )

    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    results['rf']['f1'].append(f1_score(y_test, rf_pred, average = "binary"))
    results['rf']['balanced_acc'].append(balanced_accuracy_score(y_test, rf_pred))
    results['rf']['ap_score'].append(average_precision_score(y_test, rf_scores))
    
    # XGBoost
    xgb_model = XGBClassifier(
        booster = 'gbtree', 
        colsample_bytree = 0.5, 
        eval_metric = 'aucpr', 
        gamma = 0,
        grow_policy = 'lossguide', 
        learning_rate = 0.1, 
        max_bin = 32, 
        max_depth = 10,
        max_leaves = 32, 
        min_child_weight = 1, 
        n_estimators = 300, 
        objective = 'binary:logistic',
        reg_alpha = 0, 
        reg_lambda = 1, 
        scale_pos_weight = 1, 
        subsample = 0.9769109003040041,
        tree_method = 'approx', 
        random_state = 1337
    )

    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
    results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
    results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Save aggregated results
with open("figures/bricker_lacombe2021/t5_forecast_results.txt", "w") as f:
    for model in ['logit', 'rf', 'xgb']:
        f.write(f"\n{model.upper()} Results:\n")
        f.write(f"Average F1: {np.mean(results[model]['f1']):.4f} (±{np.std(results[model]['f1']):.4f})\n")
        f.write(f"Average Balanced Acc: {np.mean(results[model]['balanced_acc']):.4f} (±{np.std(results[model]['balanced_acc']):.4f})\n")
        f.write(f"Average AP Score: {np.mean(results[model]['ap_score']):.4f} (±{np.std(results[model]['ap_score']):.4f})\n")

# Plot time series of results from t+5 rolling window
years = list(range(mid_year + 5, max_year + 1))

plt.figure(figsize = (15, 5))

# F1 Score Over Time
plt.subplot(1, 3, 1)
plt.plot(years, results['logit']['f1'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['f1'], marker = 's', label = 'Random Forest') 
plt.plot(years, results['xgb']['f1'], marker = '^', label = 'XGBoost')
plt.title('F1 Score Over Time (t+5 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, alpha = 0.3)

# Balanced Accuracy Over Time
plt.subplot(1, 3, 2)
plt.plot(years, results['logit']['balanced_acc'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['balanced_acc'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['balanced_acc'], marker = '^', label = 'XGBoost')
plt.title('Balanced Accuracy Over Time (t+5 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.grid(True, alpha = 0.3)

# AP Score Over Time
plt.subplot(1, 3, 3)
plt.plot(years, results['logit']['ap_score'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['ap_score'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['ap_score'], marker = '^', label = 'XGBoost')
plt.title('Average Precision Score Over Time (t+5 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('AP Score')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('figures/bricker_lacombe2021/t5_forecast_timeseries.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save CSV
time_series_results = pd.DataFrame({
    'year': years,
    'logit_f1': results['logit']['f1'],
    'logit_balanced_acc': results['logit']['balanced_acc'],
    'logit_ap_score': results['logit']['ap_score'],
    'rf_f1': results['rf']['f1'],
    'rf_balanced_acc': results['rf']['balanced_acc'],
    'rf_ap_score': results['rf']['ap_score'],
    'xgb_f1': results['xgb']['f1'],
    'xgb_balanced_acc': results['xgb']['balanced_acc'],
    'xgb_ap_score': results['xgb']['ap_score']
})

time_series_results.to_csv('figures/bricker_lacombe2021/t5_forecast_timeseries.csv', index = False)

#--------------------------------------------------------------------------------------------------------

### Rolling Window t+10

# Initialize storage for results
results = {
    'logit': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'rf': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'xgb': {'f1': [], 'balanced_acc': [], 'ap_score': []}
}

# Rolling window forecasting
for train_end_year in range(mid_year, max_year - 9):
    test_year = train_end_year + 10
    
    print(f"Training on years {min_year}-{train_end_year}, predicting year {test_year}")
    
    # Split data
    train_data = bricker_lacombe_2021[bricker_lacombe_2021['year'] <= train_end_year]
    test_data = bricker_lacombe_2021[bricker_lacombe_2021['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['dvadopt', 'state', 'year'])
    y_train = train_data['dvadopt']
    X_test = test_data.drop(columns = ['dvadopt', 'state', 'year'])
    y_test = test_data['dvadopt']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    logit_model = linear_model.LogisticRegression(
        C = 0.1, 
        class_weight = {0: 1, 1: 3}, 
        fit_intercept = False,
        penalty = 'elasticnet', 
        solver = 'saga',
        l1_ratio = 0.5, 
        max_iter = 2500, 
        random_state = 1337
    )

    logit_model.fit(X_train_scaled, y_train)
    logit_pred = logit_model.predict(X_test_scaled)
    logit_scores = logit_model.predict_proba(X_test_scaled)[:, 1]
    
    results['logit']['f1'].append(f1_score(y_test, logit_pred, average = "binary"))
    results['logit']['balanced_acc'].append(balanced_accuracy_score(y_test, logit_pred))
    results['logit']['ap_score'].append(average_precision_score(y_test, logit_scores))
    
    # Random Forest
    rf_model = RandomForestClassifier(
        bootstrap = True, 
        ccp_alpha = 0.0, 
        class_weight = None, 
        criterion = 'log_loss',
        max_depth = 25, 
        max_features = 'sqrt', 
        max_leaf_nodes = None, 
        max_samples = 0.5,
        min_samples_leaf = 3,
        min_samples_split = 10, 
        n_estimators = 300, 
        random_state = 1337
    )

    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    results['rf']['f1'].append(f1_score(y_test, rf_pred, average = "binary"))
    results['rf']['balanced_acc'].append(balanced_accuracy_score(y_test, rf_pred))
    results['rf']['ap_score'].append(average_precision_score(y_test, rf_scores))
    
    # XGBoost
    xgb_model = XGBClassifier(
        booster = 'gbtree', 
        colsample_bytree = 0.5, 
        eval_metric = 'aucpr', 
        gamma = 0,
        grow_policy = 'lossguide', 
        learning_rate = 0.1, 
        max_bin = 32, 
        max_depth = 10,
        max_leaves = 32, 
        min_child_weight = 1, 
        n_estimators = 300, 
        objective = 'binary:logistic',
        reg_alpha = 0, 
        reg_lambda = 1, 
        scale_pos_weight = 1, 
        subsample = 0.9769109003040041,
        tree_method = 'approx', 
        random_state = 1337
    )

    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
    results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
    results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Save aggregated results
with open("figures/bricker_lacombe2021/t10_forecast_results.txt", "w") as f:
    for model in ['logit', 'rf', 'xgb']:
        f.write(f"\n{model.upper()} Results:\n")
        f.write(f"Average F1: {np.mean(results[model]['f1']):.4f} (±{np.std(results[model]['f1']):.4f})\n")
        f.write(f"Average Balanced Acc: {np.mean(results[model]['balanced_acc']):.4f} (±{np.std(results[model]['balanced_acc']):.4f})\n")
        f.write(f"Average AP Score: {np.mean(results[model]['ap_score']):.4f} (±{np.std(results[model]['ap_score']):.4f})\n")

# Plot time series of results from t+10 rolling window
years = list(range(mid_year + 10, max_year + 1))

plt.figure(figsize = (15, 5))

# F1 Score Over Time
plt.subplot(1, 3, 1)
plt.plot(years, results['logit']['f1'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['f1'], marker = 's', label = 'Random Forest') 
plt.plot(years, results['xgb']['f1'], marker = '^', label = 'XGBoost')
plt.title('F1 Score Over Time (t+10 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, alpha = 0.3)

# Balanced Accuracy Over Time
plt.subplot(1, 3, 2)
plt.plot(years, results['logit']['balanced_acc'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['balanced_acc'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['balanced_acc'], marker = '^', label = 'XGBoost')
plt.title('Balanced Accuracy Over Time (t+10 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.grid(True, alpha = 0.3)

# AP Score Over Time
plt.subplot(1, 3, 3)
plt.plot(years, results['logit']['ap_score'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['ap_score'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['ap_score'], marker = '^', label = 'XGBoost')
plt.title('Average Precision Score Over Time (t+10 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('AP Score')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('figures/bricker_lacombe2021/t10_forecast_timeseries.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save CSV
time_series_results = pd.DataFrame({
    'year': years,
    'logit_f1': results['logit']['f1'],
    'logit_balanced_acc': results['logit']['balanced_acc'],
    'logit_ap_score': results['logit']['ap_score'],
    'rf_f1': results['rf']['f1'],
    'rf_balanced_acc': results['rf']['balanced_acc'],
    'rf_ap_score': results['rf']['ap_score'],
    'xgb_f1': results['xgb']['f1'],
    'xgb_balanced_acc': results['xgb']['balanced_acc'],
    'xgb_ap_score': results['xgb']['ap_score']
})

time_series_results.to_csv('figures/bricker_lacombe2021/t10_forecast_timeseries.csv', index = False)