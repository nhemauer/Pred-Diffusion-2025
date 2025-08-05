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
berry_berry1990_full = pd.read_csv("data/berry_berry1990.txt", delim_whitespace = True, header = None)
berry_berry1990_full.columns = ["state", "year", "adopt", "fiscal_1", "party", "elect1", "elect2", "income_1", "neighbor", "nbrpercn", "religion"]

berry_berry1990 = berry_berry1990_full[berry_berry1990_full['party'] != 9].copy() # 9 is the NA (For MN and NE)

berry_berry1990 = berry_berry1990.sort_values(["state", "year"])

# Get year range
min_year = berry_berry1990['year'].min()
max_year = berry_berry1990['year'].max()
mid_year = min_year + (max_year - min_year) // 2

# Initialize storage for results
results = {
    'original': {'f1': [], 'balanced_acc': [], 'ap_score': []},
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
    train_data = berry_berry1990[berry_berry1990['year'] <= train_end_year]
    test_data = berry_berry1990[berry_berry1990['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
    y_train = train_data['adopt']
    X_test = test_data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
    y_test = test_data['adopt']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Original Logit
    original_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_scores = original_model.predict_proba(X_test_scaled)[:, 1]
    
    results['original']['f1'].append(f1_score(y_test, original_pred, average = "binary"))
    results['original']['balanced_acc'].append(balanced_accuracy_score(y_test, original_pred))
    results['original']['ap_score'].append(average_precision_score(y_test, original_scores))
    
    # Logistic Regression
    logit_model = linear_model.LogisticRegression(
        C = 0.01, 
        class_weight = {0: 1, 1: 4}, 
        fit_intercept = True,
        penalty = 'l2', 
        solver = 'liblinear',
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
        bootstrap = False, 
        ccp_alpha = 0.023717780824102017, 
        class_weight = 'balanced', 
        criterion = 'gini',
        max_depth = 25, 
        max_features = 'sqrt', 
        max_leaf_nodes = None, 
        max_samples = None,
        min_samples_leaf = 2,
        min_samples_split = 10, 
        n_estimators = 500, 
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
        booster = 'dart', 
        colsample_bytree = 0.5, 
        eval_metric = 'auc', 
        gamma = 1,
        grow_policy = 'depthwise', 
        learning_rate = 0.01, 
        max_bin = 128, 
        max_depth = 20,
        max_leaves = 0, 
        min_child_weight = 5, 
        n_estimators = 300, 
        objective = 'binary:logistic',
        reg_alpha = 0, 
        reg_lambda = 2, 
        scale_pos_weight = 1, 
        subsample = 0.6683821824171767,
        tree_method = 'exact', 
        random_state = 1337
    )

    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
    results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
    results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Save aggregated results
with open("figures/berry_berry1990/t1_forecast_results.txt", "w") as f:
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
plt.savefig('figures/berry_berry1990/t1_forecast_timeseries.png', dpi = 300, bbox_inches = 'tight')
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

time_series_results.to_csv('figures/berry_berry1990/t1_forecast_timeseries.csv', index = False)

#--------------------------------------------------------------------------------------------------------

### Rolling Window t+5

# Initialize storage for results
results = {
    'original': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'logit': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'rf': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'xgb': {'f1': [], 'balanced_acc': [], 'ap_score': []}
}

# Rolling window forecasting
for train_end_year in range(mid_year, max_year - 4):
    test_year = train_end_year + 5
    
    print(f"Training on years {min_year}-{train_end_year}, predicting year {test_year}")
    
    # Split data
    train_data = berry_berry1990[berry_berry1990['year'] <= train_end_year]
    test_data = berry_berry1990[berry_berry1990['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
    y_train = train_data['adopt']
    X_test = test_data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
    y_test = test_data['adopt']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Original Logit
    original_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_scores = original_model.predict_proba(X_test_scaled)[:, 1]
    
    results['original']['f1'].append(f1_score(y_test, original_pred, average = "binary"))
    results['original']['balanced_acc'].append(balanced_accuracy_score(y_test, original_pred))
    results['original']['ap_score'].append(average_precision_score(y_test, original_scores))
    
    # Logistic Regression
    logit_model = linear_model.LogisticRegression(
        C = 0.01, 
        class_weight = {0: 1, 1: 4}, 
        fit_intercept = True,
        penalty = 'l2', 
        solver = 'liblinear',
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
        bootstrap = False, 
        ccp_alpha = 0.023717780824102017, 
        class_weight = 'balanced', 
        criterion = 'gini',
        max_depth = 25, 
        max_features = 'sqrt', 
        max_leaf_nodes = None, 
        max_samples = None,
        min_samples_leaf = 2,
        min_samples_split = 10, 
        n_estimators = 500, 
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
        booster = 'dart', 
        colsample_bytree = 0.5, 
        eval_metric = 'auc', 
        gamma = 1,
        grow_policy = 'depthwise', 
        learning_rate = 0.01, 
        max_bin = 128, 
        max_depth = 20,
        max_leaves = 0, 
        min_child_weight = 5, 
        n_estimators = 300, 
        objective = 'binary:logistic',
        reg_alpha = 0, 
        reg_lambda = 2, 
        scale_pos_weight = 1, 
        subsample = 0.6683821824171767,
        tree_method = 'exact', 
        random_state = 1337
    )

    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
    results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
    results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Save aggregated results
with open("figures/berry_berry1990/t5_forecast_results.txt", "w") as f:
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
plt.savefig('figures/berry_berry1990/t5_forecast_timeseries.png', dpi = 300, bbox_inches = 'tight')
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

time_series_results.to_csv('figures/berry_berry1990/t5_forecast_timeseries.csv', index = False)

#--------------------------------------------------------------------------------------------------------

### Rolling Window t+10

# Initialize storage for results
results = {
    'original': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'logit': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'rf': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'xgb': {'f1': [], 'balanced_acc': [], 'ap_score': []}
}

# Rolling window forecasting
for train_end_year in range(mid_year, max_year - 9):
    test_year = train_end_year + 10
    
    print(f"Training on years {min_year}-{train_end_year}, predicting year {test_year}")
    
    # Split data
    train_data = berry_berry1990[berry_berry1990['year'] <= train_end_year]
    test_data = berry_berry1990[berry_berry1990['year'] == test_year]
    
    if len(test_data) == 0:
        continue
    
    # Prepare features
    X_train = train_data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
    y_train = train_data['adopt']
    X_test = test_data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
    y_test = test_data['adopt']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Original Logit
    original_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_scores = original_model.predict_proba(X_test_scaled)[:, 1]
    
    results['original']['f1'].append(f1_score(y_test, original_pred, average = "binary"))
    results['original']['balanced_acc'].append(balanced_accuracy_score(y_test, original_pred))
    results['original']['ap_score'].append(average_precision_score(y_test, original_scores))
    
    # Logistic Regression
    logit_model = linear_model.LogisticRegression(
        C = 0.01, 
        class_weight = {0: 1, 1: 4}, 
        fit_intercept = True,
        penalty = 'l2', 
        solver = 'liblinear',
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
        bootstrap = False, 
        ccp_alpha = 0.023717780824102017, 
        class_weight = 'balanced', 
        criterion = 'gini',
        max_depth = 25, 
        max_features = 'sqrt', 
        max_leaf_nodes = None, 
        max_samples = None,
        min_samples_leaf = 2,
        min_samples_split = 10, 
        n_estimators = 500, 
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
        booster = 'dart', 
        colsample_bytree = 0.5, 
        eval_metric = 'auc', 
        gamma = 1,
        grow_policy = 'depthwise', 
        learning_rate = 0.01, 
        max_bin = 128, 
        max_depth = 20,
        max_leaves = 0, 
        min_child_weight = 5, 
        n_estimators = 300, 
        objective = 'binary:logistic',
        reg_alpha = 0, 
        reg_lambda = 2, 
        scale_pos_weight = 1, 
        subsample = 0.6683821824171767,
        tree_method = 'exact', 
        random_state = 1337
    )

    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
    results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
    results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Save aggregated results
with open("figures/berry_berry1990/t10_forecast_results.txt", "w") as f:
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
plt.plot(years, results['original']['f1'], marker = 'o', label = 'Original Logit')
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
plt.plot(years, results['original']['balanced_acc'], marker = 'o', label = 'Original Logit')
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
plt.plot(years, results['original']['ap_score'], marker = 'o', label = 'Original Logit')
plt.plot(years, results['logit']['ap_score'], marker = 'o', label = 'Logit')
plt.plot(years, results['rf']['ap_score'], marker = 's', label = 'Random Forest')
plt.plot(years, results['xgb']['ap_score'], marker = '^', label = 'XGBoost')
plt.title('Average Precision Score Over Time (t+10 Forecasting)')
plt.xlabel('Forecast Year')
plt.ylabel('AP Score')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('figures/berry_berry1990/t10_forecast_timeseries.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save CSV
time_series_results = pd.DataFrame({
    'year': years,
    'original_f1': results['original']['f1'],
    'original_balanced_acc': results['original']['balanced_acc'],
    'original_ap_score': results['original']['ap_score'],
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

time_series_results.to_csv('figures/berry_berry1990/t10_forecast_timeseries.csv', index = False)