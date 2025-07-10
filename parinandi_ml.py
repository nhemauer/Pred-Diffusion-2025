### Preprocessing Parinandi 2020

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import warnings

warnings.filterwarnings('ignore')

random.seed(1337)

# Data
parinandi2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]

parinandi2020 = parinandi2020_full[["oneemulation"] + covariates].dropna()

# Define X and y
X = parinandi2020.drop(columns = ['oneemulation']).copy()
y = parinandi2020['oneemulation']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 Logistic (No Optimization)

# Fit
logistic = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

start_time = time.time()
logistic.fit(X_train_scaled, y_train)
end_time = time.time()

# Predict
y_pred = logistic.predict(X_test_scaled)

# Evaluation
f1_macro = f1_score(y_test, y_pred, average = "macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/unoptimized_logistic_parinandi.txt", "w") as f:
    f.write(f"F1 Macro Score: {f1_macro}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"Model Fit Elapsed Time: {(end_time - start_time) / 60:.2f} Minutes")

# Get predicted probabilities for the positive class
y_scores = logistic.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Unoptimized Precision-Recall Curve (Logistic)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/unoptimized_logistic_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 Regularized Logistic (Optimized)

# Define parameter grid for Logistic Regression
# Base params common to all
common_params = {
    'C': [0.001, 0.01, 0.1, 1, 2],
    'class_weight': [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}, {0: 1, 1: 8}, {0: 1, 1: 9}, {0: 1, 1: 10}],
    'fit_intercept': [True, False]
}

# Build full param grid
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
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # Only used if penalty = 'elasticnet', ignored otherwise
    }
]

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337),
    param_grid = param_grid,
    scoring = "f1", # F1 score good for maximizing precision and recall, but average_precision is better for balanced accuracy
    cv = 10,
    n_jobs = -1,
    verbose = 0,
    refit = True 
)

# Fit grid search
start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
end_time = time.time()

# Get the best model
best_model = grid_search.best_estimator_

# Predict with best estimator
y_pred = best_model.predict(X_test_scaled)

# Evaluation
f1_macro = f1_score(y_test, y_pred, average = "macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/optimized_logistic_parinandi.txt", "w") as f:
    f.write(f"Best Parameters Found: {grid_search.best_params_}\n")
    f.write(f"F1 Macro Score: {f1_macro}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"Model Fit Elapsed Time: {(end_time - start_time) / 60:.2f} Minutes")

# Get predicted probabilities for the positive class
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (Regularized Logistic)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/optimized_logistic_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 RF (No Optimization)

# Fit
random_forest = RandomForestClassifier(random_state = 1337)

start_time = time.time()
random_forest.fit(X_train_scaled, y_train)
end_time = time.time()

# Predict
y_pred = random_forest.predict(X_test_scaled)

# Evaluation
f1_macro = f1_score(y_test, y_pred, average = "macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/unoptimized_rf_parinandi.txt", "w") as f:
    f.write(f"F1 Macro Score: {f1_macro}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"Model Fit Elapsed Time: {(end_time - start_time) / 60:.2f} Minutes")

# Get predicted probabilities for the positive class
y_scores = random_forest.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Unoptimized Precision-Recall Curve (Random Forest)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/unoptimized_rf_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 RF (Optimized)

# Define the parameter search space for BayesSearchCV
param_grid = [
    {
        'n_estimators': (100, 300, 500),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': (None, 10, 25, 50),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'max_features': ['sqrt', 'log2', None],
        'max_leaf_nodes': (None, 10, 25, 50),
        'bootstrap': [True],
        'class_weight': [None, 'balanced'],
        'ccp_alpha': (0.0, 0.1, 'uniform'),
        'max_samples': (None, 0.5, 0.75)
    },
    {
        'n_estimators': (100, 300, 500),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': (None, 10, 25, 50),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'max_features': ['sqrt', 'log2', None],
        'max_leaf_nodes': (None, 10, 25, 50),
        'bootstrap': [False],
        'class_weight': [None, 'balanced'],
        'ccp_alpha': (0.0, 0.1, 'uniform'),
        'max_samples': [None]
    }
]

bayes_search = BayesSearchCV(
    estimator = RandomForestClassifier(random_state = 1337),
    search_spaces = param_grid,
    n_iter = 256,
    cv = 10,
    n_jobs = -1,
    verbose = 0,
    scoring = 'f1',
    random_state = 1337
)

start_time = time.time()
bayes_search.fit(X_train_scaled, y_train)
end_time = time.time()

# Get the best model
best_model = bayes_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test_scaled)

# Evaluation
f1_macro = f1_score(y_test, y_pred, average = "macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/optimized_rf_parinandi.txt", "w") as f:
    f.write(f"Best Parameters Found: {bayes_search.best_params_}\n")
    f.write(f"F1 Macro Score: {f1_macro}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"Model Fit Elapsed Time: {(end_time - start_time) / 60:.2f} Minutes")

# Get predicted probabilities for the positive class
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (Random Forest)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/optimized_rf_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 XGBoost (No Optimization)

# Fit
xgb = XGBClassifier(random_state = 1337, use_label_encoder = False, n_jobs = -1)

start_time = time.time()
xgb.fit(X_train_scaled, y_train)
end_time = time.time()

# Predict
y_pred = xgb.predict(X_test_scaled)

# Evaluation
f1_macro = f1_score(y_test, y_pred, average = "macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/unoptimized_xgboost_parinandi.txt", "w") as f:
    f.write(f"F1 Macro Score: {f1_macro}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"Model Fit Elapsed Time: {(end_time - start_time) / 60:.2f} Minutes")

# Get predicted probabilities for the positive class
y_scores = xgb.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Unoptimized Precision-Recall Curve (XGBoost)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/unoptimized_xgboost_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 XGBoost (Optimized)

# Define the parameter search space for BayesSearchCV
param_grid = {
    'n_estimators': (100, 300, 500),
    'max_depth': (3, 7, 15, 30),
    'max_bin': (16, 32, 64, 128),
    'booster': ['gbtree', 'dart'],
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss', 'auc', 'error', 'aucpr'],
    'tree_method': ['auto', 'exact', 'approx', 'hist'],
    'grow_policy': ['depthwise', 'lossguide'],
    'learning_rate': (0.001, 0.01, 0.1),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 2),
    'reg_alpha': (0, 2),
    'reg_lambda': (0, 2),
    'min_child_weight': (1, 5, 10),
    'max_leaves': (0, 16, 32),
    'scale_pos_weight': (1, 5, 10)
}

bayes_search = BayesSearchCV(
    estimator = XGBClassifier(random_state = 1337, use_label_encoder = False),
    search_spaces = param_grid,
    n_iter = 256,
    cv = 10,
    n_jobs = -1,
    verbose = 0,
    scoring = 'f1',
    random_state = 1337
)

start_time = time.time()
bayes_search.fit(X_train_scaled, y_train)
end_time = time.time()

# Get the best model
best_model = bayes_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test_scaled)

# Evaluation
f1_macro = f1_score(y_test, y_pred, average = "macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/optimized_xgboost_parinandi.txt", "w") as f:
    f.write(f"Best Parameters Found: {bayes_search.best_params_}\n")
    f.write(f"F1 Macro Score: {f1_macro}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"Model Fit Elapsed Time: {(end_time - start_time) / 60:.2f} Minutes")

# Get predicted probabilities for the positive class
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (XGBoost)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/optimized_xgboost_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()