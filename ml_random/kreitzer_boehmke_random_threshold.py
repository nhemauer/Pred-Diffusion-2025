import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

random.seed(1337)

# Data
kreitzer_boehmke_2016_full = pd.read_stata(r"data/kreitzer_boehmke2016.dta")

covariates = [
    "norrander_legality", "religadhrate", "initdif", "dem_gov", "uni_dem_leg",
    "fem_dem", "nbrspct", "rescaledmedincome", "rescaledpopsize", "time", 
    "time2", "webster", "policy_num"
]

kreitzer_boehmke_2016 = kreitzer_boehmke_2016_full[["adopt_policy", "state"] + covariates].dropna()

# Define X and y
X = kreitzer_boehmke_2016.drop(columns = ['adopt_policy', 'state']).copy()
X = pd.get_dummies(X, columns = ['policy_num'], drop_first = True)
y = kreitzer_boehmke_2016['adopt_policy']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------

os.chdir("ml_random")

### Logit

logit_model = linear_model.LogisticRegression(max_iter = 2500, random_state = 1337)

# Fit model
logit_model.fit(X_train_scaled, y_train)

# Find optimal threshold on training data
train_probs = logit_model.predict_proba(X_train_scaled)[:, 1]
thresholds_search = np.arange(0.01, 0.50, 0.01)
f1_scores = [f1_score(y_train, train_probs >= t) for t in thresholds_search]
best_threshold = thresholds_search[np.argmax(f1_scores)]

# Predict using optimal threshold
y_scores = logit_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_scores >= best_threshold).astype(int)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/kreitzer_boehmke2016/optimized_logit_kreitzer_threshold.txt", "w") as f:
    f.write(f"Optimal Threshold: {best_threshold:.2f}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (Logit)')
plt.legend()
plt.grid(True)
plt.savefig('figures/kreitzer_boehmke2016/optimized_logit_kreitzer_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Reglogit

# Use best hyperparameters from the random-split hyperparameter search
reglogit_model = linear_model.LogisticRegression(
    C = 0.1,
    class_weight = None,
    fit_intercept = True,
    penalty = 'l2',
    solver = 'newton-cholesky',
    max_iter = 2500,
    random_state = 1337
)

# Fit model
reglogit_model.fit(X_train_scaled, y_train)

# Find optimal threshold on training data
train_probs = reglogit_model.predict_proba(X_train_scaled)[:, 1]
thresholds_search = np.arange(0.01, 0.50, 0.01)
f1_scores = [f1_score(y_train, train_probs >= t) for t in thresholds_search]
best_threshold = thresholds_search[np.argmax(f1_scores)]

# Predict using optimal threshold
y_scores = reglogit_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_scores >= best_threshold).astype(int)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/kreitzer_boehmke2016/optimized_reglogit_kreitzer_threshold.txt", "w") as f:
    f.write(f"Optimal Threshold: {best_threshold:.2f}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (Regularized Logit)')
plt.legend()
plt.grid(True)
plt.savefig('figures/kreitzer_boehmke2016/optimized_reglogit_kreitzer_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### RF

# Use best hyperparameters from the random-split hyperparameter search
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0,
    class_weight = None,
    criterion = 'entropy',
    max_depth = None,
    min_samples_leaf = 4,
    min_samples_split = 2,
    n_estimators = 171,
    random_state = 1337
)

# Fit model
rf_model.fit(X_train_scaled, y_train)

# Find optimal threshold on training data
train_probs = rf_model.predict_proba(X_train_scaled)[:, 1]
thresholds_search = np.arange(0.01, 0.50, 0.01)
f1_scores = [f1_score(y_train, train_probs >= t) for t in thresholds_search]
best_threshold = thresholds_search[np.argmax(f1_scores)]

# Predict using optimal threshold
y_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_scores >= best_threshold).astype(int)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/kreitzer_boehmke2016/optimized_rf_kreitzer_threshold.txt", "w") as f:
    f.write(f"Optimal Threshold: {best_threshold:.2f}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (Random Forest)')
plt.legend()
plt.grid(True)
plt.savefig('figures/kreitzer_boehmke2016/optimized_rf_kreitzer_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### XGBoost

# Use best hyperparameters from the random-split hyperparameter search
xgb_model = XGBClassifier(
    booster = 'dart',
    eval_metric = 'aucpr',
    grow_policy = 'depthwise',
    learning_rate = 0.1,
    max_bin = 256,
    max_depth = 6,
    min_child_weight = 7,
    n_estimators = 500,
    objective = 'binary:logistic',
    reg_alpha = 0,
    scale_pos_weight = 5,
    subsample = 1.0,
    tree_method = 'auto',
    random_state = 1337
)

# Fit model
xgb_model.fit(X_train_scaled, y_train)

# Find optimal threshold on training data
train_probs = xgb_model.predict_proba(X_train_scaled)[:, 1]
thresholds_search = np.arange(0.01, 0.50, 0.01)
f1_scores = [f1_score(y_train, train_probs >= t) for t in thresholds_search]
best_threshold = thresholds_search[np.argmax(f1_scores)]

# Predict using optimal threshold
y_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_scores >= best_threshold).astype(int)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/kreitzer_boehmke2016/optimized_xgb_kreitzer_threshold.txt", "w") as f:
    f.write(f"Optimal Threshold: {best_threshold:.2f}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized Precision-Recall Curve (XGBoost)')
plt.legend()
plt.grid(True)
plt.savefig('figures/kreitzer_boehmke2016/optimized_xgb_kreitzer_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()