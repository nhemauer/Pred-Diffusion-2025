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
mallinson_lovell2022_full = pd.read_csv(r"data/mallinson_lovell2022.csv")

# Covariates
covariates = ["republican","legprof_squire","exp_pupil10000_adj","mathscore4th","readscore4th",
              "time"]
mallinson_lovell2022 = mallinson_lovell2022_full[["adopt"] + covariates].dropna()

# Define X and y
X = mallinson_lovell2022.drop(columns = ['adopt']).copy()
y = mallinson_lovell2022['adopt']

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
with open("figures/mallinson_lovell2022/optimized_logit_lovell_threshold.txt", "w") as f:
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
plt.savefig('figures/mallinson_lovell2022/optimized_logit_lovell_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Reglogit

# Use best hyperparameters from the random-split hyperparameter search
reglogit_model = linear_model.LogisticRegression(
    C = 0.1,
    class_weight = 'balanced',
    fit_intercept = True,
    l1_ratio = None,
    penalty = 'l2',
    solver = 'saga',
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
with open("figures/mallinson_lovell2022/optimized_reglogit_lovell_threshold.txt", "w") as f:
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
plt.savefig('figures/mallinson_lovell2022/optimized_reglogit_lovell_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### RF

# Use best hyperparameters from the random-split hyperparameter search
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.006076491497524607,
    class_weight = None,
    criterion = 'log_loss',
    max_depth = 12,
    max_samples = 0.5048236241274381,
    n_estimators = 100,
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
with open("figures/mallinson_lovell2022/optimized_rf_lovell_threshold.txt", "w") as f:
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
plt.savefig('figures/mallinson_lovell2022/optimized_rf_lovell_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### XGBoost

# Use best hyperparameters from the random-split hyperparameter search
xgb_model = XGBClassifier(
    booster = 'gbtree',
    eval_metric = 'aucpr',
    gamma = 2,
    grow_policy = 'depthwise',
    learning_rate = 0.04787337145112113,
    max_bin = 128,
    max_depth = 3,
    n_estimators = 100,
    objective = 'binary:logistic',
    reg_alpha = 0,
    scale_pos_weight = 1,
    subsample = 0.8012137562136856,
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
with open("figures/mallinson_lovell2022/optimized_xgb_lovell_threshold.txt", "w") as f:
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
plt.savefig('figures/mallinson_lovell2022/optimized_xgb_lovell_threshold.png', dpi = 300, bbox_inches = 'tight')
plt.show()