### Preprocessing t+1

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
import os

warnings.filterwarnings('ignore')

random.seed(1337)

# Data
parinandi2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]

parinandi2020 = parinandi2020_full[["oneemulation", "state"] + covariates].dropna()

# Sort by state and year
parinandi2020 = parinandi2020.sort_values(["state", "year"])

# Split by last year - use last year for testing, all prior years for training
last_year = parinandi2020['year'].max()
train_data = parinandi2020[parinandi2020['year'] < last_year]
test_data = parinandi2020[parinandi2020['year'] == last_year]

# Define X and y for train and test
X_train = train_data.drop(columns = ['oneemulation', 'state'])
y_train = train_data['oneemulation']
X_test = test_data.drop(columns = ['oneemulation', 'state'])
y_test = test_data['oneemulation']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------

os.chdir("ml_forecast")

### Parinandi 2020 Logit Forecast t+1

# Initialize regularized logistic regression with best parameters
regularized_logit = linear_model.LogisticRegression(
    C = 0.1,
    class_weight = {0: 1, 1: 10},
    fit_intercept = True,
    penalty = 'l2',
    solver = 'liblinear',
    max_iter = 2500,
    random_state = 1337
)

# Fit the model
regularized_logit.fit(X_train_scaled, y_train)

# Make predictions
y_pred = regularized_logit.predict(X_test_scaled)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/forecast1_logistic_parinandi.txt", "w") as f:
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Get predicted probabilities for the positive class
y_scores = regularized_logit.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Forecast t+1 (Regularized Logistic)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/forecast1_logistic_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 RF Forecast t+1

# Initialize Random Forest with best parameters
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0,
    class_weight = 'balanced',
    criterion = 'log_loss',
    max_depth = None,
    max_features = None,
    max_leaf_nodes = None,
    max_samples = 0.5,
    min_samples_leaf = 1,
    min_samples_split = 4,
    n_estimators = 500,
    random_state = 1337
)

# Fit the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/forecast1_rf_parinandi.txt", "w") as f:
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Get predicted probabilities for the positive class
y_scores = rf_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Forecast t+1 (Random Forest)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/forecast1_rf_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 XGBoost Forecast t+1

# Initialize XGBoost with best parameters
xgb_model = XGBClassifier(
    booster = 'gbtree',
    colsample_bytree = 0.5,
    eval_metric = 'auc',
    gamma = 2,
    grow_policy = 'lossguide',
    learning_rate = 0.1,
    max_bin = 16,
    max_depth = 20,
    max_leaves = 0,
    min_child_weight = 1,
    n_estimators = 500,
    objective = 'binary:logistic',
    reg_alpha = 2,
    reg_lambda = 1,
    scale_pos_weight = 5,
    subsample = 1.0,
    tree_method = 'auto',
    random_state = 1337
)

# Fit the model
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/forecast1_xgb_parinandi.txt", "w") as f:
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Get predicted probabilities for the positive class
y_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Forecast t+1 (XGBoost)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/forecast1_xgb_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Preprocessing t+5

parinandi2020 = parinandi2020_full[["oneemulation", "state"] + covariates].dropna()

# Sort by state and year
parinandi2020 = parinandi2020.sort_values(["state", "year"])

# Split for t+5: use last year for testing, exclude 4 years prior
last_year = parinandi2020['year'].max()
cutoff_year = last_year - 5  # Remove 4 years prior, so training data ends at last_year - 5

train_data = parinandi2020[parinandi2020['year'] <= cutoff_year]
test_data = parinandi2020[parinandi2020['year'] == last_year]

# Define X and y for train and test
X_train = train_data.drop(columns = ['oneemulation', 'state'])
y_train = train_data['oneemulation']
X_test = test_data.drop(columns = ['oneemulation', 'state'])
y_test = test_data['oneemulation']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 Logit Forecast t+5

# Initialize regularized logistic regression with best parameters
regularized_logit = linear_model.LogisticRegression(
    C = 0.1,
    class_weight = {0: 1, 1: 10},
    fit_intercept = True,
    penalty = 'l2',
    solver = 'liblinear',
    max_iter = 2500,
    random_state = 1337
)

# Fit the model
regularized_logit.fit(X_train_scaled, y_train)

# Make predictions
y_pred = regularized_logit.predict(X_test_scaled)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/forecast5_logistic_parinandi.txt", "w") as f:
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Get predicted probabilities for the positive class
y_scores = regularized_logit.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Forecast t+5 (Regularized Logistic)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/forecast5_logistic_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 RF Forecast t+5

# Initialize Random Forest with best parameters
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0,
    class_weight = 'balanced',
    criterion = 'log_loss',
    max_depth = None,
    max_features = None,
    max_leaf_nodes = None,
    max_samples = 0.5,
    min_samples_leaf = 1,
    min_samples_split = 4,
    n_estimators = 500,
    random_state = 1337
)

# Fit the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/forecast5_rf_parinandi.txt", "w") as f:
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Get predicted probabilities for the positive class
y_scores = rf_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Forecast t+5 (Random Forest)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/forecast5_rf_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------------------------------------------

### Parinandi 2020 XGBoost Forecast t+5

# Initialize XGBoost with best parameters
xgb_model = XGBClassifier(
    booster = 'gbtree',
    colsample_bytree = 0.5,
    eval_metric = 'auc',
    gamma = 2,
    grow_policy = 'lossguide',
    learning_rate = 0.1,
    max_bin = 16,
    max_depth = 20,
    max_leaves = 0,
    min_child_weight = 1,
    n_estimators = 500,
    objective = 'binary:logistic',
    reg_alpha = 2,
    reg_lambda = 1,
    scale_pos_weight = 5,
    subsample = 1.0,
    tree_method = 'auto',
    random_state = 1337
)

# Fit the model
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluation
f1 = f1_score(y_test, y_pred, average = "binary")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save metrics to file
with open("figures/parinandi2020/forecast5_xgb_parinandi.txt", "w") as f:
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Balanced Accuracy Score: {balanced_acc}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Get predicted probabilities for the positive class
y_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Compute average precision (AUC PR)
ap_score = average_precision_score(y_test, y_scores)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the PR curve
plt.figure(figsize = (7, 5))
plt.plot(recall, precision, label = f'AUC PR = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Forecast t+5 (XGBoost)\n(Parinandi 2020)')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/forecast5_xgb_parinandi.png', dpi = 300, bbox_inches = 'tight')
plt.show()