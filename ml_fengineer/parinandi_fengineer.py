### Preprocessing Parinandi 2020

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
import os

warnings.filterwarnings('ignore')

random.seed(1337)

# Data
parinandi_2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year"
]

parinandi_2020 = parinandi_2020_full[["oneemulation"] + covariates].dropna()

# Define X and y
X = parinandi_2020.drop(columns = ['oneemulation']).copy()
y = parinandi_2020['oneemulation']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize storage for results
results = {
    'logit': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'rf': {'f1': [], 'balanced_acc': [], 'ap_score': []},
    'xgb': {'f1': [], 'balanced_acc': [], 'ap_score': []}
}

os.chdir("ml_fengineer")

# Logistic Regression
logit_model = linear_model.LogisticRegression(
    C = 0.001, 
    class_weight = None, 
    fit_intercept = True,
    penalty = None, 
    solver = 'lbfgs', 
    max_iter = 2500, 
    random_state = 1337
)

logit_model.fit(X_train_scaled, y_train)
logit_pred = logit_model.predict(X_test_scaled)
logit_scores = logit_model.predict_proba(X_test_scaled)[:, 1]
    
results['logit']['f1'].append(f1_score(y_test, logit_pred, average = "binary"))
results['logit']['balanced_acc'].append(balanced_accuracy_score(y_test, logit_pred))
results['logit']['ap_score'].append(average_precision_score(y_test, logit_scores))

# Generate classification report for logistic regression
logit_report = classification_report(y_test, logit_pred)
with open('figures/parinandi2020/logit_no_featureyear.txt', 'w') as f:
    f.write("Logistic Regression Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(logit_report)

# Generate PR curve and AUC-PR for logistic regression
precision, recall, _ = precision_recall_curve(y_test, logit_scores)
auc_pr = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label = f'Logistic Regression (AUC-PR = {auc_pr:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/logit_no_featureyear_pr_curve.png', dpi = 300, bbox_inches = 'tight')
plt.close()

# Random Forest
rf_model = RandomForestClassifier(
    bootstrap = True, 
    ccp_alpha = 0.0, 
    class_weight = None, 
    criterion = 'gini',
    max_depth = 10, 
    max_features = 'log2', 
    max_leaf_nodes = None, 
    max_samples = None,
    min_samples_leaf = 1,
    min_samples_split = 2, 
    n_estimators = 300, 
    random_state = 1337
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_scores = rf_model.predict_proba(X_test_scaled)[:, 1]

results['rf']['f1'].append(f1_score(y_test, rf_pred, average = "binary"))
results['rf']['balanced_acc'].append(balanced_accuracy_score(y_test, rf_pred))
results['rf']['ap_score'].append(average_precision_score(y_test, rf_scores))

# Generate classification report for Random Forest
rf_report = classification_report(y_test, rf_pred)
with open('figures/parinandi2020/rf_no_featureyear.txt', 'w') as f:
    f.write("Random Forest Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(rf_report)

# Generate PR curve and AUC-PR for Random Forest
precision, recall, _ = precision_recall_curve(y_test, rf_scores)
auc_pr = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label = f'Random Forest (AUC-PR = {auc_pr:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/rf_no_featureyear_pr_curve.png', dpi = 300, bbox_inches = 'tight')
plt.close()

# XGBoost
xgb_model = XGBClassifier(
    booster = 'gbtree', 
    colsample_bytree = 1.0, 
    eval_metric = 'error', 
    gamma = 0,
    grow_policy = 'depthwise', 
    learning_rate = 0.3, 
    max_bin = 128, 
    max_depth = 3,
    max_leaves = 16, 
    min_child_weight = 5, 
    n_estimators = 300, 
    objective = 'binary:logistic',
    reg_alpha = 0, 
    reg_lambda = 1, 
    scale_pos_weight = 1, 
    subsample = 0.7632039015158524,
    tree_method = 'exact', 
    random_state = 1337
)

xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]

results['xgb']['f1'].append(f1_score(y_test, xgb_pred, average = "binary"))
results['xgb']['balanced_acc'].append(balanced_accuracy_score(y_test, xgb_pred))
results['xgb']['ap_score'].append(average_precision_score(y_test, xgb_scores))

# Generate classification report for XGBoost
xgb_report = classification_report(y_test, xgb_pred)
with open('figures/parinandi2020/xgb_no_featureyear.txt', 'w') as f:
    f.write("XGBoost Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(xgb_report)

# Generate PR curve and AUC-PR for XGBoost
precision, recall, _ = precision_recall_curve(y_test, xgb_scores)
auc_pr = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label = f'XGBoost (AUC-PR = {auc_pr:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBoost')
plt.legend()
plt.grid(True)
plt.savefig('figures/parinandi2020/xgb_no_featureyear.png', dpi = 300, bbox_inches = 'tight')
plt.close()

# Save Results
with open('figures/parinandi2020/parinandi_no_featureyear_results.txt', 'w') as f:
    f.write("Model Performance Results\n")
    f.write("=" * 40 + "\n\n")
    
    for model in results:
        f.write(f"{model.upper()} Model:\n")
        for metric in results[model]:
            f.write(f"  {metric}: {results[model][metric][0]:.4f}\n")
        f.write("\n")