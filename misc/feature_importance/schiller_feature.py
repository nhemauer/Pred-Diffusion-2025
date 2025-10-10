import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

random.seed(1337)

# Data
schiller_sidorsky2022_full = pd.read_stata(r"data/schiller_sidorsky2022.dta")

covariates = [
    "gunhomicideslag1", "citizenideologylag1", "numregdvgunlawenactlag1", "vawa1994", "vawa1995", 
    "lautenbergamdt1996", "Lautenbergamndt1997", "legislature_election_year", "femleg", "innovation_index"
]

schiller_sidorsky2022 = schiller_sidorsky2022_full[["dvgunlaw"] + covariates].dropna()

# Define X and y
X = schiller_sidorsky2022.drop(columns = ['dvgunlaw']).copy()
y = schiller_sidorsky2022['dvgunlaw']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.chdir("misc/feature_importance")

# Use best hyperparameters from the random-split experiment
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0018207538196541142,
    class_weight = None,
    criterion = 'gini',
    max_depth = 21,
    max_leaf_nodes = 25,
    min_samples_leaf = 1,
    n_estimators = 100,
    random_state = 1337
)

# Fit rf model
rf_model.fit(X_train_scaled, y_train)

# RF feature importance
feature_names = X_train.columns.tolist()
rf_feature_importance = rf_model.feature_importances_

# Use best hyperparameters from the random-split experiment
xgb_model = XGBClassifier(
    booster = 'gbtree',
    eval_metric = 'aucpr',
    grow_policy = 'lossguide',
    learning_rate = 0.1,
    max_bin = 16,
    max_depth = 20,
    min_child_weight = 4,
    n_estimators = 100,
    objective = 'binary:logistic',
    reg_alpha = 0,
    scale_pos_weight = 6,
    subsample = 1.0,
    tree_method = 'auto',
    random_state = 1337
)

# Fit XGBoost model
xgb_model.fit(X_train_scaled, y_train)

# XGBoost feature importance
xgb_feature_importance = xgb_model.feature_importances_

# Initialize dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'rf_importance': rf_feature_importance,
    'xgb_importance': xgb_feature_importance})

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

# Plot rf feature importance
rf_top_features = importance_df.sort_values(by = 'rf_importance', ascending = False).head(20)
ax1.barh(range(len(rf_top_features)), rf_top_features['rf_importance'])
ax1.set_yticks(range(len(rf_top_features)))
ax1.set_yticklabels(rf_top_features['feature'])
ax1.set_xlabel('Feature Importance')
ax1.set_title('Top 20 Feature Importance - Random Forest')
ax1.invert_yaxis()

# Plot XGBoost feature importance
xgb_top_features = importance_df.sort_values(by = 'xgb_importance', ascending = False).head(20)
ax2.barh(range(len(xgb_top_features)), xgb_top_features['xgb_importance'])
ax2.set_yticks(range(len(xgb_top_features)))
ax2.set_yticklabels(xgb_top_features['feature'])
ax2.set_xlabel('Feature Importance')
ax2.set_title('Top 20 Feature Importance - XGBoost')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/schiller_sidorsky2022/schiller_feature_importance_comparison.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/schiller_sidorsky2022/schiller_feature_importance.csv', index = False)