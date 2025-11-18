import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

random.seed(1337)

os.chdir("ml_simulation")

# Data
berry_berry1990_full = pd.read_csv(r"figures/berry_berry1990/berry_berry_sim_data.csv")

covariates = ["fiscal_1", "party", "elect1", "elect2", "income_1", "nbrpercn", "religion"]
berry_berry1990 = berry_berry1990_full[["state", "year", "event"] + covariates].dropna()

# Rename columns
variable_names = {
    "fiscal_1": "Fiscal",
    "party": "Party", 
    "elect1": "Elect1",
    "elect2": "Elect2",
    "income_1": "Income",
    "nbrpercn": "Neighbors",
    "religion": "Religion"
}

berry_berry1990 = berry_berry1990.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Define X and y
X = berry_berry1990.drop(columns = ['adopt', 'neighbor', 'state', 'year']).copy()
y = berry_berry1990['adopt']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use best hyperparameters from the random-split experiment
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.07870049004782728,
    class_weight = 'balanced',
    criterion = 'gini',
    max_depth = 10,
    max_samples = 0.5,
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
    grow_policy = 'depthwise',
    learning_rate = 0.0885607150747851,
    max_bin = 64,
    max_depth = 20,
    max_leaves = 16,
    min_child_weight = 5,
    n_estimators = 500,
    objective = 'binary:logistic',
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

# Create Random Forest feature importance plot
fig1, ax1 = plt.subplots(1, 1, figsize = (10, 10))
rf_top_features = importance_df.sort_values(by = 'rf_importance', ascending = False).head(20)
ax1.barh(range(len(rf_top_features)), rf_top_features['rf_importance'])
ax1.set_yticks(range(len(rf_top_features)))
ax1.set_yticklabels(rf_top_features['feature'])
ax1.set_xlabel('Feature Importance')
ax1.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/berry_berry1990/berry_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Create XGBoost feature importance plot
fig2, ax2 = plt.subplots(1, 1, figsize = (10, 10))
xgb_top_features = importance_df.sort_values(by = 'xgb_importance', ascending = False).head(20)
ax2.barh(range(len(xgb_top_features)), xgb_top_features['xgb_importance'])
ax2.set_yticks(range(len(xgb_top_features)))
ax2.set_yticklabels(xgb_top_features['feature'])
ax2.set_xlabel('Feature Importance')
ax2.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/berry_berry1990/berry_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/berry_berry1990/berry_feature_importance.csv', index = False)

# Create RF PDP with the same features as the original feature importance
custom_rf_features = [
    "Income",
    "Neighbors",
    "Religion",
    "Fiscal",
    "Party",
    "Elect1",
    "Elect2",
]

top_features_rf = (
    importance_df.set_index('feature')
    .reindex(custom_rf_features)
    .dropna(subset = ['rf_importance'])
    .index.tolist()
)

# Create partial dependence plot
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(top_features_rf):
    feature_idx = feature_names.index(feature)
    display = PartialDependenceDisplay.from_estimator(
        rf_model, 
        X_train_scaled, 
        features = [feature_idx],
        feature_names = feature_names,
        ax = axes[i],
        kind = 'average',
        response_method = 'predict_proba'
    )
    axes[i].set_title(f'PDP: {feature}')
    display.axes_[0, 0].set_ylabel('Predicted Probability of Adoption')

# Hide unused subplots
for j in range(len(top_features_rf), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('figures/berry_berry1990/berry_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Create XGB PDP with the same features as the original feature importance
custom_xgb_features = [
    "Neighbors",
    "Religion",
    "Party",
    "Income",
    "Elect1",
    "Fiscal",
    "Elect2",
]

top_features_xgb = (
    importance_df.set_index('feature')
    .reindex(custom_xgb_features)
    .dropna(subset = ['xgb_importance'])
    .index.tolist()
)

# Create partial dependence plot
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(top_features_xgb):
    feature_idx = feature_names.index(feature)
    display = PartialDependenceDisplay.from_estimator(
        xgb_model, 
        X_train_scaled, 
        features = [feature_idx],
        feature_names = feature_names,
        ax = axes[i],
        kind = 'average',
        response_method = 'predict_proba'
    )
    axes[i].set_title(f'PDP: {feature}')
    display.axes_[0, 0].set_ylabel('Predicted Probability of Adoption')

# Hide unused subplots
for j in range(len(top_features_xgb), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('figures/berry_berry1990/berry_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()