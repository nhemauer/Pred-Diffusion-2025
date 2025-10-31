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

# Data
lacombe_boehmke2021_full = pd.read_stata(r"data/lacombe_boehmke2021.dta")

covariates = [
    "initiative", "init_sigs", "std_latnt_decay", "std_nbrs_lag", "std_population",
    "std_masssociallib_est", "unified", "duration", "durationsq", "durationcb", "std_income",
    "std_bowen_1", "std_bowen_2", "change_pop", "change_inc", "party_change"
]
lacombe_boehmke2021 = lacombe_boehmke2021_full[["adoption", "policyno", "year"] + covariates].dropna()

# Rename columns
variable_names = {
    "initiative": "Initiative Process",
    "init_sigs": "Signatures",
    "std_latnt_decay": "Latent Decay",
    "std_nbrs_lag": "Contiguity",
    "std_population": "Population",
    "std_masssociallib_est": "Public Liberalism",
    "unified": "Unified Control",
    "duration": "Duration",
    "durationsq": "Duration Squared",
    "durationcb": "Duration Cubed",
    "std_income": "Income per Capita",
    "std_bowen_1": "Legislative Prof. Dim. 1",
    "std_bowen_2": "Legislative Prof. Dim. 2",
    "change_pop": "Change Population",
    "change_inc": "Change Income",
    "party_change": "Change in Party"
}

lacombe_boehmke2021 = lacombe_boehmke2021.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Define X and y
X = lacombe_boehmke2021.drop(columns = ['adoption', 'policyno']).copy()
X = pd.get_dummies(X, columns = ['year'], drop_first = True)
y = lacombe_boehmke2021['adoption']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.chdir("ml_feature_importance")

# Use best hyperparameters from the random-split experiment
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 1.4673493976469224e-05,
    class_weight = None,
    criterion = 'entropy',
    max_depth = 50,
    min_samples_leaf = 2,
    n_estimators = 500,
    random_state = 1337
)

# Fit rf model
rf_model.fit(X_train_scaled, y_train)

# RF feature importance
feature_names = X_train.columns.tolist()
rf_feature_importance = rf_model.feature_importances_

# Use best hyperparameters from the random-split experiment
xgb_model = XGBClassifier(
    booster = 'dart',
    colsample_bytree = 0.5,
    eval_metric = 'aucpr',
    grow_policy = 'depthwise',
    learning_rate = 0.1,
    max_bin = 64,
    max_depth = 6,
    max_leaves = 32,
    min_child_weight = 5,
    n_estimators = 300,
    objective = 'binary:logistic',
    subsample = 0.9091013322130326,
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
plt.savefig('figures/lacombe_boehmke2021/lacombe_feature_importance_comparison.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/lacombe_boehmke2021/lacombe_feature_importance.csv', index = False)

# Create RF PDP with top 9 features
top_features_rf_all = importance_df.sort_values(by = 'rf_importance', ascending = False)
top_features_rf = top_features_rf_all[top_features_rf_all['feature'].isin(covariates_renamed)].head(9)['feature'].tolist()

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

plt.tight_layout()
plt.savefig('figures/lacombe_boehmke2021/lacombe_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Create XGBoost PDP with top 9 features
top_features_xgb_all = importance_df.sort_values(by = 'xgb_importance', ascending = False)
top_features_xgb = top_features_xgb_all[top_features_xgb_all['feature'].isin(covariates_renamed)].head(9)['feature'].tolist()

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

plt.tight_layout()
plt.savefig('figures/lacombe_boehmke2021/lacombe_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()