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
kreitzer_boehmke_2016_full = pd.read_stata(r"data/kreitzer_boehmke2016.dta")

covariates = [
    "norrander_legality", "religadhrate", "initdif", "dem_gov", "uni_dem_leg",
    "fem_dem", "nbrspct", "rescaledmedincome", "rescaledpopsize", "time", 
    "time2", "webster"
]
kreitzer_boehmke_2016 = kreitzer_boehmke_2016_full[["adopt_policy", "state", 'policy_num'] + covariates].dropna()

# Rename columns
variable_names = {
    "norrander_legality": "Abortion Opinion",
    "religadhrate": "Religious Adherence",
    "initdif": "Initiative Difficulty",
    "dem_gov": "Democratic Governor",
    "uni_dem_leg": "Unified Dem. Legislature",
    "fem_dem": "Democratic Women",
    "nbrspct": "Neighbor Adoption %",
    "rescaledmedincome": "Median Income",
    "rescaledpopsize": "Population",
    "time": "Time",
    "time2": "Time Squared",
    "webster": "Post-Webster Indicator"
}

kreitzer_boehmke_2016 = kreitzer_boehmke_2016.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

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

os.chdir("ml_feature_importance")

# Use best hyperparameters from the random-split experiment
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

# Fit rf model
rf_model.fit(X_train_scaled, y_train)

# RF feature importance
feature_names = X_train.columns.tolist()
rf_feature_importance = rf_model.feature_importances_

# Use best hyperparameters from the random-split experiment
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

# Fit XGBoost model
xgb_model.fit(X_train_scaled, y_train)

# XGBoost feature importance
xgb_feature_importance = xgb_model.feature_importances_

# Initialize dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'rf_importance': rf_feature_importance,
    'xgb_importance': xgb_feature_importance})

# Filter to only include covariates (exclude fixed effects)
importance_df = importance_df[importance_df['feature'].isin(covariates_renamed)]

# Create Random Forest feature importance plot
fig1, ax1 = plt.subplots(1, 1, figsize = (10, 10))
rf_top_features = importance_df.sort_values(by = 'rf_importance', ascending = False).head(20)
ax1.barh(range(len(rf_top_features)), rf_top_features['rf_importance'])
ax1.set_yticks(range(len(rf_top_features)))
ax1.set_yticklabels(rf_top_features['feature'])
ax1.set_xlabel('Feature Importance')
ax1.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/kreitzer_boehmke2016/kreitzer_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/kreitzer_boehmke2016/kreitzer_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/kreitzer_boehmke2016/kreitzer_feature_importance.csv', index = False)

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
plt.savefig('figures/kreitzer_boehmke2016/kreitzer_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/kreitzer_boehmke2016/kreitzer_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()