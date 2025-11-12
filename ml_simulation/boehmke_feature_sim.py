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
boehmke_2017_full = pd.read_csv(r"figures/boehmke2017/boehmke_sim_data.csv")

# Covariates
covariates = ["srcs_decay","nbrs_lag","rpcpinc","totpop","legp_squire",
                "citi6010","unif_rep","unif_dem","time","time_sq","time_cube"]
boehmke_2017 = boehmke_2017_full[["state", "year", "event"] + covariates].dropna()

# Rename columns
variable_names = {
    "srcs_decay": "Lag Source Adoptions",
    "nbrs_lag": "Lag Neighbor Adoptions", 
    "rpcpinc": "Personal Income",
    "totpop": "Total Population",
    "legp_squire": "Legislative Professionalism",
    "citi6010": "State Citizen Ideology",
    "unif_rep": "Unified Republican Control",
    "unif_dem": "Unified Democratic Control",
    "time": "Time",
    "time_sq": "Time Squared",
    "time_cube": "Time Cubed"
}

boehmke_2017 = boehmke_2017.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Define X and y
X = boehmke_2017.drop(columns = ['event', 'year']).copy()
X = pd.get_dummies(X, columns = ['state'], drop_first = True)
y = boehmke_2017['event']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use best hyperparameters from the random-split experiment
rf_model = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0,
    class_weight = None,
    criterion = 'entropy',
    max_depth = 25,
    min_samples_leaf = 4,
    min_samples_split = 8,
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
    booster = 'gbtree',
    colsample_bytree = 0.7639498961822481,
    eval_metric = 'aucpr',
    gamma = 2,
    grow_policy = 'depthwise',
    learning_rate = 0.027136817935642106,
    max_bin = 128,
    max_depth = 6,
    max_leaves = 31,
    min_child_weight = 10,
    n_estimators = 187,
    objective = 'binary:logistic',
    scale_pos_weight = 4,
    subsample = 0.6526184572569168,
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
plt.savefig('figures/boehmke2017/boehmke_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/boehmke2017/boehmke_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/boehmke2017/boehmke_feature_importance.csv', index = False)

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
plt.savefig('figures/boehmke2017/boehmke_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/boehmke2017/boehmke_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()