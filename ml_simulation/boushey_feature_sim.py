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
boushey_2016_full = pd.read_csv(r"figures/boushey2016/boushey_sim_data.csv")

# Covariates
covariates = ["policycongruent","gub_election","elect2", "hvd_4yr", "fedcrime",
                "leg_dem_per_2pty","dem_governor","insession","propneighpol",
                "citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq",
                "violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3"]
boushey_2016 = boushey_2016_full[["state", "event"] + covariates].dropna()

# Rename columns
variable_names = {
    "policycongruent": "Policy Congruence",
    "gub_election": "Elect1", 
    "elect2": "Elect2",
    "hvd_4yr": "Electoral Competition",
    "fedcrime": "National Crime Salience",
    "leg_dem_per_2pty": "Democratic Party Strength",
    "dem_governor": "Democratic Governor",
    "insession": "Legislative Session",
    "propneighpol": "Neighbors",
    "citidist": "Ideological Distance",
    "squire_prof86": "Legislative Professionalism",
    "citi6008": "Political Ideology",
    "crimespendpc": "Crime Spending per Capita",
    "crimespendpcsq": "Crime Spending (Squared)",
    "violentthousand": "Violent Crime Rate",
    "pctwhite": "Pct. Population White",
    "stateincpercap": "Per Capita Income",
    "logpop": "Logged Population",
    "counter": "Time",
    "counter2": "Time Squared",
    "counter3": "Time Cubed"
}

boushey_2016 = boushey_2016.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Define X and y
X = boushey_2016[covariates_renamed].copy()
y = boushey_2016['event']

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
    criterion = 'gini',
    max_depth = 10,
    min_samples_leaf = 3,
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
    eval_metric = 'aucpr',
    grow_policy = 'depthwise',
    learning_rate = 0.01759560389789212,
    max_bin = 256,
    max_depth = 6,
    max_leaves = 32,
    n_estimators = 228,
    objective = 'binary:logistic',
    reg_lambda = 1,
    subsample = 0.7949870515841156,
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
plt.savefig('figures/boushey2016/boushey_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/boushey2016/boushey_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/boushey2016/boushey_feature_importance.csv', index = False)

# Create RF PDP with the same features as the original feature importance
custom_rf_features = [
    "Neighbors",
    "Ideological Distance",
    "Per Capita Income",
    "Logged Population",
    "Political Ideology",
    "Crime Spending per Capita",
    "Crime Spending (Squared)",
    "Time Cubed",
    "Electoral Competition",
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

plt.tight_layout()
plt.savefig('figures/boushey2016/boushey_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Create XGB PDP with the same features as the original feature importance
custom_xgb_features = [
    "Neighbors",
    "Legislative Session",
    "Time",
    "Ideological Distance",
    "Policy Congruence",
    "Per Capita Income",
    "Logged Population",
    "National Crime Salience",
    "Pct. Population White",
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

plt.tight_layout()
plt.savefig('figures/boushey2016/boushey_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()