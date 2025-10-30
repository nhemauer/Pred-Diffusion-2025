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
mallinson_2019_full = pd.read_csv(r"data/mallinson2019.csv")

covariates = ["neighbor_prop", "ideology_relative_hm", "congress_majortopic", "init_avail", "init_qual", "divided_gov",
              "legprof_squire", "percap_log", "population_log", "mip", "complexity_topic", "mip_complexity_topic", "nyt", "year_count", "time_log"]
mallinson_2019 = mallinson_2019_full[["adopt", "policy"] + covariates].dropna()

# Rename columns
variable_names = {
    "neighbor_prop": "Neighbor Adoptions",
    "ideology_relative_hm": "Ideological Distance",
    "congress_majortopic": "Congressional Hearings",
    "init_avail": "Iniative Available",
    "init_qual": "Initiative Qual. Difficulty",
    "divided_gov": "Divided Government",
    "legprof_squire": "Legislative Professionalism",
    "percap_log": "Per Capita Income",
    "population_log": "Population",
    "mip": "Most Important Problem",
    "complexity_topic": "Complex Policy",
    "mip_complexity_topic": "MIP x Complex",
    "nyt": "New York Times",
    "year_count": "Year",
    "time_log": "Time"
}

mallinson_2019 = mallinson_2019.rename(columns = variable_names, inplace = True)

# Define X and y
X = mallinson_2019.drop(columns = ['adopt', 'policy']).copy()
y = mallinson_2019['adopt']

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
    ccp_alpha = 1.8209810854730173e-05,
    class_weight = None,
    criterion = 'entropy',
    max_depth = 50,
    min_samples_leaf = 1,
    n_estimators = 393,
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
    learning_rate = 0.1,
    max_bin = 256,
    max_depth = 20,
    max_leaves = 32,
    min_child_weight = 5,
    n_estimators = 300,
    objective = 'binary:logistic',
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

# Create Random Forest feature importance plot
fig1, ax1 = plt.subplots(1, 1, figsize = (10, 10))
rf_top_features = importance_df.sort_values(by = 'rf_importance', ascending = False).head(20)
ax1.barh(range(len(rf_top_features)), rf_top_features['rf_importance'])
ax1.set_yticks(range(len(rf_top_features)))
ax1.set_yticklabels(rf_top_features['feature'])
ax1.set_xlabel('Feature Importance')
ax1.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/mallinson2019/mallinson_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/mallinson2019/mallinson_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/mallinson2019/mallinson_feature_importance.csv', index = False)

# Create RF PDP with top 9 features
top_features_rf = importance_df.sort_values(by = 'rf_importance', ascending = False).head(9)['feature'].tolist()

# Create partial dependence plot
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(top_features_rf):
    feature_idx = feature_names.index(feature)
    PartialDependenceDisplay.from_estimator(
        rf_model, 
        X_train_scaled, 
        features = [feature_idx],
        feature_names = feature_names,
        ax = axes[i],
        kind = 'average',
        response_method = 'predict_proba'
    )
    axes[i].set_title(f'PDP: {feature}')
    axes[i].set_ylabel('Predicted Probability of Adoption')

plt.tight_layout()
plt.savefig('figures/mallinson2019/mallinson_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Create XGBoost PDP with top 9 features
top_features_xgb = importance_df.sort_values(by = 'xgb_importance', ascending = False).head(9)['feature'].tolist()

# Create partial dependence plot
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(top_features_xgb):
    feature_idx = feature_names.index(feature)
    PartialDependenceDisplay.from_estimator(
        xgb_model, 
        X_train_scaled, 
        features = [feature_idx],
        feature_names = feature_names,
        ax = axes[i],
        kind = 'average',
        response_method = 'predict_proba'
    )
    axes[i].set_title(f'PDP: {feature}')
    axes[i].set_ylabel('Predicted Probability of Adoption')

plt.tight_layout()
plt.savefig('figures/mallinson2019/mallinson_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()