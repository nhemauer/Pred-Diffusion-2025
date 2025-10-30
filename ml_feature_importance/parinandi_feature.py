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
parinandi2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]
parinandi2020 = parinandi2020_full[["oneemulation"] + covariates].dropna()

# Rename columns
variable_names = {
    "adagovideology": "Legislative Ideology",
    "citizenideology": "Citizen Ideology", 
    "medianivoteshare": "Median Incumbent Vote Share",
    "partydecline": "Party Decline",
    "squirescore": "Legislative Professionalism",
    "incunemp": "Change in Unemployment",
    "pctpercapincome": "Per Capita Income",
    "percenturban": "Urban Percentage",
    "ugovd": "Unified Dem. Government",
    "percentfossilprod": "Fossil Fuel Production",
    "renergyprice11": "Real Energy Price",
    "deregulated": "Deregulated",
    "geoneighborlag": "Lagged Geographic Neighbor",
    "ideoneighborlag": "Lagged Ideological Neighbor",
    "premulation1": "Prior Borrowing",
    "year": "Year",
    "featureyear": "Provision Year"
}

parinandi2020 = parinandi2020.rename(columns = variable_names, inplace = True)

# Define X and y
X = parinandi2020.drop(columns = ['oneemulation']).copy()
y = parinandi2020['oneemulation']

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
    max_depth = 10,
    max_samples = 0.75,
    min_samples_leaf = 1,
    n_estimators = 245,
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
    gamma = 0,
    grow_policy = 'depthwise',
    learning_rate = 0.1,
    max_bin = 256,
    max_depth = 3,
    min_child_weight = 5,
    n_estimators = 500,
    objective = 'binary:logistic',
    reg_alpha = 0,
    reg_lambda = 1,
    scale_pos_weight = 1,
    subsample = 0.5,
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
plt.savefig('figures/parinandi2020/parinandi_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/parinandi2020/parinandi_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Save output
importance_df.to_csv('figures/parinandi2020/parinandi_feature_importance.csv', index = False)

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
plt.savefig('figures/parinandi2020/parinandi_partial_dependence_rf.png', dpi = 300, bbox_inches = 'tight')
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
plt.savefig('figures/parinandi2020/parinandi_partial_dependence_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()