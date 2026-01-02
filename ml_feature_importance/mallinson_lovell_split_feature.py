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
import numpy as np

random.seed(1337)

# Data
mallinson_lovell2022_full = pd.read_csv(r"data/mallinson_lovell2022.csv")

# Covariates
covariates = ["republican","legprof_squire","exp_pupil10000_adj","mathscore4th","readscore4th",
              "time"]
mallinson_lovell2022 = mallinson_lovell2022_full[["adopt", "year"] + covariates].dropna()

# Rename columns
variable_names = {
    "republican": "Republican",
    "legprof_squire": "Legislative Professionalism",
    "exp_pupil10000_adj": "Net Expenditures Per Pupil",
    "readscore4th": "Reading",
    "mathscore4th": "Math",
    "time": "Time"
}

mallinson_lovell2022 = mallinson_lovell2022.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Split data based on midpoint year
midyear = np.floor((mallinson_lovell2022['year'].min() + mallinson_lovell2022['year'].max()) / 2)
mallinson_lovell2022['sample_half'] = np.where(mallinson_lovell2022['year'] <= midyear, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = mallinson_lovell2022[mallinson_lovell2022['sample_half'] == half]
    
    # Define X and y for this half
    X_half = half_data[covariates_renamed].copy()
    y_half = half_data['adopt']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_half, y_half, test_size = 0.2, random_state = 1337, stratify = y_half
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use best hyperparameters from the random-split experiment
    xgb_model = XGBClassifier(
        booster = 'gbtree',
        eval_metric = 'aucpr',
        gamma = 2,
        grow_policy = 'depthwise',
        learning_rate = 0.04787337145112113,
        max_bin = 128,
        max_depth = 3,
        n_estimators = 100,
        objective = 'binary:logistic',
        reg_alpha = 0,
        scale_pos_weight = 1,
        subsample = 0.8012137562136856,
        tree_method = 'auto',
        random_state = 1337
    )
    
    # Fit xgb model
    xgb_model.fit(X_train_scaled, y_train)
    models[half] = xgb_model
    
    # xgb feature importance
    feature_names = X_train.columns.tolist()
    xgb_feature_importance = xgb_model.feature_importances_
    
    # Store importance dataframe
    importance_dfs[half] = pd.DataFrame({
        'feature': feature_names,
        'xgb_importance': xgb_feature_importance
    })

# Create combined feature importance plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

# Plot for Sample Half 1
xgb_top_features_1 = importance_dfs[1].sort_values(by = 'xgb_importance', ascending = False).head(20)
ax1.barh(range(len(xgb_top_features_1)), xgb_top_features_1['xgb_importance'])
ax1.set_yticks(range(len(xgb_top_features_1)))
ax1.set_yticklabels(xgb_top_features_1['feature'])
ax1.set_xlabel('Feature Importance')
ax1.set_title(f'Sample Half 1 (Years ≤ {int(midyear)})')
ax1.invert_yaxis()

# Plot for Sample Half 2
xgb_top_features_2 = importance_dfs[2].sort_values(by = 'xgb_importance', ascending = False).head(20)
ax2.barh(range(len(xgb_top_features_2)), xgb_top_features_2['xgb_importance'])
ax2.set_yticks(range(len(xgb_top_features_2)))
ax2.set_yticklabels(xgb_top_features_2['feature'])
ax2.set_xlabel('Feature Importance')
ax2.set_title(f'Sample Half 2 (Years > {int(midyear)})')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/mallinson_lovell2022/mallinson_lovell_split_feature_importance_xgb.png', dpi = 300, bbox_inches = 'tight')
plt.show()