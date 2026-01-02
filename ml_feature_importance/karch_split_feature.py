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
karch_2016_full = pd.read_stata(r"data/karch2016.dta")

# Covariates
covariates = [
    "traditional", "nborsstd", "prevadoptstd", "complexity", "igrole",
    "regov", "unified", "perdemstd", "incpcadjstd", "exppcadjstd",
    "logpopstd", "collegstd", "perurbanstd", "profstd",
    "traditional_nborsstd", "traditional_prevadoptstd", "traditional_complexity",
    "traditional_igrole", "traditional_regov", "traditional_unified",
    "traditional_perdemstd", "traditional_incpcadjstd", "traditional_exppcadjstd",
    "traditional_logpopstd", "traditional_collegstd", "traditional_perurbanstd",
    "traditional_profstd"
]
karch_2016 = karch_2016_full[["adopt", "year"] + covariates].dropna()

# Rename columns
variable_names = {
    "traditional": "Traditional",
    "nborsstd": "Neighbors",
    "prevadoptstd": "Previous Adopters",
    "complexity": "Complexity",
    "igrole": "Interest Group Role",
    "regov": "Republican Governor",
    "unified": "Unified",
    "perdemstd": "Democratic Legislature",
    "incpcadjstd": "Income per Capita",
    "exppcadjstd": "Expenditures per Capita",
    "logpopstd": "Population",
    "collegstd": "Pct College Educated",
    "perurbanstd": "Pct Urban",
    "profstd": "Legislative Professionalism",
    "traditional_nborsstd": "Traditional x Neighbors",
    "traditional_prevadoptstd": "Traditional x Prev. Adopters",
    "traditional_complexity": "Traditional x Complexity",
    "traditional_igrole": "Traditional x Interest Group",
    "traditional_regov": "Traditional x Rep. Governor",
    "traditional_unified": "Traditional x Unified",
    "traditional_perdemstd": "Traditional x Dem. Legislature",
    "traditional_incpcadjstd": "Traditional x Income",
    "traditional_exppcadjstd": "Traditional x Expenditures",
    "traditional_logpopstd": "Traditional x Population",
    "traditional_collegstd": "Traditional x College",
    "traditional_perurbanstd": "Traditional x Urban",
    "traditional_profstd": "Traditional x Professionalism"
}

karch_2016 = karch_2016.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Split data based on midpoint year
midyear = np.floor((karch_2016['year'].min() + karch_2016['year'].max()) / 2)
karch_2016['sample_half'] = np.where(karch_2016['year'] <= midyear, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = karch_2016[karch_2016['sample_half'] == half]
    
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
    rf_model = RandomForestClassifier(
        bootstrap = True,
        ccp_alpha = 0.0,
        class_weight = None,
        criterion = 'entropy',
        max_depth = 50,
        max_samples = 0.7236542889639881,
        n_estimators = 500,
        random_state = 1337
    )
    
    # Fit rf model
    rf_model.fit(X_train_scaled, y_train)
    models[half] = rf_model
    
    # RF feature importance
    feature_names = X_train.columns.tolist()
    rf_feature_importance = rf_model.feature_importances_
    
    # Store importance dataframe
    importance_dfs[half] = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_feature_importance
    })

# Create combined feature importance plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

# Plot for Sample Half 1
rf_top_features_1 = importance_dfs[1].sort_values(by = 'rf_importance', ascending = False).head(20)
ax1.barh(range(len(rf_top_features_1)), rf_top_features_1['rf_importance'])
ax1.set_yticks(range(len(rf_top_features_1)))
ax1.set_yticklabels(rf_top_features_1['feature'])
ax1.set_xlabel('Feature Importance')
ax1.set_title(f'Sample Half 1 (Years ≤ {int(midyear)})')
ax1.invert_yaxis()

# Plot for Sample Half 2
rf_top_features_2 = importance_dfs[2].sort_values(by = 'rf_importance', ascending = False).head(20)
ax2.barh(range(len(rf_top_features_2)), rf_top_features_2['rf_importance'])
ax2.set_yticks(range(len(rf_top_features_2)))
ax2.set_yticklabels(rf_top_features_2['feature'])
ax2.set_xlabel('Feature Importance')
ax2.set_title(f'Sample Half 2 (Years > {int(midyear)})')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/karch2016/karch_split_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()