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
boushey_2016_full = pd.read_stata(r"data/boushey2016.dta")

# Covariates
covariates = ["policycongruent","gub_election","elect2", "hvd_4yr", "fedcrime",
                "leg_dem_per_2pty","dem_governor","insession","propneighpol",
                "citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq",
                "violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3"]
boushey_2016 = boushey_2016_full[["state", "year", "dvadopt"] + covariates].dropna()

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

# Split data based on midpoint year
midyear = np.floor((boushey_2016['year'].min() + boushey_2016['year'].max()) / 2)
boushey_2016['sample_half'] = np.where(boushey_2016['year'] <= midyear, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = boushey_2016[boushey_2016['sample_half'] == half]
    
    # Define X and y for this half
    X_half = half_data[covariates_renamed].copy()
    y_half = half_data['dvadopt']
    
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
        criterion = 'gini',
        max_depth = 10,
        min_samples_leaf = 3,
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
plt.savefig('figures/boushey2016/boushey_split_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()