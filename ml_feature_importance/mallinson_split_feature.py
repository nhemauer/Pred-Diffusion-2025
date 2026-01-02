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
mallinson_2019_full = pd.read_csv(r"data/mallinson2019.csv")

covariates = ["neighbor_prop", "ideology_relative_hm", "congress_majortopic", "init_avail", "init_qual", "divided_gov",
              "legprof_squire", "percap_log", "population_log", "mip", "complexity_topic", "mip_complexity_topic", "nyt", "year_count", "time_log"]
mallinson_2019 = mallinson_2019_full[["adopt", "year"] + covariates].dropna()

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

mallinson_2019 = mallinson_2019.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Split data based on midpoint year
midyear = np.floor((mallinson_2019['year'].min() + mallinson_2019['year'].max()) / 2)
mallinson_2019['sample_half'] = np.where(mallinson_2019['year'] <= midyear, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = mallinson_2019[mallinson_2019['sample_half'] == half]
    
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
plt.savefig('figures/mallinson2019/mallinson_split_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()