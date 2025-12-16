import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

random.seed(1337)

os.chdir("ml_simulation")

# Data
kreitzer_sim_full = pd.read_csv(r"figures/kreitzer_boehmke2016/kreitzer_boehmke_sim_data.csv")

covariates = [
    "norrander_legality", "religadhrate", "initdif", "dem_gov", "uni_dem_leg",
    "fem_dem", "nbrspct", "rescaledmedincome", "rescaledpopsize", "time", 
    "time2", "webster"
]
kreitzer_sim = kreitzer_sim_full[["event", "billnum"] + covariates].dropna()

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

kreitzer_sim = kreitzer_sim.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Define X and y
X = kreitzer_sim[covariates_renamed + ["billnum"]].copy()
X = pd.get_dummies(X, columns = ['billnum'], drop_first = True)
y = kreitzer_sim['event']

# Define custom features
custom_rf_features = [
    "Religious Adherence",
    "Median Income",
    "Population",
    "Democratic Women",
    "Time",
    "Neighbor Adoption %",
    "Time Squared",
    "Abortion Opinion",
    "Initiative Difficulty",
]

# Store ICE data for simulated models
rf_ice_data = {feature: [] for feature in custom_rf_features}

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337, stratify = y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get feature names
feature_names = X_train.columns.tolist()

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

# Collect RF ICE data
for feature in custom_rf_features:
    feature_idx = feature_names.index(feature)
    pd_result = partial_dependence(
        rf_model, 
        X_train_scaled, 
        features = [feature_idx],
        response_method = 'predict_proba',
        kind = 'individual'
    )
    rf_ice_data[feature] = (pd_result['grid_values'][0], pd_result['individual'][0])

# Plot ICE plots
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(custom_rf_features):
    # Get ICE data
    x_vals, ice_curves = rf_ice_data[feature]
    
    # Plot ICE curves (subset to avoid overcrowding)
    n_samples = min(100, ice_curves.shape[0])
    for j in range(n_samples):
        axes[i].plot(x_vals, ice_curves[j], alpha = 0.3, linewidth = 0.8, color = 'gray')
    
    # Add PDP as average
    pdp_vals = ice_curves.mean(axis = 0)
    axes[i].plot(x_vals, pdp_vals, linewidth = 2.5, color = 'black', label = 'PDP (Average)')
    
    axes[i].set_title(f'ICE Plot: {feature}', fontsize = 12, fontweight = 'bold')
    axes[i].set_xlabel(feature, fontsize = 10)
    axes[i].set_ylabel('Predicted Probability of Adoption', fontsize = 10)
    axes[i].grid(True, alpha = 0.3)
    
    if i == 0:
        axes[i].legend(loc = 'best', fontsize = 9)

plt.tight_layout()
plt.savefig('figures/kreitzer_boehmke2016/kreitzer_boehmke_ice_rf_simulation.png', dpi = 300, bbox_inches = 'tight')
plt.show()