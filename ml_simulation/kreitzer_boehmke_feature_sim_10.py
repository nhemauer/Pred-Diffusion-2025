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

# Store PDP data for all models
rf_pdp_data = {feature: [] for feature in custom_rf_features}

for seed in range(10):
    random.seed(1337 + seed)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337 + seed, stratify = y)

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

    # Collect RF PDP data
    for feature in custom_rf_features:
        feature_idx = feature_names.index(feature)
        pd_result = partial_dependence(
            rf_model, 
            X_train_scaled, 
            features = [feature_idx],
            response_method = 'predict_proba',
            kind = 'average'
        )
        rf_pdp_data[feature].append((pd_result['values'][0], pd_result['average'][0]))

# Load real data for baseline
os.chdir("..")
kreitzer_real_full = pd.read_stata(r"data/kreitzer_boehmke2016.dta")
kreitzer_real = kreitzer_real_full[["adopt_policy", "policy_num"] + covariates].dropna()
kreitzer_real = kreitzer_real.rename(columns = variable_names)

# Create dummy variables
kreitzer_real = pd.get_dummies(kreitzer_real, columns = ['policy_num'], drop_first = True)

# Define baseline X and y
policy_columns = [col for col in kreitzer_real.columns if col.startswith('policy_num_')]
X_real = kreitzer_real[covariates_renamed + policy_columns].copy()
y_real = kreitzer_real['adopt_policy']

# Split baseline data
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size = 0.2, random_state = 1337, stratify = y_real)

# Scale baseline features
scaler_real = StandardScaler()
X_train_real_scaled = scaler_real.fit_transform(X_train_real)
X_test_real_scaled = scaler_real.transform(X_test_real)

# Fit baseline RF model
rf_model_real = RandomForestClassifier(
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
rf_model_real.fit(X_train_real_scaled, y_train_real)

# Get baseline PDP data
feature_names_real = X_train_real.columns.tolist()
rf_baseline_pdp = {}
for feature in custom_rf_features:
    feature_idx = feature_names_real.index(feature)
    pd_result = partial_dependence(
        rf_model_real, 
        X_train_real_scaled, 
        features = [feature_idx],
        response_method = 'predict_proba',
        kind = 'average'
    )
    rf_baseline_pdp[feature] = (pd_result['values'][0], pd_result['average'][0])

os.chdir("ml_simulation")

# Plot PDPs
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(custom_rf_features):
    # Plot simulated data
    for seed in range(10):
        x_vals, y_vals = rf_pdp_data[feature][seed]
        axes[i].plot(x_vals, y_vals, alpha = 0.5, linewidth = 1, color = 'lightgray', 
                    label='Simulated Data' if seed == 0 and i == 0 else "")
    
    # Plot baseline
    x_baseline, y_baseline = rf_baseline_pdp[feature]
    axes[i].plot(x_baseline, y_baseline, alpha = 1.0, linewidth = 2, color = 'black',
                label='Real Data' if i == 0 else "")
    
    axes[i].set_title(f'PDP: {feature}')
    axes[i].set_ylabel('Predicted Probability of Adoption')
    axes[i].grid(True, alpha = 0.3)
    
    # Add legend only to the first subplot
    if i == 0:
        axes[i].legend(loc = 'upper left')

plt.tight_layout()
plt.savefig('figures/kreitzer_boehmke2016/kreitzer_boehmke_partial_dependence_rf_simulation.png', dpi = 300, bbox_inches = 'tight')
plt.show()