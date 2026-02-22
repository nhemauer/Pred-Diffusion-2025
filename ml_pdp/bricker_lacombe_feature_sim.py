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

os.chdir("ml_pdp")

# Data
bricker_lacombe_sim_full = pd.read_csv(r"figures/bricker_lacombe2021/bricker_lacombe_sim_data.csv")

# Covariates
covariates = ["std_score","initiative","init_sigs","std_population",
                "std_citideology","unified","std_income","std_legp_squire",
                "duration","durationsq","durationcb"]
bricker_lacombe_sim = bricker_lacombe_sim_full.dropna()

# Rename columns
variable_names = {
    "std_score": "Similarity",
    "initiative": "Initiative Process",
    "init_sigs": "Average Signatures",
    "std_population": "Population",
    "std_citideology": "Citizen Ideology",
    "unified": "Unified Control",
    "std_income": "Income",
    "std_legp_squire": "Legislative Professionalism",
    "duration": "Duration",
    "durationsq": "Duration Squared",
    "durationcb": "Duration Cubed"
}

bricker_lacombe_sim = bricker_lacombe_sim.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Define X and y
year_columns = [col for col in bricker_lacombe_sim.columns if col.startswith('year_')]
X = bricker_lacombe_sim[covariates_renamed + year_columns].copy()
y = bricker_lacombe_sim['event']

# Define custom features
custom_xgb_features = [
    "Similarity",
    "Duration",
    "Population",
    "Legislative Professionalism",
    "Income",
    "Citizen Ideology",
    "Average Signatures",
    "Initiative Process",
    "Unified Control",
]

# Store PDP data for all models
xgb_pdp_data = {feature: [] for feature in custom_xgb_features}

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
    xgb_model = XGBClassifier(
        booster = 'gbtree',
        eval_metric = 'aucpr',
        gamma = 2,
        grow_policy = 'depthwise',
        learning_rate = 0.033245479413080606,
        max_bin = 64,
        max_depth = 6,
        min_child_weight = 5,
        n_estimators = 465,
        objective = 'binary:logistic',
        scale_pos_weight = 4,
        subsample = 0.8753024969914462,
        tree_method = 'auto',
        random_state = 1337
    )

    # Fit xgb model
    xgb_model.fit(X_train_scaled, y_train)

    # Collect xgb PDP data
    for feature in custom_xgb_features:
        feature_idx = feature_names.index(feature)
        pd_result = partial_dependence(
            xgb_model, 
            X_train_scaled, 
            features = [feature_idx],
            response_method = 'predict_proba',
            kind = 'average'
        )
        xgb_pdp_data[feature].append((pd_result['values'][0], pd_result['average'][0]))

# Load real data for baseline
os.chdir("..")
bricker_lacombe_real_full = pd.read_stata(r"data/bricker_lacombe2021.dta")
bricker_lacombe_real = bricker_lacombe_real_full[["year", "adoption"] + covariates].dropna()
bricker_lacombe_real = bricker_lacombe_real.rename(columns = variable_names)

# Create dummy variables
bricker_lacombe_real = pd.get_dummies(bricker_lacombe_real, columns = ['year'], drop_first = True)

# Define X and y
year_columns = [col for col in bricker_lacombe_real.columns if col.startswith('year_')]
X_real = bricker_lacombe_real[covariates_renamed + year_columns].copy()
y_real = bricker_lacombe_real['adoption']

# Split baseline data
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size = 0.2, random_state = 1337, stratify = y_real)

# Scale baseline features
scaler_real = StandardScaler()
X_train_real_scaled = scaler_real.fit_transform(X_train_real)
X_test_real_scaled = scaler_real.transform(X_test_real)

# Fit baseline xgb model
xgb_model_real = XGBClassifier(
    booster = 'gbtree',
    eval_metric = 'aucpr',
    gamma = 2,
    grow_policy = 'depthwise',
    learning_rate = 0.033245479413080606,
    max_bin = 64,
    max_depth = 6,
    min_child_weight = 5,
    n_estimators = 465,
    objective = 'binary:logistic',
    scale_pos_weight = 4,
    subsample = 0.8753024969914462,
    tree_method = 'auto',
    random_state = 1337
)
xgb_model_real.fit(X_train_real_scaled, y_train_real)

# Get baseline PDP data
feature_names_real = X_train_real.columns.tolist()
xgb_baseline_pdp = {}
for feature in custom_xgb_features:
    feature_idx = feature_names_real.index(feature)
    pd_result = partial_dependence(
        xgb_model_real, 
        X_train_real_scaled, 
        features = [feature_idx],
        response_method = 'predict_proba',
        kind = 'average'
    )
    xgb_baseline_pdp[feature] = (pd_result['values'][0], pd_result['average'][0])

os.chdir("ml_pdp")

# Plot PDPs
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.ravel()

for i, feature in enumerate(custom_xgb_features):
    # Plot simulated data
    for seed in range(10):
        x_vals, y_vals = xgb_pdp_data[feature][seed]
        axes[i].plot(x_vals, y_vals, alpha = 0.5, linewidth = 1, color = 'lightgray', 
                    label='Simulated Data' if seed == 0 and i == 0 else "")
    
    # Plot baseline
    x_baseline, y_baseline = xgb_baseline_pdp[feature]
    axes[i].plot(x_baseline, y_baseline, alpha = 1.0, linewidth = 2, color = 'black',
                label='Real Data' if i == 0 else "")
    
    axes[i].set_title(f'PDP: {feature}')
    axes[i].set_ylabel('Predicted Probability of Adoption')
    axes[i].grid(True, alpha = 0.3)
    
    # Add legend only to the first subplot
    if i == 0:
        axes[i].legend(loc = 'upper left')

plt.tight_layout()
plt.savefig('figures/bricker_lacombe2021/bricker_lacombe_partial_dependence_xgb_simulation.png', dpi = 300, bbox_inches = 'tight')
plt.show()