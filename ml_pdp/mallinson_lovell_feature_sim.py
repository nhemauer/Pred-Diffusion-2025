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
mallinson_lovell_sim_full = pd.read_csv(r"figures/mallinson_lovell2022/mallinson_lovell_sim_data.csv")

covariates = ["republican","legprof_squire","exp_pupil10000_adj","mathscore4th","readscore4th",
              "time"]
mallinson_lovell2022 = mallinson_lovell_sim_full[["event"] + covariates].dropna()

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

# Define X and y
X = mallinson_lovell2022[covariates_renamed].copy()
y = mallinson_lovell2022['event']

# Define custom features
custom_xgb_features = [
    "Republican", 
    "Legislative Professionalism",
    "Net Expenditures Per Pupil",
    "Math",
    "Reading",
    "Time"
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
mallinson_lovell_real_full = pd.read_csv(r"data/mallinson_lovell2022.csv")
mallinson_lovell_real = mallinson_lovell_real_full[["adopt"] + covariates].dropna()
mallinson_lovell_real = mallinson_lovell_real.rename(columns = variable_names)

# Define baseline X and y
X_real = mallinson_lovell_real[covariates_renamed].copy()
y_real = mallinson_lovell_real['adopt']

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

    # Hide unused subplots
    for j in range(len(custom_xgb_features), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('figures/mallinson_lovell2022/mallinson_lovell_partial_dependence_xgb_simulation.png', dpi = 300, bbox_inches = 'tight')
plt.show()