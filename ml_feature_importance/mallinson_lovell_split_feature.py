import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
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
    rf_model = RandomForestClassifier(
        bootstrap = True,
        ccp_alpha = 0.006076491497524607,
        class_weight = None,
        criterion = 'log_loss',
        max_depth = 12,
        max_samples = 0.5048236241274381,
        n_estimators = 100,
        random_state = 1337
    )
    
    # Fit rf model
    rf_model.fit(X_train_scaled, y_train)
    models[half] = rf_model
    
    # rf feature importance
    feature_names = X_train.columns.tolist()
    rf_feature_importance = rf_model.feature_importances_
    
    # Store importance dataframe
    importance_dfs[half] = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_feature_importance
    })

# Create combined feature importance plot with grouped bars
fig, ax = plt.subplots(figsize = (12, 10))

# Get top features from Sample Half 1 (this determines the order)
rf_top_features_1 = importance_dfs[1].sort_values(by = 'rf_importance', ascending = False).head(20)

# Get corresponding importance values from Sample Half 2
importance_df_2 = importance_dfs[2].set_index('feature')
half_2_importances = [importance_df_2.loc[feat, 'rf_importance'] if feat in importance_df_2.index else 0 
                      for feat in rf_top_features_1['feature']]

# Set up bar positions
x = np.arange(len(rf_top_features_1))
width = 0.35

# Create grouped bars
bars1 = ax.barh(x - width/2, rf_top_features_1['rf_importance'], width, 
                label=f'First 50% (Years ≤ {int(midyear)})', color = 'black')
bars2 = ax.barh(x + width/2, half_2_importances, width, 
                label=f'Last 50% (Years > {int(midyear)})', color = 'gray')

# Customize plot
ax.set_yticks(x)
ax.set_yticklabels(rf_top_features_1['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance Comparison Across Sample Halves')
ax.legend(loc = 'lower right')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/mallinson_lovell2022/mallinson_lovell_split_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()