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
berry_berry1990_full = pd.read_csv("data/berry_berry1990.txt", delim_whitespace = True, header = None)
berry_berry1990_full.columns = ["state", "year", "adopt", "fiscal_1", "party", "elect1", "elect2", "income_1", "neighbor", "nbrpercn", "religion"]
berry_berry1990 = berry_berry1990_full[berry_berry1990_full['party'] != 9].copy() # 9 is the NA (For MN and NE)

# Rename columns
variable_names = {
    "fiscal_1": "Fiscal",
    "party": "Party", 
    "elect1": "Elect1",
    "elect2": "Elect2",
    "income_1": "Income",
    "nbrpercn": "Neighbors",
    "religion": "Religion"
}

berry_berry1990 = berry_berry1990.rename(columns = variable_names)

# Split data based on midpoint year
midyear = np.floor((berry_berry1990['year'].min() + berry_berry1990['year'].max()) / 2)
berry_berry1990['sample_half'] = np.where(berry_berry1990['year'] <= midyear, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = berry_berry1990[berry_berry1990['sample_half'] == half]
    
    # Define X and y for this half
    X_half = half_data.drop(columns = ['adopt', 'neighbor', 'state', 'year', 'sample_half']).copy()
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
        ccp_alpha = 0.07870049004782728,
        class_weight = 'balanced',
        criterion = 'gini',
        max_depth = 10,
        max_samples = 0.5,
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
                label=f'Sample Half 1 (Years ≤ 19{int(midyear)})', color = 'black')
bars2 = ax.barh(x + width/2, half_2_importances, width, 
                label=f'Sample Half 2 (Years > 19{int(midyear)})', color = 'gray')

# Customize plot
ax.set_yticks(x)
ax.set_yticklabels(rf_top_features_1['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance Comparison Across Sample Halves')
ax.legend()
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/berry_berry1990/berry_split_feature_importance_rf.png', dpi=300, bbox_inches = 'tight')
plt.show()