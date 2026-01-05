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
lacombe_boehmke2021_full = pd.read_stata(r"data/lacombe_boehmke2021.dta")

covariates = [
    "initiative", "init_sigs", "std_latnt_decay", "std_nbrs_lag", "std_population",
    "std_masssociallib_est", "unified", "duration", "durationsq", "durationcb", "std_income",
    "std_bowen_1", "std_bowen_2", "change_pop", "change_inc", "party_change"
]
lacombe_boehmke2021 = lacombe_boehmke2021_full[["adoption", "year"] + covariates].dropna()

# Rename columns
variable_names = {
    "initiative": "Initiative Process",
    "init_sigs": "Signatures",
    "std_latnt_decay": "Latent Decay",
    "std_nbrs_lag": "Contiguity",
    "std_population": "Population",
    "std_masssociallib_est": "Public Liberalism",
    "unified": "Unified Control",
    "duration": "Duration",
    "durationsq": "Duration Squared",
    "durationcb": "Duration Cubed",
    "std_income": "Income per Capita",
    "std_bowen_1": "Legislative Prof. Dim. 1",
    "std_bowen_2": "Legislative Prof. Dim. 2",
    "change_pop": "Change Population",
    "change_inc": "Change Income",
    "party_change": "Change in Party"
}

lacombe_boehmke2021 = lacombe_boehmke2021.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Split data based on midpoint year
midyear = np.floor((lacombe_boehmke2021['year'].min() + lacombe_boehmke2021['year'].max()) / 2)
lacombe_boehmke2021['sample_half'] = np.where(lacombe_boehmke2021['year'] <= midyear, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = lacombe_boehmke2021[lacombe_boehmke2021['sample_half'] == half]
    
    # Define X and y for this half
    X_half = half_data[['year'] + covariates_renamed].copy()
    X_half = pd.get_dummies(X_half, columns = ['year'], drop_first = True)
    y_half = half_data['adoption']
    
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
        ccp_alpha = 1.4673493976469224e-05,
        class_weight = None,
        criterion = 'entropy',
        max_depth = 50,
        min_samples_leaf = 2,
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
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_feature_importance
    })
    
    # Filter to only include covariates (exclude state dummies)
    importance_dfs[half] = importance_df[importance_df['feature'].isin(covariates_renamed)]

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
                label=f'Sample Half 1 (Years ≤ {int(midyear)})', color = 'black')
bars2 = ax.barh(x + width/2, half_2_importances, width, 
                label=f'Sample Half 2 (Years > {int(midyear)})', color = 'gray')

# Customize plot
ax.set_yticks(x)
ax.set_yticklabels(rf_top_features_1['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance Comparison Across Sample Halves')
ax.legend(loc = 'lower right')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/lacombe_boehmke2021/lacombe_split_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()