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
parinandi2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]
parinandi2020 = parinandi2020_full[["oneemulation"] + covariates].dropna()

# Rename columns
variable_names = {
    "adagovideology": "Legislative Ideology",
    "citizenideology": "Citizen Ideology", 
    "medianivoteshare": "Median Incumbent Vote Share",
    "partydecline": "Party Decline",
    "squirescore": "Legislative Professionalism",
    "incunemp": "Change in Unemployment",
    "pctpercapincome": "Per Capita Income",
    "percenturban": "Urban Percentage",
    "ugovd": "Unified Dem. Government",
    "percentfossilprod": "Fossil Fuel Production",
    "renergyprice11": "Real Energy Price",
    "deregulated": "Deregulated",
    "geoneighborlag": "Lagged Geographic Neighbor",
    "ideoneighborlag": "Lagged Ideological Neighbor",
    "premulation1": "Prior Borrowing",
    "year": "Year",
    "featureyear": "Provision Year"
}

parinandi2020 = parinandi2020.rename(columns = variable_names)

# Update covariates list with new names
covariates_renamed = [variable_names[var] for var in covariates]

# Split data based on 70/30 year split
year_range = parinandi2020['Year'].max() - parinandi2020['Year'].min()
split_year = parinandi2020['Year'].min() + (year_range * 0.7)
parinandi2020['sample_half'] = np.where(parinandi2020['Year'] <= split_year, 1, 2)

os.chdir("ml_feature_importance")

# Train separate models for each half
sample_halves = [1, 2]
importance_dfs = {}
models = {}

for half in sample_halves:
    # Filter data by sample half
    half_data = parinandi2020[parinandi2020['sample_half'] == half]
    
    # Define X and y for this half
    X_half = half_data[covariates_renamed].copy()
    y_half = half_data['oneemulation']
    
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
        max_depth = 10,
        max_samples = 0.75,
        min_samples_leaf = 1,
        n_estimators = 245,
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
                label=f'First 70% (Years ≤ {int(split_year)})', color = 'black')
bars2 = ax.barh(x + width/2, half_2_importances, width, 
                label=f'Last 30% (Years > {int(split_year)})', color = 'gray')

# Customize plot
ax.set_yticks(x)
ax.set_yticklabels(rf_top_features_1['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance Comparison Across Sample Splits')
ax.legend(loc = 'lower right')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/parinandi2020/parinandi_split_feature_importance_rf.png', dpi = 300, bbox_inches = 'tight')
plt.show()