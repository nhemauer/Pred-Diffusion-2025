import pandas as pd
import numpy as np
import random
import warnings
import os
from scipy.stats import f_oneway
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
random.seed(1337)

### Load Data

# Data
parinandi2020_full = pd.read_stata(r"data/parinandi2020.dta")

covariates = [
    "adagovideology", "citizenideology", "medianivoteshare", "partydecline", "squirescore",
    "incunemp", "pctpercapincome", "percenturban", "ugovd", "percentfossilprod", "renergyprice11",
    "deregulated", "geoneighborlag", "ideoneighborlag", "premulation1", "year", "featureyear"
]

parinandi2020 = parinandi2020_full[["oneemulation"] + covariates].dropna()

# Define X and y
X = parinandi2020.drop(columns = ['oneemulation']).copy()
y = parinandi2020['oneemulation']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### Original Parameter Grid

param_grid = {
    'n_estimators': (100, 300, 500),
    'max_depth': (3, 6, 10, 20),
    'max_bin': (16, 32, 64, 128, 256),
    'booster': ['gbtree', 'dart'],
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss', 'auc', 'error', 'aucpr'],
    'tree_method': ['auto', 'exact', 'approx', 'hist'],
    'grow_policy': ['depthwise', 'lossguide'],
    'learning_rate': (0.01, 0.1, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 2),
    'reg_alpha': (0, 2),
    'reg_lambda': (1, 2),
    'min_child_weight': (1, 5, 10),
    'max_leaves': (0, 16, 32),
    'scale_pos_weight': (1, 5, 10)
}

### Coarse Sampling of Hyperparameters

def sample_grid(param_grid, n_samples = 40):
    samples = []
    for _ in range(n_samples):
        combo = {k: random.choice(v if isinstance(v, (list, tuple)) else [v])
                 for k, v in param_grid.items()}
        samples.append(combo)
    return samples

sampled_params = sample_grid(param_grid, n_samples = 40)

results = []
for params in sampled_params:
    model = XGBClassifier(**params, random_state = 1337, use_label_encoder = False)
    scores = cross_val_score(model, X_scaled, y, cv = 3, scoring = "average_precision", n_jobs = -1)
    results.append({**params, "metric": np.mean(scores)})

df_results = pd.DataFrame(results)

### F-Test for Hyperparameter Significance

def ftest_for_param(df, param_name):
    groups = [group["metric"].values for _, group in df.groupby(param_name)]
    if len(groups) > 1:
        return f_oneway(*groups)
    return np.nan, np.nan

ftest_df = []
for param in param_grid.keys():
    f_stat, p_val = ftest_for_param(df_results, param)
    ftest_df.append({"param": param, "F-stat": f_stat, "p-value": p_val})

ftest_df = pd.DataFrame(ftest_df).sort_values("p-value")

os.chdir("ml_forecast\ml_hyperparameter")

### Save Outputs
with open("figures/parinandi2019/hyperparameter_results.txt", "w") as f:
    f.write("F-test results:\n")
    f.write(str(ftest_df))
    f.write("\n\n")
    
    f.write("Top values for each parameter:\n")
    for param in param_grid.keys():
        top_values = df_results.groupby(param)["metric"].mean().sort_values(ascending = False)
        f.write(f"{param}: {top_values}\n")
