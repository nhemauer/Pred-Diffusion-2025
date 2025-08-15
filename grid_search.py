import pandas as pd
import numpy as np
import random
import warnings
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
random.seed(1337)

### Load Data

boushey_2016_full = pd.read_stata(r"data/boushey2016.dta")

covariates = [
    "policycongruent", "gub_election", "elect2", "hvd_4yr", "fedcrime",
    "leg_dem_per_2pty", "dem_governor", "insession", "propneighpol",
    "citidist", "squire_prof86", "citi6008", "crimespendpc", "crimespendpcsq",
    "violentthousand", "pctwhite", "stateincpercap", "logpop",
    "counter", "counter2", "counter3"
]

boushey_2016 = boushey_2016_full[["state", "styear", "dvadopt"] + covariates].dropna()

X = boushey_2016[covariates].copy()
y = boushey_2016['dvadopt']

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
print("\nF-test results:\n", ftest_df)

print("\nTop values for each parameter:")
for param in param_grid.keys():
    top_values = df_results.groupby(param)["metric"].mean().sort_values(ascending = False)
    print(f"{param}: {top_values}")
