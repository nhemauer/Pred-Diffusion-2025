from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import warnings
import os

warnings.filterwarnings('ignore')
random.seed(1337)

# Data
boushey_2016_full = pd.read_stata(r"data/boushey2016.dta")

covariates = ["policycongruent","gub_election","elect2", "hvd_4yr", "fedcrime",
                "leg_dem_per_2pty","dem_governor","insession","propneighpol",
                "citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq",
                "violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3"]
boushey_2016 = boushey_2016_full[["billname", "dvadopt"] + covariates].dropna()

for bill in boushey_2016['billname'].unique():
    # Create datasets
    train_data = boushey_2016[boushey_2016['billname'] != bill]
    val_data = boushey_2016[boushey_2016['billname'] == bill]
    
    # Define X and y for the current bill
    X_train = train_data[covariates].copy()
    y_train = train_data['dvadopt']
    X_val = val_data[covariates].copy()
    y_val = val_data['dvadopt']

    # Scale features
    scaler = StandardScaler()
    X
    X_val_scaled = scaler.fit_transform(X_val)
    
    print(f"Processing bill: {bill}")
