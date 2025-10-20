import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import warnings
import os

warnings.filterwarnings('ignore')

random.seed(1337)

# Data
boehmke_2017_full = pd.read_stata(r"data/boehmke2017.dta")

os.chdir("ml_coef_split")

# Covariates
covariates = ["srcs_decay","nbrs_lag","rpcpinc","totpop","legp_squire",
                "citi6010","unif_rep","unif_dem","time","time_sq","time_cube"]
boehmke_2017 = boehmke_2017_full[["state", "year", "statepol", "adopt"] + covariates].dropna()

# Get unique year values and split into 50/50
unique_styears = sorted(boehmke_2017['year'].unique())
n_styears = len(unique_styears)
split_point = int(n_styears * 0.5)

# Define ranges for each split
first_50_styears = unique_styears[:split_point]
last_50_styears = unique_styears[split_point:]

# Split data by 50/50
splits = {
    'First_50': boehmke_2017[boehmke_2017['year'].isin(first_50_styears)],
    'Last_50': boehmke_2017[boehmke_2017['year'].isin(last_50_styears)],
    'Full_Dataset': boehmke_2017
}

# Store results for comparison
results_dict = {}

# Run logistic regression for each split
for split_name, data in splits.items():
        # Define X and y
        X = data[covariates + ["state"]].copy()
        X = pd.get_dummies(X, columns = ['state'], drop_first = True)
        X = sm.add_constant(X)
        y = data['adopt']
        
        # Fit Logistic Regression model with clustering
        logistic = sm.Logit(y.astype(float), X.astype(float)).fit(
            cov_type = "cluster", 
            cov_kwds = {'groups': data['statepol']}, 
            disp = 0
        )
        
        # Extract summary table
        summary_df = logistic.summary2().tables[1]
        
        # Filter out state dummy variables
        summary_filtered = summary_df[~summary_df.index.str.startswith("state_")]
        
        # Store coefficients and p-values
        results_dict[split_name] = {
            'feature': summary_filtered.index.tolist(),
            'coef': summary_filtered['Coef.'],
            'sd': summary_filtered['Std.Err.'],
            'p_value': summary_filtered['P>|z|'],
            'n_obs': len(data)
        }

# Convert results to DataFrames
df_first = pd.DataFrame(results_dict['First_50'])
df_last = pd.DataFrame(results_dict['Last_50'])

# Merge on feature name
coef_compare = pd.merge(
    df_first[['feature', 'coef', 'sd']],
    df_last[['feature', 'coef', 'sd']],
    on = 'feature',
    suffixes = ('_first', '_last')
)

# Compute difference and standard error of the difference (approximate)
coef_compare['diff'] = coef_compare['coef_first'] - coef_compare['coef_last']
coef_compare['se_diff'] = np.sqrt(coef_compare['sd_first']**2 + coef_compare['sd_last']**2)

rope_min, rope_max = -0.1, 0.1

# Sort features by difference magnitude
coef_compare = coef_compare.sort_values('diff', ascending = True)

plt.figure(figsize = (8, 7))
plt.axvline(0, color = 'black', linestyle = '--', linewidth = 1)
plt.axvline(rope_min, color = 'gray', linestyle = ':')
plt.axvline(rope_max, color = 'gray', linestyle = ':')
plt.fill_betweenx(
    coef_compare['feature'],
    rope_min, rope_max,
    color = 'gray', alpha = 0.1, label = 'ROPE Region (-0.1, 0.1)'
)

# Plot coefficient differences with approximate 95% CIs
plt.errorbar(
    coef_compare['diff'], coef_compare['feature'],
    xerr = 1.96 * coef_compare['se_diff'],
    fmt = 'o', color = 'steelblue', ecolor = 'lightgray', elinewidth = 2, capsize = 3
)

plt.title('ROPE Plot: Difference in Logistic Coefficients\n(50/50 Year Split)')
plt.xlabel('Difference in logit coefficients')
plt.ylabel('Feature')
plt.legend(loc = 'lower right')
plt.grid(axis = 'x', linestyle = ':', alpha = 0.4)
plt.tight_layout()
plt.savefig('figures/boehmke2017/boehmke_coef_comparison.png', dpi = 300, bbox_inches = 'tight')
plt.show()

