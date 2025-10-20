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
berry_berry1990_full = pd.read_csv("data/berry_berry1990.txt", delim_whitespace = True, header = None)

os.chdir("ml_coef_split")

# Covariates
berry_berry1990_full.columns = ["state", "year", "adopt", "fiscal_1", "party", "elect1", "elect2", "income_1", "neighbor", "nbrpercn", "religion"]

berry_berry1990 = berry_berry1990_full[berry_berry1990_full['party'] != 9].copy() # 9 is the NA (For MN and NE)

# Get unique year values and split into 50/50
unique_styears = sorted(berry_berry1990['year'].unique())
n_styears = len(unique_styears)
split_point = int(n_styears * 0.5)

# Define ranges for each split
first_50_styears = unique_styears[:split_point]
last_50_styears = unique_styears[split_point:]

# Split data by 50/50
splits = {
    'First_50': berry_berry1990[berry_berry1990['year'].isin(first_50_styears)],
    'Last_50': berry_berry1990[berry_berry1990['year'].isin(last_50_styears)],
    'Full_Dataset': berry_berry1990
}

# Store results for comparison
results_dict = {}

# Run logistic regression for each split
for split_name, data in splits.items():
        # Define X and y
        X = data.drop(columns = ['adopt', 'neighbor', 'state', 'year'])
        X = sm.add_constant(X)
        y = data['adopt']
        
        # Fit Logistic Regression model with clustering
        logistic = sm.Logit(y.astype(float), X.astype(float)).fit()
        
        # Extract summary table
        summary_df = logistic.summary2().tables[1]
        
        # Filter out dummy variables
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
plt.savefig('figures/berry_berry1990/berry_coef_comparison.png', dpi = 300, bbox_inches = 'tight')
plt.show()

