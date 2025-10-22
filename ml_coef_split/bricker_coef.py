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

os.chdir("ml_coef_split")

# Data
bricker_lacombe_2021 = pd.read_csv(
    r"figures/bricker_lacombe2021/bricker_stata_results.csv", 
    dtype = str 
)

# Clean the data by removing the =" and " from each cell (Stata...)
for col in bricker_lacombe_2021.columns:
    bricker_lacombe_2021[col] = bricker_lacombe_2021[col].str.replace('="', '').str.replace('"', '')

# Grab needed data
selected_rows = pd.concat([
    bricker_lacombe_2021.iloc[3:14], 
    bricker_lacombe_2021.iloc[[39]]   
])

# Reset index
coef_compare = selected_rows.reset_index(drop = True)

# Rename columns
coef_compare = coef_compare.iloc[:, :5]
coef_compare.columns = ['feature', 'coef_first', 'sd_first', 'coef_last', 'sd_last']

# Convert numeric columns, handling empty strings and dots
numeric_cols = ['coef_first', 'sd_first', 'coef_last', 'sd_last']
for col in numeric_cols:
    coef_compare[col] = coef_compare[col].replace(['', '.'], np.nan)
    coef_compare[col] = pd.to_numeric(coef_compare[col], errors='coerce')

print(coef_compare)

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
plt.savefig('figures/bricker_lacombe2021/bricker_coef_comparison.png', dpi = 300, bbox_inches = 'tight')
plt.show()