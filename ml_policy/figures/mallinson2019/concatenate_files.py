import pandas as pd
import glob
import os

# Get all CSV files in the policy_files directory
csv_files = glob.glob("ml_policy/figures/mallinson2019/policy_files/*.csv")

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Calculate average rf_ap_score
average_score = combined_df['rf_ap_score'].mean()

# Save combined data
combined_df.to_csv("ml_policy/figures/mallinson2019/mallinson_policy_results_rf.csv", index = False)