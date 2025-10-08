from pathlib import Path
import pandas as pd

figures_dir = Path(".")  # Current directory (figures folder)

# Find all CSV files recursively
csv_files = list(figures_dir.rglob("*.csv"))

# Create a dictionary to store DataFrames with file names as keys
dataframes = {}

for csv_file in csv_files:
    file_name = csv_file.stem
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Store in dictionary
    dataframes[file_name] = df

# Calculate averages and group by base name
results_dict = {}

for file_name, df in dataframes.items():
    # Extract base name
    base_name = file_name.replace('_logit', '').replace('_xgb', '').replace('_rf', '')
    
    if base_name not in results_dict:
        results_dict[base_name] = {'Article': base_name}
    
    if "logit" in file_name.lower():
        # Average columns 1 and 2 separately
        avg_col1 = df.iloc[:, 1].mean()
        avg_col2 = df.iloc[:, 2].mean()
        results_dict[base_name][f'{df.columns[1]}'] = avg_col1
        results_dict[base_name][f'{df.columns[2]}'] = avg_col2
        print(f"{file_name}: {df.columns[1]} = {avg_col1:.4f}, {df.columns[2]} = {avg_col2:.4f}")
    elif "xgb" in file_name.lower():
        avg_value = df.iloc[:, 1].mean()
        results_dict[base_name][f'{df.columns[1]}'] = avg_value
        print(f"{file_name}: {df.columns[1]} = {avg_value:.4f}")
    elif "rf" in file_name.lower():
        avg_value = df.iloc[:, 1].mean()
        results_dict[base_name][f'{df.columns[1]}'] = avg_value
        print(f"{file_name}: {df.columns[1]} = {avg_value:.4f}")

# Convert to list of dictionaries for DataFrame creation
results = list(results_dict.values())

# Create a summary dataframe
summary_df = pd.DataFrame(results)

# Write the summary dataframe to CSV
summary_df.to_csv('policy_averages.csv', index = False)