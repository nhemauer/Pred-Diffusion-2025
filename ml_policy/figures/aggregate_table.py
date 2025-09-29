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

# Calculate averages
averages = {}
for file_name, df in dataframes.items():
    # Get the average of the second column (index 1)
    avg_value = df.iloc[:, 1].mean()
    averages[file_name] = avg_value
    print(f"{file_name}: {avg_value:.4f}")

# Create a summary dataframe
summary_df = pd.DataFrame(list(averages.items()), columns=['File', 'Column_1_Average'])
print("\nSummary DataFrame:")
print(summary_df)


# If file name contains logit, average columns 1 and 2
# else just average column 1