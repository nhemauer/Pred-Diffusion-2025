import pandas as pd
import os
from pathlib import Path
import glob

def full_join_csvs_in_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Start with first dataframe
    result_df = pd.read_csv(csv_files[0])
    
    # Full join with each subsequent dataframe
    for csv_file in csv_files[1:]:
        df = pd.read_csv(csv_file)
        
        # This is the pandas equivalent of R's full_join()
        result_df = pd.merge(result_df, df, on='state', how='outer')

    return result_df

# Use absolute path or check current directory
figures_dir = Path("ml_state/figures")

# Check if path exists
if not figures_dir.exists():
    print(f"Path does not exist: {figures_dir}")
    print(f"Current working directory: {os.getcwd()}")
else:
    # Process all subfolders in figures directory
    for folder in figures_dir.iterdir():
        if folder.is_dir():
            print(f"\n=== Processing {folder.name} ===")
            combined_df = full_join_csvs_in_folder(folder)
            
            if combined_df is not None:
                # Save combined result
                last_column = combined_df.columns[-1]
                combined_df = combined_df.sort_values(by=last_column, ascending=False)
                output_file = folder / f'{folder.name}_combined.csv'
                combined_df.to_csv(output_file, index=False)
                print(f"Saved to: {output_file}")