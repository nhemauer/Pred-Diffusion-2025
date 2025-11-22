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

def handle_messy_folders(folder_path):
    """Handle kreitzer_boehmke2016, lacombe_boehmke2021, and mallinson2019 folders"""
    folder_name = folder_path.name
    print(f"Special processing for {folder_name}")
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None
    
    # First, group files by base model name (e.g., "xgb_5" and "xgb_10" both go to "xgb")
    model_groups = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).replace('.csv', '')
        
        # Check if it's a numbered model file (like xgb_5, logit_10, etc.)
        if '_' in filename and filename.split('_')[-1].isdigit():
            base_model = '_'.join(filename.split('_')[:-1])  # Remove the number part
            
            if base_model not in model_groups:
                model_groups[base_model] = []
            model_groups[base_model].append(csv_file)
        else:
            # Files that don't match the pattern go into their own group
            if filename not in model_groups:
                model_groups[filename] = []
            model_groups[filename].append(csv_file)
    
    # Combine files within each model group first
    pre_combined_files = {}
    
    for model_name, files in model_groups.items():
        if len(files) > 1:
            print(f"Combining {len(files)} files for model: {model_name}")
            
            # Read all dataframes and concatenate them (rbind equivalent)
            dfs_to_combine = []
            for csv_file in files:
                df = pd.read_csv(csv_file)
                dfs_to_combine.append(df)
            
            # Concatenate all dataframes vertically (equivalent to R's rbind)
            result_df = pd.concat(dfs_to_combine, ignore_index=True)
            
            # Save the pre-combined file temporarily
            temp_file = folder_path / f'{model_name}_precombined.csv'
            result_df.to_csv(temp_file, index=False)
            pre_combined_files[model_name] = str(temp_file)
        else:
            # Single file, use as-is
            pre_combined_files[model_name] = files[0]
    
    # Now group by author and model type using the pre-combined files
    file_groups = {}
    
    for model_name, file_path in pre_combined_files.items():
        filename = os.path.basename(file_path).replace('_precombined.csv', '').replace('.csv', '')
        
        # Look for pattern: AUTHORNAME_state_results_MODEL
        if '_state_results_' in filename:
            # Split to get author and model parts
            parts = filename.split('_state_results_')
            if len(parts) == 2:
                author = parts[0]
                model_part = parts[1]
                
                group_key = f"{author}_state_results_{model_part}"
                
                if group_key not in file_groups:
                    file_groups[group_key] = []
                file_groups[group_key].append(file_path)
    
    # Process each author/model group
    combined_results = {}
    
    for group_name, files in file_groups.items():
        print(f"Processing group: {group_name} with {len(files)} files")
        
        if not files:
            continue
            
        # Start with first dataframe in group
        result_df = pd.read_csv(files[0])
        
        # Full join with each subsequent dataframe in the group
        for csv_file in files[1:]:
            df = pd.read_csv(csv_file)
            result_df = pd.merge(result_df, df, on='state', how='outer')
        
        combined_results[group_name] = result_df
    
    # Combine all the grouped results into one final dataframe
    if combined_results:
        # Get the first dataframe as the base
        final_result_df = None
        
        for group_name, df in combined_results.items():
            if final_result_df is None:
                final_result_df = df
            else:
                final_result_df = pd.merge(final_result_df, df, on='state', how='outer')
        
        # Sort by last column
        if final_result_df is not None:
            last_column = final_result_df.columns[-1]
            final_result_df = final_result_df.sort_values(by=last_column, ascending=False)
    
    # Clean up temporary pre-combined files
    for model_name, file_path in pre_combined_files.items():
        if '_precombined.csv' in file_path:
            try:
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
            except:
                pass
    
    return final_result_df if 'final_result_df' in locals() else None

# Use absolute path or check current directory
figures_dir = Path("ml_state/figures")

# Folders to handle separately
messy_folders = {'boushey2016', 'kreitzer_boehmke2016', 'lacombe_boehmke2021', 'mallinson2019'}
clean_folders = {'berry_berry1990', 'mallinson_lovell2022'}

# Check if path exists
if not figures_dir.exists():
    print(f"Path does not exist: {figures_dir}")
    print(f"Current working directory: {os.getcwd()}")
else:
    # Process all subfolders in figures directory
    for folder in figures_dir.iterdir():
        if folder.is_dir():
            if folder.name in messy_folders:
                print(f"\n=== Special processing for {folder.name} ===")
                combined_df = handle_messy_folders(folder)
                # Add any special saving logic here if needed
            elif folder.name in clean_folders:
                print(f"\n=== Processing {folder.name} ===")
                continue
            else:
                print(f"\n=== Processing {folder.name} ===")
                combined_df = full_join_csvs_in_folder(folder)
                
            if combined_df is not None:
                # Calculate averages for all numeric columns
                averages = combined_df.select_dtypes(include='number').mean()
                print(f"Average scores by model for {folder.name}:")
                print(averages)
                
                # # Save combined result
                # last_column = combined_df.columns[-1]
                # combined_df = combined_df.sort_values(by=last_column, ascending=False)
                # output_file = folder / f'{folder.name}_combined.csv'
                # combined_df.to_csv(output_file, index=False)
                # print(f"Saved to: {output_file}")
                
                # Save averages to separate file
                averages_df = pd.DataFrame({'model': averages.index, 'average_score': averages.values})
                averages_df = averages_df.sort_values('average_score', ascending=False)
                avg_output_file = folder / f'{folder.name}_averages.csv'
                averages_df.to_csv(avg_output_file, index=False)
                print(f"Model averages saved to: {avg_output_file}")