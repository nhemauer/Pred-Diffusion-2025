import pandas as pd
import os
import re
from pathlib import Path
import glob


def combine_state_files(state_files_dir, output_dir):
    """
    Combine per-state replication files from state_files/ into one results file
    per model type. Each input file is named like:
      {author}_state_results_{model}_{index}.csv
    and contains a single state's result.

    Returns a dict of {model_key: DataFrame} for all models processed.
    """
    csv_files = glob.glob(os.path.join(state_files_dir, '*.csv'))

    if not csv_files:
        return {}

    # Group files by (author, model) key, e.g. "boushey_state_results_xgb"
    model_groups = {}
    pattern = re.compile(r'^(.+_state_results_[a-zA-Z]+)_\d+$')

    for csv_file in csv_files:
        stem = Path(csv_file).stem  # filename without extension
        match = pattern.match(stem)
        if match:
            key = match.group(1)
            model_groups.setdefault(key, []).append(csv_file)

    combined = {}
    for key, files in model_groups.items():
        # Sort by the trailing index so states come out in order
        files.sort(key=lambda f: int(re.search(r'_(\d+)$', Path(f).stem).group(1)))

        dfs = [pd.read_csv(f) for f in files]
        result_df = pd.concat(dfs, ignore_index=True)

        out_path = output_dir / f'{key}.csv'
        result_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        combined[key] = result_df

    return combined


def compute_averages(folder, model_dfs):
    """
    Compute the mean of the numeric score column for each model DataFrame.
    Saves a per-folder *_averages.csv and returns a dict of {model_label: mean}.

    Handles two formats:
      1. Standard: columns like rf_ap_score, xgb_ap_score — use column name as label.
      2. model/overall_ap_score: a 'model' column names the model and
         'overall_ap_score' holds the value — group by model and rename to
         {model}_ap_score.

    model_dfs: dict of {key: DataFrame}
    Returns: dict of {model_label: average_score}
    """
    rows = []
    for _, df in model_dfs.items():
        if 'model' in df.columns and 'overall_ap_score' in df.columns:
            for model_name, group in df.groupby('model'):
                rows.append({
                    'model': f'{model_name}_ap_score',
                    'average_score': group['overall_ap_score'].mean()
                })
        else:
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                rows.append({'model': col, 'average_score': df[col].mean()})

    if not rows:
        return {}

    averages_df = pd.DataFrame(rows).sort_values('average_score', ascending=False)
    out_path = folder / f'{folder.name}_averages.csv'
    averages_df.to_csv(out_path, index=False)
    print(f"  Averages saved: {out_path}")
    print(averages_df.to_string(index=False))

    return dict(zip(averages_df['model'], averages_df['average_score']))


def load_existing_results(folder):
    """
    Load any *_state_results_*.csv files already present in folder
    (not inside state_files/) that were not generated from state_files.
    """
    existing = {}
    for csv_file in folder.glob('*_state_results_*.csv'):
        key = csv_file.stem
        existing[key] = pd.read_csv(csv_file)
    return existing


# ---------------------------------------------------------------------------

SCORE_COLS = ['original_ap_score', 'logit_ap_score', 'rf_ap_score', 'xgb_ap_score']

figures_dir = Path("ml_state/figures")

if not figures_dir.exists():
    print(f"Path does not exist: {figures_dir}")
    print(f"Current working directory: {os.getcwd()}")
else:
    summary_rows = []

    for folder in sorted(figures_dir.iterdir()):
        if not folder.is_dir():
            continue

        print(f"\n=== {folder.name} ===")
        state_files_dir = folder / 'state_files'

        # Start with any results files already sitting at the folder level
        all_model_dfs = load_existing_results(folder)

        # If there is a state_files sub-directory, combine those files
        if state_files_dir.is_dir():
            generated = combine_state_files(state_files_dir, folder)
            all_model_dfs.update(generated)

        if not all_model_dfs:
            print("  No results files found.")
            continue

        averages = compute_averages(folder, all_model_dfs)

        # Derive article label from the folder's result file stems, e.g.
        # "berry_state_results_logit" -> "berry_state_results"
        stems = list(all_model_dfs.keys())
        article = re.sub(r'_(logit|rf|xgb|original)$', '', stems[0])
        row = {'Article': article}
        for col in SCORE_COLS:
            row[col] = averages.get(col, None)
        summary_rows.append(row)

    # Write the combined summary table
    summary_df = pd.DataFrame(summary_rows, columns=['Article'] + SCORE_COLS)
    summary_path = figures_dir / 'state_averages.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved: {summary_path}")
    print(summary_df.to_string(index=False))
