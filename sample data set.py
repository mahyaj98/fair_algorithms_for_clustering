import pandas as pd
import os

# Define file paths
input_file_path = 'data/adult.csv'
output_file_path = 'data/subsampled_adult_200.csv'

# Check if input file exists
if os.path.exists(input_file_path):
    # Load the dataset (assuming semicolon separator based on bank.csv preview)
    df = pd.read_csv(input_file_path, sep=';')

    # Sample 200 rows
    # Use random_state for reproducibility, replace with None if not needed
    if len(df) >= 200:
        sampled_df = df.sample(n=200, random_state=42)
    else:
        print(f"Dataset has fewer than 200 rows ({len(df)}). Saving the entire dataset.")
        sampled_df = df

    # Save to a new file
    sampled_df.to_csv(output_file_path, index=False, sep=';')
    print(f"Successfully saved {len(sampled_df)} rows to {output_file_path}")
else:
    print(f"Input file not found: {input_file_path}")