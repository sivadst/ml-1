import pandas as pd
import numpy as np
import os

def load_and_combine_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    for f in files:
        brand_name = f.split('_')[0].capitalize()
        df = pd.read_csv(os.path.join(data_dir, f))
        df['Brand'] = brand_name
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # For numerical columns, fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def validate_roi(df):
    # ROI = (Revenue - Acquisition_Cost) / Acquisition_Cost
    # Calculate expected ROI
    expected_roi = (df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost']
    
    # Update ROI where it differs significantly from expected (allowing small float differences)
    # Since we don't know if the original ROI is a percentage or fraction, let's assume it's meant to match expected_roi
    df['ROI'] = expected_roi
    
    return df

if __name__ == "__main__":
    raw_dir = 'data/raw/'
    cleaned_dir = 'data/cleaned/'
    os.makedirs(cleaned_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_and_combine_data(raw_dir)
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    print("Validating ROI...")
    df_clean = validate_roi(df_clean)
    
    print("Saving cleaned data...")
    df_clean.to_csv(os.path.join(cleaned_dir, 'cleaned_dataset.csv'), index=False)
    print("Preprocessing completed.")
