import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

def engineer_features(df):
    # 1. Create Profit_Flag from ROI
    # If ROI > 0, Profit_Flag = 1, else 0
    df['Profit_Flag'] = (df['ROI'] > 0).astype(int)
    
    # 2. Multi-label encoding for Channel_Used
    # Assuming Channel_Used contains comma-separated values like 'Email, Social Media, Search'
    if 'Channel_Used' in df.columns:
        # Fill missing with empty string just in case, though it should be cleaned
        df['Channel_Used'] = df['Channel_Used'].fillna('')
        
        # Split channels and strip whitespace
        channels_list = df['Channel_Used'].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip()])
        
        mlb = MultiLabelBinarizer()
        channel_encoded = mlb.fit_transform(channels_list)
        
        # Create a DataFrame for encoded columns
        channel_df = pd.DataFrame(channel_encoded, columns=[f"Channel_{c}" for c in mlb.classes_])
        
        # Save the mlb for later use in prediction
        os.makedirs('models', exist_ok=True)
        joblib.dump(mlb, 'models/mlb_encoder.pkl')
        
        # Concatenate and drop original Channel_Used
        df = pd.concat([df.reset_index(drop=True), channel_df.reset_index(drop=True)], axis=1)
        df = df.drop('Channel_Used', axis=1)
        
    return df

if __name__ == "__main__":
    cleaned_path = 'data/cleaned/cleaned_dataset.csv'
    engineered_dir = 'data/feature_engineered/'
    os.makedirs(engineered_dir, exist_ok=True)
    
    print("Loading cleaned data...")
    df = pd.read_csv(cleaned_path)
    
    print("Engineering features...")
    df_eng = engineer_features(df)
    
    print("Saving feature engineered data...")
    df_eng.to_csv(os.path.join(engineered_dir, 'feature_engineered_dataset.csv'), index=False)
    print("Feature engineering completed.")
