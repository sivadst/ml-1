"""Generate synthetic marketing campaign data for Nykaa, Purplle, and Tira."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_brand_data(brand_name, n=300):
    campaign_types = ['Awareness', 'Conversion', 'Retention', 'Consideration']
    audiences = ['Men 18-24', 'Women 25-34', 'All Ages', 'Teens']
    languages = ['English', 'Hindi', 'Regional']
    segments = ['New', 'Returning', 'Premium', 'Discount']
    channels = ['Email', 'Social Media', 'Search', 'Display', 'Influencer', 'Affiliate']

    data = {
        'Campaign_ID': [f'{brand_name[:3].upper()}_{i:04d}' for i in range(1, n + 1)],
        'Date': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d').tolist(),
        'Campaign_Type': np.random.choice(campaign_types, n),
        'Target_Audience': np.random.choice(audiences, n),
        'Language': np.random.choice(languages, n),
        'Customer_Segment': np.random.choice(segments, n),
        'Duration': np.random.randint(7, 90, n),
        'Impressions': np.random.randint(5000, 500000, n),
        'Clicks': np.random.randint(100, 25000, n),
        'Leads': np.random.randint(10, 5000, n),
        'Conversions': np.random.randint(1, 1000, n),
        'Acquisition_Cost': np.round(np.random.uniform(500, 50000, n), 2),
        'Engagement_Score': np.round(np.random.uniform(1.0, 10.0, n), 2),
        'Channel_Used': [', '.join(np.random.choice(channels, size=np.random.randint(1, 4), replace=False)) for _ in range(n)],
    }

    df = pd.DataFrame(data)

    # Generate Revenue with some correlation to metrics
    base_rev = (df['Conversions'] * np.random.uniform(50, 200, n)
                + df['Clicks'] * np.random.uniform(0.5, 2.0, n)
                + df['Engagement_Score'] * np.random.uniform(100, 500, n))
    noise = np.random.normal(0, 500, n)
    df['Revenue'] = np.round(np.maximum(base_rev + noise, 100), 2)

    # ROI = (Revenue - Acquisition_Cost) / Acquisition_Cost
    df['ROI'] = np.round((df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost'], 4)

    return df

os.makedirs('data/raw', exist_ok=True)

for brand in ['nykaa', 'purplle', 'tira']:
    df = generate_brand_data(brand)
    df.to_csv(f'data/raw/{brand}_campaigns.csv', index=False)
    print(f"Generated {len(df)} rows for {brand}")

print("Synthetic data generation complete!")
