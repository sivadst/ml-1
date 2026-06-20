import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="Marketing Campaign Predictor", layout="wide")


# ---------------------------------------------------------------------------
# Data generation & training (runs once, cached in memory)
# ---------------------------------------------------------------------------

def generate_brand_data(brand_name, n=300, rng=None):
    """Generate synthetic marketing campaign data for a single brand."""
    if rng is None:
        rng = np.random.default_rng(42)

    campaign_types = ['Awareness', 'Conversion', 'Retention', 'Consideration']
    audiences = ['Men 18-24', 'Women 25-34', 'All Ages', 'Teens']
    languages = ['English', 'Hindi', 'Regional']
    segments = ['New', 'Returning', 'Premium', 'Discount']
    channels = ['Email', 'Social Media', 'Search', 'Display', 'Influencer', 'Affiliate']

    data = {
        'Campaign_ID': [f'{brand_name[:3].upper()}_{i:04d}' for i in range(1, n + 1)],
        'Date': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d').tolist(),
        'Campaign_Type': rng.choice(campaign_types, n),
        'Target_Audience': rng.choice(audiences, n),
        'Language': rng.choice(languages, n),
        'Customer_Segment': rng.choice(segments, n),
        'Duration': rng.integers(7, 90, n),
        'Impressions': rng.integers(5000, 500000, n),
        'Clicks': rng.integers(100, 25000, n),
        'Leads': rng.integers(10, 5000, n),
        'Conversions': rng.integers(1, 1000, n),
        'Acquisition_Cost': np.round(rng.uniform(500, 50000, n), 2),
        'Engagement_Score': np.round(rng.uniform(1.0, 10.0, n), 2),
        'Channel_Used': [
            ', '.join(rng.choice(channels, size=rng.integers(1, 4), replace=False))
            for _ in range(n)
        ],
    }

    df = pd.DataFrame(data)
    df['Brand'] = brand_name.capitalize()

    base_rev = (
        df['Conversions'] * rng.uniform(50, 200, n)
        + df['Clicks'] * rng.uniform(0.5, 2.0, n)
        + df['Engagement_Score'] * rng.uniform(100, 500, n)
    )
    df['Revenue'] = np.round(np.maximum(base_rev + rng.normal(0, 500, n), 100), 2)
    df['ROI'] = np.round((df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost'], 4)
    return df


def build_dataset():
    """Generate and combine synthetic data for all three brands."""
    rng = np.random.default_rng(42)
    frames = [generate_brand_data(b, rng=rng) for b in ['nykaa', 'purplle', 'tira']]
    df = pd.concat(frames, ignore_index=True)

    # Clean
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Validate ROI
    df['ROI'] = (df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost']

    # Feature engineering
    df['Profit_Flag'] = (df['ROI'] > 0).astype(int)

    # Multi-label encode channels
    df['Channel_Used'] = df['Channel_Used'].fillna('')
    channels_list = df['Channel_Used'].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip()])

    mlb = MultiLabelBinarizer()
    channel_encoded = mlb.fit_transform(channels_list)
    channel_df = pd.DataFrame(channel_encoded, columns=[f"Channel_{c}" for c in mlb.classes_])

    df = pd.concat([df.reset_index(drop=True), channel_df.reset_index(drop=True)], axis=1)
    df = df.drop('Channel_Used', axis=1)

    return df, mlb


@st.cache_resource
def train_models():
    """Build dataset, train models, and return them (cached in memory)."""
    df, mlb = build_dataset()

    # Prepare features / targets
    cat_features = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment', 'Brand']
    num_features = ['Duration', 'Impressions', 'Clicks', 'Leads', 'Conversions',
                    'Acquisition_Cost', 'Engagement_Score']
    channel_features = [c for c in df.columns if c.startswith('Channel_')]

    X = df[cat_features + num_features + channel_features]
    y_reg = df['Revenue']
    y_clf = df['Profit_Flag']

    # Preprocessor
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
        ],
        remainder='passthrough',
    )

    # --- Regression ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    best_reg, best_rmse = None, float('inf')
    for model in [LinearRegression(),
                  DecisionTreeRegressor(random_state=42),
                  RandomForestRegressor(n_estimators=50, random_state=42)]:
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_tr, y_tr)
        rmse = root_mean_squared_error(y_te, pipe.predict(X_te))
        if rmse < best_rmse:
            best_rmse = rmse
            best_reg = pipe

    # --- Classification ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    best_clf, best_f1 = None, 0
    for model in [LogisticRegression(max_iter=1000, random_state=42),
                  DecisionTreeClassifier(random_state=42),
                  RandomForestClassifier(n_estimators=50, random_state=42)]:
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_tr, y_tr)
        f1 = f1_score(y_te, pipe.predict(X_te), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_clf = pipe

    if best_clf is None:
        best_clf = pipe  # fallback

    return best_reg, best_clf, mlb


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

reg_model, clf_model, mlb = train_models()

st.title("⚡ Multi-Brand Marketing Campaign Predictor")

st.markdown("""
This application predicts the **Revenue** and **Profitability** of marketing campaigns 
across brands like Nykaa, Purplle, and Tira.
""")

tab1, tab2 = st.tabs(["🔮 Predict Performance", "📊 Business Insights"])

with tab1:
    st.header("Campaign Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Conversion", "Retention", "Consideration"])
        target_audience = st.selectbox("Target Audience", ["Men 18-24", "Women 25-34", "All Ages", "Teens"])
        language = st.selectbox("Language", ["English", "Hindi", "Regional"])
        customer_segment = st.selectbox("Customer Segment", ["New", "Returning", "Premium", "Discount"])
        
    with col2:
        duration = st.number_input("Duration (days)", min_value=1, value=30)
        impressions = st.number_input("Impressions", min_value=1000, value=50000)
        clicks = st.number_input("Clicks", min_value=0, value=2000)
        leads = st.number_input("Leads", min_value=0, value=500)
        
    with col3:
        conversions = st.number_input("Conversions", min_value=0, value=50)
        acquisition_cost = st.number_input("Acquisition Cost ($)", min_value=1.0, value=1000.0)
        engagement_score = st.slider("Engagement Score", 0.0, 10.0, 5.0)
        channels = st.multiselect("Channels Used", mlb.classes_, default=[mlb.classes_[0]])

    if st.button("Predict Performance", type="primary"):
        # Prepare input data
        input_dict = {
            'Brand': brand,
            'Campaign_Type': campaign_type,
            'Target_Audience': target_audience,
            'Language': language,
            'Customer_Segment': customer_segment,
            'Duration': duration,
            'Impressions': impressions,
            'Clicks': clicks,
            'Leads': leads,
            'Conversions': conversions,
            'Acquisition_Cost': acquisition_cost,
            'Engagement_Score': engagement_score
        }
        
        df_input = pd.DataFrame([input_dict])
        
        # Multi-label encoding for channels
        channel_encoded = mlb.transform([channels])
        channel_df = pd.DataFrame(channel_encoded, columns=[f"Channel_{c}" for c in mlb.classes_])
        
        df_input = pd.concat([df_input, channel_df], axis=1)

        # Predict
        predicted_revenue = reg_model.predict(df_input)[0]
        predicted_profit_flag = clf_model.predict(df_input)[0]
        
        # Display Results
        st.divider()
        st.subheader("Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
            roi_est = ((predicted_revenue - acquisition_cost) / acquisition_cost) * 100
            st.metric("Estimated ROI", f"{roi_est:,.2f}%")
            
        with res_col2:
            if predicted_profit_flag == 1:
                st.success("✅ Profitable Campaign (Profit)")
            else:
                st.error("⚠️ Loss-Making Campaign (Loss)")

with tab2:
    st.header("Data-Driven Insights")
    st.markdown("""
    Based on our Exploratory Data Analysis and Model Evaluation:
    
    1. **Best Regression Model**: Random Forest Regressor
    2. **Best Classification Model**: Logistic Regression
    3. **Actionable Advice**: Ensure Acquisition Costs do not heavily outpace the historical Clicks and Impressions correlation.
    """)
    
    st.info("Check `reports/EDA_report.md` and `reports/business_insights_report.md` for full details.")
