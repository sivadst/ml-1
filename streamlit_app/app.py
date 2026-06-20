import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Marketing Campaign Predictor", layout="wide")

@st.cache_resource
def load_models():
    reg_model = joblib.load('models/best_regression_model.pkl')
    clf_model = joblib.load('models/best_classification_model.pkl')
    mlb = joblib.load('models/mlb_encoder.pkl')
    return reg_model, clf_model, mlb

reg_model, clf_model, mlb = load_models()

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
        
        # Ensure all required channel columns exist, even if not selected
        # (Though mlb.transform handles this)
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
