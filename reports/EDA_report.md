# Exploratory Data Analysis (EDA) Report

## Overview
This report outlines the insights gathered from analyzing the marketing campaign dataset for the brands Nykaa, Purplle, and Tira.

## Key Insights
1. **Data Distribution**: The initial dataset contained raw campaign data which has been cleaned (missing values imputed and duplicates removed).
2. **Brand Comparison**: The average revenue and acquisition costs were analyzed across different brands. The dataset includes relatively balanced distributions among the three major brands.
3. **Correlation Analysis**: A correlation matrix highlighted that `Clicks`, `Impressions`, and `Acquisition_Cost` have varying degrees of influence on `Revenue`. Notably, `Acquisition_Cost` should be monitored to maintain a positive `Profit_Flag` (ROI > 0).
4. **Channel Performance**: The multi-label binarization of `Channel_Used` allows us to isolate which channels (e.g., Email, Social Media) contribute most effectively to high ROI campaigns.

## Next Steps
The engineered features and cleaned dataset are now suitable for training predictive models for both regression (Revenue) and classification (Profit vs Loss).
