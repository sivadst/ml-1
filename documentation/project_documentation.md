# Project Documentation

## 1. Data Preprocessing
- **Source**: Three raw CSV files corresponding to Nykaa, Purplle, and Tira.
- **Handling NaNs**: Numerical missing values filled with the median. Categorical filled with the mode.
- **ROI Validation**: Standardized the `ROI` column using `(Revenue - Acquisition_Cost) / Acquisition_Cost`.

## 2. Feature Engineering
- **Multi-Label Encoding**: Processed `Channel_Used` using `MultiLabelBinarizer` to generate discrete `Channel_` boolean flags.
- **Target Creation**: Engineered the `Profit_Flag` binary feature based on whether ROI is > 0.

## 3. Modeling
- Regressions (Target `Revenue`): Linear Regression, Decision Tree Regressor, Random Forest Regressor. Random Forest achieved the best metrics.
- Classifiers (Target `Profit_Flag`): Logistic Regression, Decision Tree, Random Forest. Excluded features like `ROI` and `Revenue` to stop data leakage.

## 4. Evaluation
- Used RMSE and R² for Regression.
- Used Accuracy, Precision, Recall, and F1 for Classification.

## 5. App Deployment
- Streamlit script predicts single-instance outcomes based on user input, loading pickled scikit-learn models and providing real-time metric projections.
