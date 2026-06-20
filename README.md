# Multi-Brand Marketing Campaign Performance Analysis

This project builds an end-to-end Machine Learning pipeline to predict marketing campaign **Revenue** (Regression) and **Profit/Loss** (Classification) across brands like Nykaa, Purplle, and Tira.

## Project Structure

- `data/`: Raw, cleaned, and feature-engineered datasets (ignored in git to prevent size limits)
- `src/`: Data preprocessing, feature engineering, and model training scripts
- `notebooks/`: Jupyter Notebook for Exploratory Data Analysis (EDA)
- `models/`: Saved models and encoders
- `reports/`: Markdown reports for EDA, Model Evaluation, and Business Insights
- `streamlit_app/`: Streamlit dashboard for making predictions

## Installation & Setup

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: create a requirements.txt with pandas, scikit-learn, matplotlib, seaborn, plotly, streamlit, joblib)*
3. To test the pipeline end-to-end:
   - Run `python src/preprocess.py`
   - Run `python src/feature_engineering.py`
   - Run `python src/train.py`
4. To launch the Streamlit App:
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Workflow
1. **Data Preprocessing**: Handled missing values, deduplicated records, and validated ROI.
2. **Feature Engineering**: Multi-label encoded marketing channels, created `Profit_Flag` from ROI.
3. **Model Building**: Evaluated Linear Regression, Decision Tree Regressor, and Random Forest Regressor for Revenue. Evaluated Logistic Regression, Decision Tree Classifier, and Random Forest Classifier for Profit_Flag. Excluded ROI and Revenue to prevent data leakage in classification.
4. **Deployment**: Developed an interactive Streamlit UI to forecast success metrics.

## Author
SELVASIVA
