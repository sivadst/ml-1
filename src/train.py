import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def prepare_data(df):
    # Separate features and targets
    # For Regression: Predict Revenue
    # For Classification: Predict Profit_Flag
    # Note: Exclude ROI and Profit_Flag from features. Exclude Revenue for Classification.
    
    # Drop IDs and Dates that are not useful for basic modeling
    drop_cols = ['Campaign_ID', 'Date', 'ROI', 'Profit_Flag']
    
    # We will handle categorical features
    cat_features = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment', 'Brand']
    
    # Features specific to Channel (already multi-label encoded)
    channel_features = [c for c in df.columns if c.startswith('Channel_')]
    
    num_features = ['Duration', 'Impressions', 'Clicks', 'Leads', 'Conversions', 'Acquisition_Cost', 'Engagement_Score']
    
    X = df[cat_features + num_features + channel_features]
    y_reg = df['Revenue']
    y_clf = df['Profit_Flag']
    
    return X, y_reg, y_clf, cat_features, num_features, channel_features

def create_preprocessor(cat_features, num_features):
    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough' # For channel_features
    )
    return preprocessor

def train_and_evaluate_models(X, y_reg, y_clf, preprocessor):
    # Train-test split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Dictionary to store results
    results = {'Regression': {}, 'Classification': {}}
    
    # Regression Models
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=50, random_state=42) # Smaller n_estimators for speed
    }

    best_reg_model = None
    best_reg_score = float('inf') # RMSE

    for name, model in reg_models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        pipeline.fit(X_train_reg, y_train_reg)
        y_pred = pipeline.predict(X_test_reg)
        rmse = root_mean_squared_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)
        results['Regression'][name] = {'RMSE': rmse, 'R2': r2}
        
        if rmse < best_reg_score:
            best_reg_score = rmse
            best_reg_model = pipeline
            best_reg_name = name

    # Classification Models
    clf_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=50, random_state=42)
    }

    best_clf_model = None
    best_clf_score = 0 # F1 Score

    for name, model in clf_models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        pipeline.fit(X_train_clf, y_train_clf)
        y_pred = pipeline.predict(X_test_clf)
        
        acc = accuracy_score(y_test_clf, y_pred)
        prec = precision_score(y_test_clf, y_pred, zero_division=0)
        rec = recall_score(y_test_clf, y_pred, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, zero_division=0)
        
        results['Classification'][name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        
        if f1 > best_clf_score:
            best_clf_score = f1
            best_clf_model = pipeline
            best_clf_name = name

    # Fallback to accuracy if F1 is identical or 0
    if best_clf_model is None:
        best_clf_model = pipeline
        best_clf_name = name

    return results, best_reg_model, best_reg_name, best_clf_model, best_clf_name

def write_evaluation_report(results, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        
        f.write("## Regression Models (Target: Revenue)\n")
        f.write("| Model | RMSE | R2 Score |\n")
        f.write("|-------|------|----------|\n")
        for name, metrics in results['Regression'].items():
            f.write(f"| {name} | {metrics['RMSE']:.4f} | {metrics['R2']:.4f} |\n")
        
        f.write("\n## Classification Models (Target: Profit_Flag)\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        for name, metrics in results['Classification'].items():
            f.write(f"| {name} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1']:.4f} |\n")

if __name__ == "__main__":
    print("Loading engineered data...")
    df = pd.read_csv('data/feature_engineered/feature_engineered_dataset.csv')
    
    print("Preparing data...")
    X, y_reg, y_clf, cat_features, num_features, channel_features = prepare_data(df)
    preprocessor = create_preprocessor(cat_features, num_features)
    
    print("Training models... (This might take a few minutes)")
    results, best_reg_model, best_reg_name, best_clf_model, best_clf_name = train_and_evaluate_models(X, y_reg, y_clf, preprocessor)
    
    print("Writing evaluation report...")
    write_evaluation_report(results, 'reports/model_evaluation_report.md')
    
    print(f"Best Regression Model: {best_reg_name}")
    print(f"Best Classification Model: {best_clf_name}")
    
    print("Saving best models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_reg_model, 'models/best_regression_model.pkl')
    joblib.dump(best_clf_model, 'models/best_classification_model.pkl')
    print("Model building and evaluation completed.")
