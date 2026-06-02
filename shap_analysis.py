import joblib
import numpy as np
import pandas as pd
from feature_engineering import generate_nasa_data, engineer_features, preprocess

def run_shap():
    df = generate_nasa_data()
    df = engineer_features(df)
    X, y, features = preprocess(df)

    model = joblib.load('pm_model.pkl')

    # Get feature importances from Random Forest
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))

    # Keep only top 70% of features
    threshold = feature_importance_df['importance'].quantile(0.30)
    top_features = feature_importance_df[
        feature_importance_df['importance'] >= threshold
    ]['feature'].tolist()

    print(f"\nOriginal features : {len(features)}")
    print(f"Selected features : {len(top_features)}")
    print(f"Reduction         : {(1 - len(top_features)/len(features)):.0%}")

    joblib.dump(top_features, 'pm_top_features.pkl')
    print("Top features saved as pm_top_features.pkl")

if __name__ == "__main__":
    run_shap()