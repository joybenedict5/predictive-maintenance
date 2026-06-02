import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def generate_nasa_data(n_engines=100, max_cycles=200):
    np.random.seed(42)
    data = []
    for engine_id in range(1, n_engines + 1):
        n_cycles = np.random.randint(100, max_cycles)
        for cycle in range(1, n_cycles + 1):
            row = {'engine_id': engine_id, 'cycle': cycle}
            for i in range(1, 11):
                row[f'sensor_{i}'] = np.random.normal(100 * i, 5)
            data.append(row)
    return pd.DataFrame(data)
def engineer_features(df):
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['failure'] = (df['RUL'] <= 30).astype(int)
    df = df[df['cycle'] > df['max_cycle'] * 0.2].copy()
    for i in range(1, 11):
        sensor = f'sensor_{i}'
        df[f'{sensor}_rolling_mean'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    print(f"Dataset shape: {df.shape}")
    print(f"Failure rate: {df['failure'].mean():.2%}")
    return df
def preprocess(df):
    feature_cols = [f'sensor_{i}' for i in range(1, 11)] + \
                   [f'sensor_{i}_rolling_mean' for i in range(1, 11)]
    X = df[feature_cols]
    y = df['failure']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'pm_scaler.pkl')
    print("Scaler saved as pm_scaler.pkl")
    return X_scaled, y, feature_cols

if __name__ == "__main__":
    df = generate_nasa_data()
    df = engineer_features(df)
    X, y, features = preprocess(df)
    print(f"Features: {len(features)}")
    print(f"Samples: {len(X)}")