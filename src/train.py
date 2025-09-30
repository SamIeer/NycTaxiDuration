# src/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ========== Utility functions (could move to utils.py) ==========
def haversine(lat1, lon1, lat2, lon2, R=6371):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def prepare_features(df, is_train=True):
    # parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['rush_hour'] = df['hour'].isin([7,8,9,16,17,18,19]).astype(int)
    
    # distances
    df['haversine'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    # manhattan approximation
    df['manhattan'] = (
        np.abs(df['dropoff_longitude'] - df['pickup_longitude']) +
        np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
    )
    
    # map store_and_fwd_flag if exists
    if 'store_and_fwd_flag' in df.columns:
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N':0, 'Y':1})
    
    # drop columns
    drop_cols = ['pickup_datetime', 'dropoff_datetime', 'id']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    return df

def rmsle(y_true, y_pred):
    # ensures non-negative
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def main():
    # Paths
    train_path = os.path.join("data", "train.csv")
    model_out = os.path.join("models", "xgb_model.joblib")
    pipeline_out = os.path.join("models", "preproc_pipeline.joblib")
    
    # Create directories
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    
    # Load
    df = pd.read_csv(train_path)
    # Optionally remove outliers
    df = df[(df['trip_duration'] >= 10) & (df['trip_duration'] <= 3600)]
    
    # Prepare features
    df = prepare_features(df)
    
    # Define features & target
    target = 'trip_duration'
    features = [c for c in df.columns if c != target]
    
    X = df[features]
    y = np.log1p(df[target])
    
    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocessing pipelines
    num_cols = [c for c in X_train.columns if X_train[c].dtype in [np.int64, np.float64]]
    if 'weekday' in num_cols:
        num_cols.remove('weekday')  # treat weekday as categorical
    cat_cols = ['weekday']
    
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    # Full pipeline + model
    model = xgb.XGBRegressor(
        object
, n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    pipeline = Pipeline([
        ("preproc", preproc),
        ("xgb", model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Validate
    y_valid_pred_log = pipeline.predict(X_valid)
    y_valid_pred = np.expm1(y_valid_pred_log)
    y_valid_true = np.expm1(y_valid)
    score = rmsle(y_valid_true, y_valid_pred)
    print("Validation RMSLE:", score)
    
    # Save
    joblib.dump(pipeline, pipeline_out)
    print("Saved preprocessing + model pipeline to:", pipeline_out)

if __name__ == "__main__":
    main()
