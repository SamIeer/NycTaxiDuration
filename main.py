# =============================
# NYC Taxi Trip Duration ML Model
# =============================

# 1. Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, os, warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


# 2. Load Dataset
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print(train_df.head())

# 3. Target Transformation (log1p)
train_df['log_trip_duration'] = np.log1p(train_df['trip_duration'])

# 4. Feature Engineering
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))  # distance in km

def manhattan(lat1, lon1, lat2, lon2):
    a = haversine(lat1, lon1, lat1, lon2)
    b = haversine(lat1, lon2, lat2, lon2)
    return a + b

# Distance Features
train_df['haversine'] = haversine(train_df['pickup_latitude'], train_df['pickup_longitude'],
                                  train_df['dropoff_latitude'], train_df['dropoff_longitude'])
train_df['manhattan'] = manhattan(train_df['pickup_latitude'], train_df['pickup_longitude'],
                                  train_df['dropoff_latitude'], train_df['dropoff_longitude'])

# Time Features
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['hour']   = train_df['pickup_datetime'].dt.hour
train_df['day']    = train_df['pickup_datetime'].dt.day
train_df['weekday']= train_df['pickup_datetime'].dt.weekday
train_df['month']  = train_df['pickup_datetime'].dt.month
train_df['is_weekend'] = (train_df['weekday'] >= 5).astype(int)
train_df['rush_hour']  = train_df['hour'].isin([7,8,9,16,17,18,19]).astype(int)

# Repeat for test_df
test_df['haversine'] = haversine(test_df['pickup_latitude'], test_df['pickup_longitude'],
                                 test_df['dropoff_latitude'], test_df['dropoff_longitude'])
test_df['manhattan'] = manhattan(test_df['pickup_latitude'], test_df['pickup_longitude'],
                                 test_df['dropoff_latitude'], test_df['dropoff_longitude'])

test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
test_df['hour']   = test_df['pickup_datetime'].dt.hour
test_df['day']    = test_df['pickup_datetime'].dt.day
test_df['weekday']= test_df['pickup_datetime'].dt.weekday
test_df['month']  = test_df['pickup_datetime'].dt.month
test_df['is_weekend'] = (test_df['weekday'] >= 5).astype(int)
test_df['rush_hour']  = test_df['hour'].isin([7,8,9,16,17,18,19]).astype(int)

# 5. Feature Selection
features = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude',
            'dropoff_longitude','dropoff_latitude','haversine','manhattan',
            'hour','day','weekday','month','is_weekend','rush_hour']

X = train_df[features]
y = train_df['log_trip_duration']
X_test = test_df[features]

# 6. Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training (XGBoost)
model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_valid)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred)))
print("Validation RMSLE:", rmsle)

# 9. Final Training on Full Data
model.fit(X, y)

# 10. Prediction on Test Data
test_pred = model.predict(X_test)
test_pred = np.expm1(test_pred)  # inverse log transform

# 11. Save Submission
submission = pd.DataFrame({"id": test_df["id"], "trip_duration": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
