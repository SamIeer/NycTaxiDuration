import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer #For creating Full Pipelines

# Train and Test Models on the Training Set
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , root_mean_squared_error
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv("data/train.csv")  # <-- check folder name & case!
print(df.head())                          # ✅ shows first 5 rows


#Shuffling the data Getting Train and Test 
def shuffle_and_split(data, test_ratio):
    np.random.seed(42) # set the seed for reproducibility
    shuffled_indices = np.random.permutation(len(data)) # this return the shuffled indices
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train , test = shuffle_and_split(df, 0.4)
df = train.copy()


# For Getting the Distance From Cordinates
locs = pd.DataFrame({
    'lat1':df['pickup_longitude'],
    'lon1':df['pickup_latitude'],
    'lat2':df['dropoff_longitude'],
    'lon2':df['pickup_latitude']    })
    
    # Haversine vector 

def haversine_vector(lat1, lon1, lat2, lon2, radius=6371):
    """
    Calculate the great-circle distance between two sets of coordinates
    using the Haversine formula (vectorized).

    Parameters
    ----------
    lat1, lon1 : array-like
        Latitudes & longitudes of the first set of points (in degrees).
    lat2, lon2 : array-like
        Latitudes & longitudes of the second set of points (in degrees).
    radius : float
        Earth radius in kilometers (default 6371 km). Use 3956 for miles.

    Returns
    -------
    numpy.ndarray
        Distances in kilometers (same shape as input arrays).
    """
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c
df['distance_km'] = haversine_vector(locs.lat1, locs.lon1, locs.lat2, locs.lon2)


# Converting the Pickup time to timestamp 
# and pickup date to week days
def add_day_and_duration(data,pic_time):
    data[pic_time] = pd.to_datetime(data[pic_time])
    data['week_day'] = data[pic_time].dt.day_name()
    return data
df = add_day_and_duration(df, 'pickup_datetime')

# Timestamp
df['timestamp'] = df['pickup_datetime'].astype('int64') // 10**9  # seconds


#Making a Copies
temp = train.copy()
train = df.copy()
df = temp.copy()


#Extracting the "id" from id column  and Dropinmg some columns
train['id'] = train['id'].str.extract("(\d+)").astype(int)

train.drop(["pickup_datetime","dropoff_datetime", "store_and_fwd_flag" ],axis=1,inplace=True)

# #Seperating Features and Labels Form train Dataset
train_features = train.drop("trip_duration", axis=1)
train_labels = train["trip_duration"].copy()


# Save to CSV
train_features.to_csv("data/input.csv", index=False)




#Creating the pipelines for the Numerical and Categorical data
#Seperate numerical and categorical columns 
num_attribs = train_features.drop("week_day", axis=1)
cat_attribs = train_features[["week_day"]]

#PipeLInes
# NUmerical pipeline
# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
# ])

# #Categorical pipeline
# cat_pipeline = Pipeline([
#     # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
#     ("onehot", OneHotEncoder(handle_unknown="ignore"))
# ])

# # Full Pipeline 
# full_pipeline = ColumnTransformer([
#     ("num", num_pipeline, num_attribs.columns),
#     ("cat", cat_pipeline, cat_attribs.columns)
# ])

# # Transform the data
# train_prepared = full_pipeline.fit_transform(train_features)
# train_prepared.shape


# #Training and Testing the Data


# #Linear Regression
# lin_reg = LinearRegression()
# lin_reg.fit(train_prepared, train_labels)

# # Decission Tree
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(train_prepared,train_labels)

# # Random Forest
# forest_reg = RandomForestRegressor(
#     n_estimators=50,      # default 100
#     max_depth=15,         # limit depth
#     max_features='sqrt',  # use fewer features per split
#     n_jobs=-1,
#     random_state=42)
# forest_reg.fit(train_prepared, train_labels)


# # Predict using training data
# lin_preds = lin_reg.predict(train_prepared)
# tree_preds = tree_reg.predict(train_prepared)
# forest_preds = forest_reg.predict(train_prepared)

# # Calculate RMSE
# lin_rmse = -cross_val_score(
#     lin_reg,
#     train_prepared,
#     train_labels,
#     scoring="neg_root_mean_squared_error",
#     cv=10)

# tree_rmse = -cross_val_score(
#     tree_reg,
#     train_prepared,
#     train_labels,
#     scoring="neg_root_mean_squared_error",
#     cv=10)

# forest_rmse = -cross_val_score(
#     forest_reg,
#     train_prepared,
#     train_labels,
#     scoring="neg_root_mean_squared_error",
#     cv=10)


# # Evaluate Decision Tree with cross-validation


# # WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# # We use minus (-) to convert it back to positive RMSE.
# print("Linear Regrosser CV RMSEs:", lin_rmse)
# print("\nCross-Validation Performance (Decision Tree):")
# print(pd.Series(lin_rmse).describe())

# print("Decision Tree CV RMSEs:", tree_rmse)
# print("\nCross-Validation Performance (Decision Tree):")
# print(pd.Series(tree_rmse).describe())

# print("Random Forest CV RMSEs:", forest_rmse)
# print("\nCross-Validation Performance (Decision Tree):")
# print(pd.Series(forest_rmse).describe())
