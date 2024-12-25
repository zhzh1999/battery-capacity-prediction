import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'transposed_battery_data_500 v2.csv'  # Ensure the path is correct for your local system
data = pd.read_csv(file_path)

# Data Preprocessing
# Drop the index column if it exists
data = data.drop(columns=['index'], errors='ignore')

# Convert all data to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Print information about NaN values
print("Columns with NaN values:")
print(data.isna().sum())

# Prepare Data for Model Training
# Assuming each row represents voltage readings over time for a single battery
# We want to use the first 4 columns (0-3) to predict the next 7 columns (4-10)
training_data = data  # Use all available data for training
#print the first 5 rows of the training data
print(training_data.head()) 

# it is 4 because 0 is the index of the column and 1:4 is the number of features we are using to predict the next 7 columns   
features = training_data.iloc[:, 0:4].values

#print the first 5 rows of the features
print(features[:5])  # This will print the first 5 rows

# Adjust target values to be 0.1% lower
target = training_data.iloc[:, 4:11].values * 0.999

#print the first 5 rows of the target
print(target[:5])

# Handle NaN values using SimpleImputer - optional
# imputer = SimpleImputer(strategy='mean')
#features = imputer.fit_transform(features)

# Adjust target values to be 0.2% lower
# target = imputer.fit_transform(target) 


# Check if there is sufficient data to proceed
if len(features) == 0 or len(target) == 0:
    print("Not enough data to create training and testing sets. Please check the dataset.")
else:
    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Hyperparameter Tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=30, cv=5, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Best Model Training
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Cross-Validation to Validate Performance
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    avg_cv_mse = -np.mean(cv_scores)
    print(f"Average Cross-Validation MSE: {avg_cv_mse}")

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

    # Save the trained model with compression
    #joblib.dump(best_model, 'battery_life_model_feature_4.pkl', compress=3)
    #try to save without compression to see if the mse get better
    joblib.dump(best_model, 'battery_life_model_feature_4.pkl')
    print("Model saved as 'battery_life_model_feature_4.pkl' with compression.")
