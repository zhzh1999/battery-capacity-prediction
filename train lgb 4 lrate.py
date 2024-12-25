import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import joblib
import lightgbm as lgb

# Load the dataset
file_path = 'transposed_battery_data_500 v2.csv'  # Ensure the path is correct for your local system
data = pd.read_csv(file_path)

# Data Preprocessing
# Drop the index column if it exists
data = data.drop(columns=['index'], errors='ignore')

# Convert all data to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Prepare Data for Model Training
# Assuming each row represents voltage readings over time for a single battery
# We want to use the first 4 columns (0-3) to predict the next 7 columns (4-10)
training_data = data  # Use all available data for training
features = training_data.iloc[:, 0:4].values

# Adjust target values to be 1% lower
target = training_data.iloc[:, 4:11].values * 0.999

# Check if there is sufficient data to proceed
if len(features) == 0 or len(target) == 0:
    print("Not enough data to create training and testing sets. Please check the dataset.")
else:
    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the LightGBM model without hyperparameter tuning
    lgb_reg = lgb.LGBMRegressor(num_leaves=50, learning_rate=0.05, min_data_in_leaf=10, lambda_l1=0.0, lambda_l2=0.0, random_state=42)
    multi_output_reg = MultiOutputRegressor(lgb_reg)
    multi_output_reg.fit(X_train, y_train)

    # Cross-Validation to Validate Performance
    cv_scores = cross_val_score(multi_output_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    avg_cv_mse = -np.mean(cv_scores)
    print(f"Average Cross-Validation MSE: {avg_cv_mse}")

    # Evaluate the model on the test set
    y_pred = multi_output_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

    # Save the trained model with compression
    joblib.dump(multi_output_reg, 'battery_life_model.pkl', compress=3)
    print("Model saved as 'battery_life_model.pkl' with compression.")
