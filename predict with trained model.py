import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the prediction dataset
file_path = 'predict.csv'  # Ensure the path is correct for your local system
predict_data = pd.read_csv(file_path)

# Data Preprocessing
# Drop the index column if it exists
predict_data = predict_data.drop(columns=['index'], errors='ignore')

# Convert all data to numeric
predict_data = predict_data.apply(pd.to_numeric, errors='coerce')

# Prepare Data for Prediction
# Assuming each row represents voltage readings over time for a single battery
# We want to use the first 5 columns (0-4) for prediction
features_predict = predict_data.iloc[:, 0:4].values

#print the first 5 rows of the features_predict
print(features_predict[:5])

# Load the trained model
# model = joblib.load('battery_life_model_feature_4.pkl')

# check if the file'battery_life_model.pkl' does exist in the same folder as this python file and print the path of battery_life_model.pkl
import os
print(os.path.abspath('battery_life_model.pkl'))   

# Load the trained model
model = joblib.load('battery_life_model.pkl')

# Predict the next 5 hours for the batteries in the prediction dataset
y_pred = model.predict(features_predict)

# fill the predicted values into the predict_data dataframe and write it back to the predict.csv file
predict_data.iloc[:, 4:11] = y_pred
predict_data.to_csv('predict.csv', index=False) 


# # Display Predicted Values for each battery
# for i, prediction in enumerate(y_pred):
#     print(f"Predicted Voltage for the next 5 hours for Battery {i + 1}:")
#     print(prediction)
#     plt.figure(figsize=(10, 6))
#     plt.plot(prediction, label=f'Predicted Voltage for Battery {i + 1} (5th-10th hours)', linestyle='--')
#     plt.title(f'Predicted Voltage for Battery {i + 1} (5th-10th Hours)')
#     plt.xlabel('Time Interval (minutes)')
#     plt.ylabel('Voltage (V)')
#     plt.legend()
#     plt.show()