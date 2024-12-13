Battery prediction



0 Project Requirements



1. Technical goal: By analyzing the battery voltage data of the first 3 hours of battery core capacity discharge , predict the single cell voltage value in the next 7 hours and predict faults with a prediction accuracy of more than 90% , and apply the model to EMT monitoring equipment.

2. Technical content: (1) Select the correct model according to Party A's needs; (2) Conduct model training; (3) Provide prediction analysis results; (4) Debug the model when the prediction accuracy is insufficient; (5) Provide guidance on the application of the model in EMT.





1 Project analysis and data analysis

1.1 Project Analysis

Analysis: Time series prediction tasks;



1. Based on the battery voltage data of the first three hours, that is, from the four data moments 0, 1, 2, and 3, predict the voltage value of the next 7 hours.

2. And judge whether it is a usable battery according to the voltage value of the 10th hour, whether it is greater than or less than 1.8. A normal battery has a voltage greater than 1.8v in 10 hours, and an abnormal battery has a voltage less than 1.8v in 10 hours;





1.2 Analysis of battery capacity data samples



1.2.1 Data volume:

Training data: 500 data samples for training AI models;

(400 of them are qualified data and 100 are unqualified data)



Test data: 40 test samples to test the accuracy of the AI model;

(20 of them are qualified data, 20 are unqualified data, and they are not repeated in the previous 500)



1.2.2 Training Data Analysis



Visualize 500 training data:



From the above analysis chart, we can see that

Q1: There is an abnormal value, the voltage is over 200;

Checking the table, we can see that the data in item 268 is abnormal: remove the abnormal data; and display it visually;







Q2: Initial voltage abnormality: The voltage value at the first moment is greater than 2.2v: Screening:





Visual display:

Q3: When the voltage is lower than 1.8V, no sampling is performed, which is not in line with objective laws if you want to fit the curve.

So we can sample the data 10 hours ago, when the voltage value is 1.8v, according to the slope at that moment to get the value at the next moment. Or we can re-interpolate according to the regression model;









400 data visualizations:

Why do most data have a significant turning point in the second hour?



It can be clearly seen from the figure that for a normal battery, the discharge slope is relatively stable; for an abnormal battery, the initial voltage is lower, but the discharge slope is larger; this is in line with the real physical laws.





The first 100 cases of data:

Q5: Some data at the later moment is higher than that at the previous moment;

Need to be screened out; these data are equivalent to noise data and will affect the training of the model. The higher the quality of the training data, the better the performance of the model.





The first 50 cases of normal data:







Abnormal battery voltage data:

Q6: The data changes dramatically and does not change when it is below 1.8V.













40 test data:

Q: Is the slope of the unqualified battery below 1.8v the real value?

Therefore, the accuracy of the predicted value of the data after 7 hours is not that great, and the focus is on the accuracy of the voltage value at the 10th hour.

And in line with the principle of not letting go of the wrong ones, when it is very close to 1.8, it may also be judged as unqualified.

In fact, the test data here is lower than 1.8, or around 1.8, and it is also filled in manually , which does not conform to the actual rules .











1.3 Data preprocessing:



1 Delete the data whose later moment is higher than the previous moment:

Before and after deletion:

Check: There are the following data:

[12, 12, 105, 106, 154, 154, 275, 275, 276, 276, 277, 309, 334, 334, 336, 337, 338, 338, 357, 358, 359, 360, 397, 411, 412, 414, 415, 415, 436, 453]

30

[3, 8, 2, 2, 3, 7, 2, 5, 2, 5, 2, 1, 1, 6, 1, 1, 6, 10, 1, 1, 1, 1, 10, 1, 1, 1, 1, 3, 1, 1]

30





2 For the voltage that hovered around 1.8 before 10 hours, the training data was regressed and interpolated:



For the voltage that hovered around 1.8 before 10 hours, regression interpolation was performed and the minimum value was 0;



Method 1: Use the simplest method. When a value is around 1.8, start from the current value and subtract 0.05 each time.

However, as can be seen from the figure below, the slope of this area is still problematic and does not conform to the normal decline law;





The effect of training rf:



train:

Mean Squared Error on test Set: 0.00106514857381888

Accuracy: 0.967741935483871

Precision: 0.9726027397260274

Recall: 0.9861111111111112

F1 Score: 0.9793103448275863



Test set:

MSE: 0.0005896421954737117

Accuracy: 0.85

Precision: 0.9375

Recall: 0.75

F1 Score: 0.8333333333333334



40 test results: actual on the left, predicted on the right;







A higher precision means that the model is more willing to predict 1 when it is near 0/1. We want to be more willing to predict 0.

Therefore, we will further refine the preprocessing of training data and perform data augmentation for abnormal data.



Conduct a keys study:





I think the problem lies in the data at point 2. If the slope changes for no reason at point 2, the final result will be 1.8 or greater than 1.8. So in a sense, the test data also needs to be interpolated using the same method.







3 Test data interpolation:

















1.4 Data augmentation:













2 Model training and testing- Machine learning



Data collection : Collect data for training the model. This data can come from databases, files, the Internet, or other data sources.

Data preprocessing :

Cleaning data : handling missing values, outliers, and duplicate data.

Feature selection : Select features that are useful for the model.

Feature Engineering : Creating new features or transforming existing features to improve model performance.

Data standardization/normalization : Scaling the data to the same scale to prevent certain features from having a disproportionate impact on the model due to their large value range.

Divide the dataset : Divide the dataset into training set, validation set and test set. The training set is used to train the model, the validation set is used for model selection and hyperparameter adjustment, and the test set is used to finally evaluate the model performance.

Select the model : Choose the appropriate machine learning algorithm based on the problem type. Common algorithms include decision trees, support vector machines, neural networks, random forests, etc.

Train Model : Use the training set data to train the selected model.

Model Evaluation :

Use the validation set to evaluate the performance of different models and select the best model.

Adjust the model's hyperparameters to optimize performance.

Model tuning : Find the optimal parameters of the model through cross-validation, grid search and other methods.

Model Validation : Use the test set to evaluate the performance of the final model and ensure that the model is not overfitting or underfitting.







2.1 Model training effect before data preprocessing:



2.1.1 Dividing the Dataset

Divide the current training data into training and validation sets according to a 4:1 ratio:

The predicted data is the test data;



2.1.2 Model Performance Evaluation



Evaluation Metrics: Time Series Prediction Task:

1. The gap between the predicted value and the true value in the last 7 hours: the smaller the gap, the more accurate the prediction ; regression problem:

2. Classification of the predicted value of 10 hours. If it is less than 1.8, it is 0, and if it is higher than 1.8, it is the true value. This is equivalent to a classification task. It can be represented by values such as acc, tpr, and recall. Classification problem:





4-10h prediction performance evaluation indicators:



Mean Absolute Error (MAE) : The average of the absolute values of the differences between the predicted values and the actual values. It measures the average size of the prediction error, but does not consider the direction of the error.



Mean Squared Error (MSE) : The average of the squares of the differences between the predicted values and the actual values. MSE gives a greater penalty for large prediction errors.













10h Whether the predicted result is 0/1 correct classification indicator:







2.1.3 Random Forest

Random Forest is an ensemble learning method that builds multiple decision trees for classification or regression prediction. The core idea of Random Forest is "brainstorming", that is, improving the overall prediction accuracy and robustness by combining the prediction results of multiple models.





1. Training and validation:



According to the 4:1 random division, train the random forest:

Use the first 0-3 data as x, and use 4-10 as predicted y;





predict:

Training: MSE: 0.00033746353177420104

Verification: MSE: 0.00016454365824939



Classification: 1.8 is used as the threshold;



Validation set;

Accuracy: 0.9591836734693877

Precision: 0.9473684210526315

Recall: 1.0

F1 Score: 0.972972972972973



Saved Model:

battery_life_model_feature_4_test1.pkl



2 Test set test:



40 test data:

MSE: 0.0005271824049329601

Accuracy: 0.775

Precision: 0.7619047619047619

Recall: 0.8

F1 Score: 0.7804878048780488





Visual display of test data:

Visualization of 40 data from the original test:





RF forecast visualization:

It can be observed from the figure that after approaching 1.8, the predicted results of the model are basically locked at 1.8 and no longer change.

This is actually fitting the noise of the training data, because the training data is also like this. (This is in line with the rules of machine learning training.)

Therefore, this also shows the significance of re-interpolating and cleaning the data after the unqualified data is below 1.8.

















2.1.4 Some other regression models:







2.2 Model training effect after data preprocessing:



Some other regression models









1 RF









2 lightGBM

