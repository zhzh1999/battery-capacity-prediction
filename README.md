
# battery-capacity-prediction
battery capacity prediction


0 project requirements

1. Technical goal: By analyzing the battery voltage data 3 hours before the battery core capacity is discharged, predict the cell voltage value in the next 7 hours, and predict the fault. The prediction accuracy exceeds 90%, and apply the model to EMT monitoring equipment. . 
2． Technical content: (1) Select the correct model according to Party A's needs; (2) Conduct model training; (3) Provide prediction analysis results; (4) Conduct model debugging when the prediction accuracy is insufficient; (5) Guide the completion of the model in EMT applications in.


1 Project analysis and data analysis
1.1 Project Analysis
Analysis: Time series prediction tasks;

1. Based on the battery voltage data in the first three hours, that is, from the four data moments of 0, 1, 2, and 3, predict the voltage value in the next 7 hours.
2. Determine whether the battery is usable based on the voltage value at the 10th hour, which is greater than or less than 1.8. The voltage of a normal battery is greater than 1.8v for 10 hours, and the voltage of an abnormal battery is less than 1.8v for 10 hours;


1.2 Analysis of battery capacity data sample

1.2.1 Data volume:
Training data: 500 data samples, used to train the AI ​​model;
(400 pieces of qualified data and 100 pieces of unqualified data)

Test data: 40 test samples, used to test the accuracy of the AI ​​model;
(20 pieces of qualified data, 20 pieces of unqualified data, and the above 500 pieces are not repeated)

1.2.2 Training data analysis
Visually display 500 pieces of training data:

As can be seen from the above analysis diagram,
Q1: There is an outlier, the voltage is over 200;
Looking up the table, we can see that the 268th data is abnormal: eliminate the abnormal data; and perform visual display;



Q2: Initial voltage abnormality: The voltage value at the first moment is greater than 2.2v: Screen out:


Visual display:
Q3: When the voltage is lower than 1.8v, sampling will no longer occur. This does not comply with objective laws, if you want to fit a curve. 
Therefore, for the data 10 hours ago, the voltage value is 1.8v, and the value of the next moment can be obtained by sampling according to the slope at that moment. Or reinterpolate according to the regression model;




400 pieces of data visualization:
Q4: Why does most data have an obvious turning point in the second hour?

It can be clearly seen from the figure that the discharge slope of a normal battery is relatively stable; the initial voltage of an abnormal battery is lower, but the discharge slope is larger; it conforms to the laws of real physics.


Data of the first 100 cases:
Q5: There is some data that the later moment is higher than the previous moment;
Needs to be filtered out; these data are equivalent to noise data and will affect model training. The higher the quality of the training data, the better the model's performance.


Normal data for the first 50 cases:



Abnormal battery voltage data:
Q6: The data changes drastically, and there is a problem of no change below 1.8v.
40 cases of test data:
Q: Is the slope of an unqualified battery below 1.8v the true value? 
Therefore, the accuracy of the numerical value of the 7h data after prediction is not that great. The focus is on the accuracy of the voltage value at the 10th hour.
And in line with the principle of not letting go if you kill someone by mistake. When it is very close to 1.8, it may also be judged as unqualified.
In fact, the test data here is lower than 1.8, and is still around 1.8. It is also filled in manually, which does not conform to the actual rules.





1.3 Data preprocessing:

1 Delete the data at the next moment that is higher than the previous moment:
Before and after deletion:
Check: There is the following data:
[12, 12, 105, 106, 154, 154, 275, 275, 276, 276, 277, 309, 334, 334, 336, 337, 338, 338, 357, 358, 359, 360, 397, 411, 412, 414, 415, 415, 436, 453]
30
[3, 8, 2, 2, 3, 7, 2, 5, 2, 5, 2, 1, 1, 6, 1, 1, 6, 10, 1, 1, 1, 1, 10, 1, 1 , 1, 1, 3, 1, 1]
30


2 For the voltage hovering around 1.8 10h ago, the training data regression interpolation:
For voltages hovering around 1.8 10 hours ago, regression interpolation is performed, and the minimum value is 0;

Method 1: Use the simplest method. When a value is near 1.8, start from the current value and subtract 0.05 each time;
However, as you can see from the picture below, there is still a problem with the slope of this area, which does not conform to the normal decline law;


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
F1 Score: 0.83333333333333334

40 test results: actual on the left, prediction on the right;



Precision is high, which means that when the model is near 0/1, it is more willing to predict 1. We want to be more willing to predict 0.
Therefore, we will refine the preprocessing of the training data later. and perform data augmentation. for abnormal data.

Conduct a keys study:


I think the problem is the data at point 2. If the slope changes for no reason at point 2, the final result will be 1.8, which is still greater than 1.8, so in a sense, the test data also needs to be used Interpolation is performed in the same way.



3 Test data for interpolation:








1.4 Data augmentation:






2 Model training and testing -Machine learning

Data collection: Collect data for training the model. This data can come from databases, files, networks, or other data sources.
Data preprocessing:
Clean data: deal with missing values, outliers and duplicate data.
Feature selection: Select features useful to the model.
Feature engineering: Create new features or transform existing features to improve model performance.
Data standardization/normalization: Scaling the data to the same scale to prevent certain features from having a disproportionate impact on the model due to a large range of values.
Divide the data set: Divide the data set into training set, validation set and test set. The training set is used to train the model, the validation set is used for model selection and hyperparameter tuning, and the test set is used for final evaluation of model performance.
Choose a model: Choose an appropriate machine learning algorithm based on the type of problem. Common algorithms include decision trees, support vector machines, neural networks, random forests, etc.
Train model: Use the training set data to train the selected model.
Model evaluation:
Use the validation set to evaluate the performance of different models and select the best model.
Tune the model's hyperparameters to optimize performance.
Model tuning: Find the optimal parameters of the model through cross-validation, grid search and other methods.
Model validation: Use the test set to evaluate the performance of the final model and ensure that the model is not overfitting or underfitting.



2.1 Model training effect before data preprocessing:

2.1.1 Divide the data set
Divide the current training data into training and validation sets according to 4:1:
The prediction is the test data;

2.1.2 Model performance evaluation

Evaluation Metrics: Time Series Forecasting Task:
1. The gap between the predicted value and the true value in the next 7 hours: the smaller the gap, the more accurate the prediction is; Regression problem:
2. Classification of the 10-hour predicted value. If it is less than 1.8, it is 0, if it is higher than 1.8, it is the true value; it is equivalent to a classification task; it can be represented by values ​​such as acc, tpr, and recall; Classification problem:


Prediction performance evaluation indicators after 4-10h:

Mean Absolute Error (MAE): The average of the absolute value of the difference between the predicted value and the actual value. It measures the average size of forecast errors but does not take into account the direction of the errors.
Mean Squared Error (MSE): The average of the squared differences between predicted and actual values. MSE imposes a larger penalty on large prediction errors.






Is the 10h prediction result 0/1 correct? Classification indicators:



2.1.3 Random Forest
Random Forest is an ensemble learning method that performs classification or regression prediction by building multiple decision trees. The core idea of ​​random forest is "brainstorming", that is, by combining the prediction results of multiple models to improve the overall prediction accuracy and robustness.


1 Training and verification:

Randomly divide according to 4:1 to train the random forest:
Use the first 0-3 data as x, and use 4-10 as predicted y;


predict:
Training: MSE: 0.00033746353177420104
Verification: MSE: 0.00016454365824939

Classification: Use 1.8 as the threshold;

validation set;
Accuracy: 0.9591836734693877
Precision: 0.9473684210526315
Recall: 1.0
F1 Score: 0.972972972972973

Saved model:
battery\_life\_model\_feature\_4\_test1.pkl

2 Test set test:

40 cases of test data:
MSE: 0.0005271824049329601
Accuracy: 0.775
Precision: 0.7619047619047619
Recall: 0.8
F1 Score: 0.7804878048780488


Visual display of test data:
40 data visualizations of the original test:


RF prediction visualization:
It can be observed from the figure that after the model prediction results approach 1.8, the prediction results are basically locked at 1.8 and no longer change.
This is actually the noise that fits the training data, because the training data is also like this. (This is in line with the rules of machine learning training.)
Therefore, this also shows the significance of reinterpolating and cleaning the data after the unqualified data is lower than 1.8.








2.1.4 Some other regression models:



2.2 Model training effect after data preprocessing:

Some other regression models




1 RF




2 lightGBM:
