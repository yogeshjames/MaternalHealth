# MaternalHealth
 Predicting Pregnancy Risk Levels Using Machine Learning


# Predicting Pregnancy Risk Levels Using Machine Learning

## Problem Statement

Pregnancy complications are a significant health concern globally, affecting both maternal and fetal outcomes. Identifying high-risk pregnancies early on can allow for better medical interventions and improved health outcomes. In this study, we explore the use of machine learning models, particularly Random Forest, to predict the risk level of pregnancy complications. The goal is to classify pregnancy data into two categories—**Low Risk** and **High Risk**—based on various health metrics such as age, blood pressure, blood sugar levels, BMI, and other relevant factors.

## Dataset

The dataset used for this project was obtained from [Mendeley Data](https://data.mendeley.com/datasets/p5w98dvbbk/1). The data consists of several features related to the maternal health of pregnant women, such as age, blood pressure, body temperature, BMI, and whether the individual has preexisting conditions like diabetes or mental health issues. The target variable in this dataset is the **Risk Level**, which indicates the overall risk of complications during pregnancy. This dataset serves as the foundation for developing a predictive model to classify the risk levels.

### Features of the Dataset:

- **Age**: The age of the pregnant woman, which can be a significant factor in predicting pregnancy complications.
- **Systolic BP**: Systolic blood pressure, representing the pressure when the heart beats. High levels may indicate hypertension or other complications.
- **Diastolic BP**: Diastolic blood pressure, measuring the pressure between heartbeats.
- **Blood Sugar (BS)**: The blood sugar levels of the patient, which are critical for diagnosing gestational diabetes.
- **Body Temperature**: The patient’s body temperature, which can be indicative of infections.
- **BMI**: Body Mass Index, a measure of body fat based on weight and height.
- **Previous Complications**: Whether the patient had pregnancy complications in the past.
- **Preexisting Diabetes**: Whether the patient had diabetes before pregnancy.
- **Gestational Diabetes**: Whether the patient developed diabetes during the current pregnancy.
- **Mental Health**: Indicates if the patient has mental health concerns that might affect pregnancy.
- **Heart Rate**: The heart rate of the patient, which may indicate stress or strain.
- **Risk Level**: The target variable representing the classification of pregnancy risk (either **Low Risk** or **High Risk**).

## Data Preprocessing

To ensure the quality and reliability of the model, preprocessing was a key step in the data pipeline. The dataset was cleaned to handle any missing or defective values.

### Cleaning the Data:
We first loaded the dataset and noticed that there were 1025 rows in total. After inspecting the data, we found that only 18 rows contained missing or defective values. To address this, we used the following approach:

```python
# Drop rows with missing values
data_cleaned = data.dropna()


```

Labeling and Classification of Data
The Risk Level column in the dataset was used as the target variable, with two possible labels: Low Risk (0) and High Risk (1). This was the primary outcome variable that we aimed to predict using machine learning algorithms.

We performed label encoding on the Risk Level feature to convert the categorical labels into numerical values (0 and 1), making it compatible with machine learning models.

```python
# Encode target labels
label_encoder = LabelEncoder()
data_cleaned['Risk Level'] = label_encoder.fit_transform(data_cleaned['Risk Level
```

### Data Distribution
To gain insights into the distribution of each feature and understand how the data is structured, we visualized the distributions of the key features. Visualizations help identify any imbalances, patterns, or outliers in the data.
![heart_rate_distribution](https://github.com/user-attachments/assets/cbd31027-c887-430f-9f7a-e0fa9c3a861c)

![age_distribution](https://github.com/user-attachments/assets/b54b05c6-4c4e-4e3b-a991-c6f74c77e377)

## Model Selection

After preprocessing the data, we tested several machine learning models to find the one that best predicts pregnancy risk levels. The models evaluated include:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### Code for Training and Evaluating Models

```python
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and evaluate models
accuracy_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_results[name] = accuracy_score(y_test, y_pred)

# Display results
for model_name, accuracy in accuracy_results.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")
```
![Screenshot 2024-12-26 150109](https://github.com/user-attachments/assets/4ad18656-f349-45f5-b72e-dd4eecf97697)

As u can see random forest gave us the best accuracy soo we choose to go with random forest but....

## Beyond Accuracy: Evaluating Precision, Recall, F1-Score, and AUC

### Why Accuracy Alone is Not Enough

Even though we achieved high accuracy with the Random Forest model, accuracy alone does not provide a complete picture of the model's performance. This is especially critical in imbalanced datasets like ours, where the number of "Low Risk" cases significantly outweighs the "High Risk" cases. A model that predicts only the majority class could still achieve high accuracy while completely failing to identify high-risk pregnancies.  

### Why Additional Metrics Matter

1. **Precision**: Helps us measure how many of the predicted "High Risk" cases were actually correct. This ensures we minimize false positives, which could cause unnecessary stress or interventions.  
2. **Recall**: Measures how many of the actual "High Risk" cases were correctly identified by the model. High recall is crucial in healthcare to avoid missing any critical cases (false negatives).  
3. **F1-Score**: Balances precision and recall, providing a single score that accounts for both. This is particularly useful when there’s a trade-off between precision and recall.  
4. **AUC (Area Under the Curve)**: Evaluates the model’s ability to distinguish between "Low Risk" and "High Risk" cases at various thresholds. A higher AUC indicates better classification performance.  
5. **Confusion Matrix**: Gives a detailed breakdown of true positives, true negatives, false positives, and false negatives, offering insights into where the model might be making errors.

### Generating Evaluation Metrics

To evaluate our model, we generated the confusion matrix, precision, recall, F1-score, and AUC using the following code:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

# Generate predictions
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# AUC Score
auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC Score: {auc_score:.4f}")

# Plot AUC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("AUC Curve")
plt.show()
```
![Screenshot 2024-12-26 151019](https://github.com/user-attachments/assets/2bd974c5-1f2f-42b4-8694-e4b74a4c2527)
![myplot1](https://github.com/user-attachments/assets/b828093b-bb71-4c05-9793-dfafdbc36e48)
![myplot](https://github.com/user-attachments/assets/67800397-aeaf-40b8-922e-feadd2d31b61)

## Observation

The following metrics were used to evaluate the performance of the classification model for predicting maternal health complications:

### Precision, Recall, and F1-Score

- **No Complication:**
  - **Precision:** 0.96
  - **Recall:** 0.99
  - **F1-Score:** 0.97
- **Complication:**
  - **Precision:** 0.99
  - **Recall:** 0.97
  - **F1-Score:** 0.98

### Accuracy

- **Accuracy:** 0.98 (98% of the predictions are correct)

### AUC (Area Under the Curve)

- **AUC:** 0.9991 (Exceptional discriminatory power)

## Model Interpretation

- **Precision**: For both classes, the model is highly accurate when it predicts "No Complication" (0.96) and "Complication" (0.99). This means the model has very few false positives.
  
- **Recall**: The model is good at capturing actual cases, with high recall scores for both "No Complication" (0.99) and "Complication" (0.97). This ensures minimal false negatives.

- **F1-Score**: The F1-Score is a balanced measure of precision and recall. The model achieves 0.97 for "No Complication" and 0.98 for "Complication", demonstrating that it is both accurate and reliable.

- **Accuracy**: With an accuracy of 98%, it shows strong overall performance in predicting the correct class for the majority of the instances.

- **AUC**: An AUC of 0.9991 signifies that the model almost perfectly distinguishes between complications and non-complications, providing high confidence in its predictions.

## Conclusion

Soo we could say that the model performs well  in predicting maternal health complications with high precision, recall, and accuracy. The AUC score further supports its excellent ability to distinguish between the two classes,

### Overfitting Considerations

Even though the model shows great performance in terms of accuracy, precision, recall, F1-score, and AUC, overfitting could still be an issue. Overfitting happens when the model doesn't just learn the real patterns in the data but also learns the noise or random fluctuations. This results in fantastic performance on the training data but poor performance on new, unseen data.

In this case, overfitting could have been a risk for a few reasons:

- **High Model Complexity**: Random Forests, with their many trees and deep splits, are powerful but can sometimes get too caught up in the noise, especially if the data is complex or small.
- **Lack of Regularization**: While Random Forests usually handle overfitting well by averaging over multiple decision trees, very deep trees or too many features can still cause the model to overfit.

### How We Tackled Overfitting

To ensure our model generalizes well, we took several steps to reduce the risk of overfitting:

- **Cross-Validation**: We used cross-validation, which means we tested the model on different subsets of the data, not just one. This helps us make sure the model isn't just memorizing the data, but rather, learning the real patterns that will work on new, unseen data.
  
- **Hyperparameter Tuning**: We also tuned the hyperparameters, like the number of trees and the depth of each tree, to find the right balance between underfitting (not learning enough) and overfitting (learning too much noise).

Next, we'll take a check  at the cross-validation results to make sure our model is really generalizing well and not overfitting.

![Screenshot 2024-12-26 155816](https://github.com/user-attachments/assets/f58fd2ed-d0b7-4b72-8291-45cb73051a7d)

The **average cross-validation score** is **0.9563**. Now, if you compare that to the model's **accuracy** of **0.98**, you can see there’s a bit of a difference—about 0.04. This suggests the model might be overfitting, as it's performing slightly better on the training data compared to new, unseen data in cross-validation.

### Why Overfitting Happens and How We Can Fix It

So, why do we care about this? Well, overfitting happens when the model learns the details and noise of the training data too well, and then struggles to generalize when it sees new data. It's like memorizing answers for a test but not really understanding the material. We don't want that.

To solve this, we can try **hyperparameter tuning**.

### What is Hyperparameter Tuning and Why Do We Do It?

**Hyperparameter tuning** is the process of adjusting the settings (hyperparameters) of our model to get the best possible performance without overfitting. In Random Forests, some of the key hyperparameters we can tweak include:

- **Number of Trees**: More trees might reduce overfitting, but too many can also cause the model to become too complex.
- **Maximum Depth of Trees**: Limiting how deep the trees can go can prevent the model from fitting too closely to the training data.
- **Minimum Samples per Leaf**: Setting a minimum number of samples for a leaf node can ensure the model doesn’t create branches based on just a few examples.

By tuning these hyperparameters, we can find the sweet spot where the model is complex enough to capture the underlying patterns, but simple enough to generalize well to new data.

Next, we'll apply hyperparameter tuning and see how it impacts the model’s performance!

### Grid Search for Hyperparameter Tuning

To address the overfitting issue, we applied **Grid Search** to fine-tune our model’s hyperparameters. **Grid Search** is a technique that systematically tests different combinations of hyperparameters (like the number of trees or the maximum depth) and evaluates the model's performance for each combination. This helps us find the best possible set of hyperparameters for the model, improving its generalization ability.

### Improved Performance After Tuning

After running the Grid Search with different hyperparameter configurations, we found the best set of parameters that resulted in a significant improvement. The **average cross-validation score** increased to **98.246**. This improvement indicates that the model is now better generalizing to unseen data, solving the overfitting problem we saw earlier. 

In simpler terms, the model is no longer just memorizing the training data, but is now able to make accurate predictions on new data as well.

### Conclusion

By applying Grid Search and fine-tuning the hyperparameters, we successfully reduced overfitting, leading to a more robust model that performs well both on the training data and on unseen data. The increase in the average cross-validation score shows that the model is now better at generalizing, making it more reliable for real-world applications.

## Deploy
Now just integrate the model with flask and deploy it....

