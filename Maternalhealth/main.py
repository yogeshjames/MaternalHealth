# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder


# Load the data
data_path = './data.csv'
data = pd.read_csv(data_path)

# Drop rows with missing values
data = data.dropna()

# Encode target labels
label_encoder = LabelEncoder()
data['Risk Level'] = label_encoder.fit_transform(data['Risk Level'])

# Features and target
X = data.drop('Risk Level', axis=1)  # features
y = data['Risk Level']  # target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: RandomizedSearchCV for Random Forest Hyperparameter Tuning
best_rf_model = RandomForestClassifier(
    bootstrap=True,
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=3,
    min_samples_split=4,
    n_estimators=186
)

# Train the model on the training data
best_rf_model.fit(X_train, y_train)

# Step 4: Model Evaluation

# Make predictions using the best model
y_pred_rf = best_rf_model.predict(X_test)



# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix using a heatmap for better visualization
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Complication', 'Complication'], yticklabels=['No Complication', 'Complication'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print the classification report (Precision, Recall, F1-Score)
print(classification_report(y_test, y_pred_rf, target_names=['No Complication', 'Complication']))

# Calculate ROC AUC score
y_prob = best_rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC score
print(f"AUC: {roc_auc:.4f}")

# Step 5: Cross-Validation Scores
# Perform cross-validation to evaluate the model's generalization
scores = cross_val_score(best_rf_model, X, y, cv=10)
print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean():.4f}")
