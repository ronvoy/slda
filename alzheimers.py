# Complete Python Code for Alzheimer’s Disease Risk Factor Analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, recall_score

# 1. Data Import
data = pd.read_csv("alzheimers_dataset.csv")
print("Data Imported:")
print(data.head())

# 2. Data Encoding
# Encode ordinal variable: EducationLevel
le = LabelEncoder()
data['EducationLevel'] = le.fit_transform(data['EducationLevel'])

# One-hot encode nominal categorical variables: Gender and Ethnicity
data = pd.get_dummies(data, columns=['Gender', 'Ethnicity'], drop_first=True)
print("After Encoding:")
print(data.head())

# 3. Data Cleaning
# 3.1 Handling Missing Values
print("Missing Values Before Imputation:")
print(data.isnull().sum())
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col].fillna(data[col].median(), inplace=True)

# 3.2 Outlier Detection and Removal (IQR method)
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Example on BMI and SystolicBP
data = remove_outliers(data, 'BMI')
data = remove_outliers(data, 'SystolicBP')

# 3.3 Removing Duplicates
data.drop_duplicates(inplace=True)
print("Data Shape after Cleaning:", data.shape)

# 4. Data Transformation
# 4.1 Feature Scaling (Standardization)
features_to_scale = [col for col in data.columns if col not in ['PatientID', 'Diagnosis']]
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# 4.2 Feature Engineering: Create a binary feature for high blood pressure
data['HighBP'] = (data['SystolicBP'] > 140).astype(int)

# 5. Data Visualization
# Histograms
data.hist(bins=20, figsize=(14, 10))
plt.tight_layout()
plt.show()

# Box plot for BMI
sns.boxplot(x=data['BMI'])
plt.title("Box Plot of BMI")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 6. Feature Selection
# Separate features and target variable
X = data.drop(['PatientID', 'Diagnosis'], axis=1)
y = data['Diagnosis']

# Feature importance using ExtraTreesClassifier
model_ft = ExtraTreesClassifier(n_estimators=50)
model_ft.fit(X, y)
feat_importances = pd.Series(model_ft.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(10, 8))
plt.title("Feature Importances")
plt.show()

# 7. Model Training
# 7.1 Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 7.2 Initialize Models
log_reg = LogisticRegression(max_iter=1000)
ada_clf = AdaBoostClassifier(n_estimators=100)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train models
log_reg.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test)
y_pred_ada = ada_clf.predict(X_test)
y_pred_xgb = xgb_clf.predict(X_test)

# 8. Model Evaluation
print("=== Model Evaluation ===")
print("Logistic Regression F1 Score:", f1_score(y_test, y_pred_log))
print("AdaBoost F1 Score:", f1_score(y_test, y_pred_ada))
print("XGBoost F1 Score:", f1_score(y_test, y_pred_xgb))

print("Logistic Regression Recall:", recall_score(y_test, y_pred_log))
print("AdaBoost Recall:", recall_score(y_test, y_pred_ada))
print("XGBoost Recall:", recall_score(y_test, y_pred_xgb))

# Cross Validation with XGBoost
cv_scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='f1')
print("XGBoost 5-Fold CV F1 Score:", cv_scores.mean())

# End of Script
