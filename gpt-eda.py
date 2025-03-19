import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size for all plots
plt.figure(figsize=(10, 6))

# Load the dataset into pandas DataFrames for each location
cleveland_df = pd.read_csv('heart+disease/processed.cleveland.data', header=None)
hungarian_df = pd.read_csv('heart+disease/processed.hungarian.data', header=None)
switzerland_df = pd.read_csv('heart+disease/processed.switzerland.data', header=None)
longbeach_df = pd.read_csv('heart+disease/processed.va.data', header=None)

# Set column names for each dataset based on the attributes mentioned
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']
cleveland_df.columns = columns
hungarian_df.columns = columns
switzerland_df.columns = columns
longbeach_df.columns = columns

# Data Preprocessing: Check for missing values
def check_missing_data(df):
    return df.isnull().sum()

# Check missing data in each dataset
cleveland_missing = check_missing_data(cleveland_df)
hungarian_missing = check_missing_data(hungarian_df)
switzerland_missing = check_missing_data(switzerland_df)
longbeach_missing = check_missing_data(longbeach_df)

# Display missing data analysis for each dataset
print("Cleveland Missing Data:\n", cleveland_missing)
print("\nHungarian Missing Data:\n", hungarian_missing)
print("\nSwitzerland Missing Data:\n", switzerland_missing)
print("\nLong Beach Missing Data:\n", longbeach_missing)

# Fill missing values if any, for simplicity, use forward fill
cleveland_df.fillna(method='ffill', inplace=True)
hungarian_df.fillna(method='ffill', inplace=True)
switzerland_df.fillna(method='ffill', inplace=True)
longbeach_df.fillna(method='ffill', inplace=True)

# 1. Plotting Age vs Cholesterol (for Cleveland dataset as an example)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='chol', data=cleveland_df, hue='class', palette="coolwarm")
plt.title('Age vs Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

# 2. Plotting Distribution of Cholesterol Levels
plt.figure(figsize=(10, 6))
sns.histplot(cleveland_df['chol'], kde=True, color='skyblue')
plt.title('Cholesterol Distribution')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

# 3. Plotting Correlation Heatmap
correlation_matrix = cleveland_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Cleveland Dataset')
plt.show()

# 4. Plotting Age vs Max Heart Rate (thalach) and relationship with Disease
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='thalach', data=cleveland_df, palette="Set3")
plt.title('Max Heart Rate (thalach) vs Disease Class')
plt.xlabel('Disease Class (0=Healthy, 1-4=Sick)')
plt.ylabel('Max Heart Rate')
plt.show()

# 5. Exploring Cost Analysis (from heart-disease.cost file)
cost_data = {
    'test': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    'cost': [1.00, 1.00, 1.00, 1.00, 7.27, 5.20, 15.50, 102.90, 87.30, 87.30, 87.30, 100.90, 102.90]
}
cost_df = pd.DataFrame(cost_data)
plt.figure(figsize=(12, 6))
sns.barplot(x='test', y='cost', data=cost_df)
plt.title('Cost of Tests')
plt.xlabel('Test')
plt.ylabel('Cost (in CAD)')
plt.xticks(rotation=90)
plt.show()

# 6. Plotting Delay Information (from heart-disease.delay file)
delay_data = {
    'test': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    'delay': ['immediate', 'immediate', 'immediate', 'immediate', 'delayed', 'delayed', 'delayed', 'delayed', 'delayed', 'delayed', 'delayed', 'delayed', 'delayed']
}
delay_df = pd.DataFrame(delay_data)
sns.countplot(x='delay', data=delay_df, palette="Set2")
plt.title('Test Delay Information')
plt.xlabel('Delay Type')
plt.ylabel('Frequency')
plt.show()

# 7. Expense Analysis (from heart-disease.expense file)
expense_data = {
    'test': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    'full_cost': [1.00, 1.00, 1.00, 1.00, 7.27, 5.20, 15.50, 102.90, 87.30, 87.30, 87.30, 100.90, 102.90],
    'discount_cost': [1.00, 1.00, 1.00, 1.00, 5.17, 3.10, 15.50, 1.00, 1.00, 1.00, 1.00, 100.90, 1.00]
}
expense_df = pd.DataFrame(expense_data)
plt.figure(figsize=(12, 6))
sns.barplot(x='test', y='full_cost', data=expense_df, color='skyblue', label='Full Cost')
sns.barplot(x='test', y='discount_cost', data=expense_df, color='orange', label='Discount Cost')
plt.title('Full and Discount Cost of Tests')
plt.xlabel('Test')
plt.ylabel('Cost (in CAD)')
plt.xticks(rotation=90)
plt.legend()
plt.show()
