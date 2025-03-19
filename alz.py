import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
from matplotlib import cm

# Load the dataset from a CSV file
df = pd.read_csv('alzheimers_dataset.csv')

# List of columns that are relevant to the box plot
columns_to_plot = [
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
    "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", 
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL"
]

################################################

# Define BMI categories
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

# Apply the function to create a new 'BMI_Category' column
df['BMI_Category'] = df['BMI'].apply(bmi_category)

# Create a colormap for the scatter plot based on BMI categories
cmap = plt.cm.viridis  # or you can use another colormap like 'coolwarm', 'plasma', etc.
categories = {'Underweight': 'blue', 'Normal': 'green', 'Overweight': 'orange', 'Obese': 'red'}

# Assign colors to each point based on BMI category
colors = df['BMI_Category'].map(categories)

# Create a scatter plot for Age vs BMI with color based on BMI category
plt.figure(figsize=(10, 6))

# Scatter plot with color coding by BMI category
scatter = plt.scatter(df['Age'], df['BMI'], c=colors, alpha=0.6, edgecolors='k', s=50)

# Add vertical lines for each BMI threshold
plt.axhline(y=18.5, color='blue', linestyle='--', label='BMI 18.5 (Underweight/Normal)')
plt.axhline(y=24.9, color='green', linestyle='--', label='BMI 24.9 (Normal/Overweight)')
plt.axhline(y=29.9, color='orange', linestyle='--', label='BMI 29.9 (Overweight/Obese)')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Scatter Plot: Age vs BMI with BMI Categories')

# Show legend for the lines
plt.legend(loc='upper right')

# Show the plot
plt.show()

################################################

# Function to calculate outliers using IQR
def calculate_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Set up the plot
plt.figure(figsize=(15, 10))

# Plot box plots for each relevant parameter
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 4, i)  # Set a 4x4 grid for box plots
    sns.boxplot(x=df[column])  # Create the box plot
    
    # Calculate outliers
    outliers = calculate_outliers(df, column)
    
    # Plot outliers as red points
    plt.scatter(outliers[column], [1] * len(outliers), color='red', label='Outliers', zorder=5)
    
    # Set the title for each box plot
    plt.title(f"{column}")

plt.tight_layout()
plt.show()

##############################################

# Set up the plot grid
plt.figure(figsize=(15, 10))

# Plot the histogram with KDE for each relevant parameter
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 4, i)  # Set a 4x4 grid for the plots
    
    # Plot histogram with KDE curve
    sns.histplot(df[column], kde=True, bins=20, color="skyblue", line_kws={"color": "red", "linewidth": 2})
    
    # Set title and labels for each subplot
    plt.title(f"Histogram with KDE for {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

##############################################

# Create age bins (60-65, 65-70, 70-75, etc.)
age_bins = [60, 65, 70, 75, 80, 85, 90, 95, 100]
age_labels = ['60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Set up the figure
plt.figure(figsize=(12, 8))

# Create a bar plot for the average BMI by Age Group and BMI Category
sns.barplot(data=df, x='Age_Group', y='BMI', hue='BMI_Category', errorbar=None, palette="Set2", dodge=True)

# Add titles and labels
plt.title('BMI Categories by Age Group')
plt.xlabel('Age Group')
plt.ylabel('BMI')
plt.legend(title='BMI Category')

# Show the plot
plt.tight_layout()
plt.show()

########################################

# Create age bins (60-65, 65-70, 70-75, etc.)
age_bins = [60, 65, 70, 75, 80, 85, 90, 95, 100]
age_labels = ['60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculate standard deviation for Age and BMI for each age group
age_bmi_std = df.groupby('Age_Group', observed=True)[['Age', 'BMI']].std()

# Reset index to have Age_Group as a column
age_bmi_std = age_bmi_std.reset_index()

# Melt the data for easier plotting
age_bmi_std_melted = pd.melt(age_bmi_std, id_vars='Age_Group', value_vars=['Age', 'BMI'], 
                             var_name='Parameter', value_name='Standard Deviation')

# Set the plot size
plt.figure(figsize=(12, 8))

# Plot bar graph for the standard deviation of Age and BMI
sns.barplot(data=age_bmi_std_melted, x='Age_Group', y='Standard Deviation', hue='Parameter', palette="Set2")

# Add labels and title
plt.title('Standard Deviation of Age and BMI by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Standard Deviation')

# Show the plot
plt.tight_layout()
plt.show()