![screenshot](images/studentperformanceimage.jpg?raw=True)

# Student Performance Analysis and Prediction
This repository contains a comprehensive analysis and prediction model for student performance based on a rich dataset (student-mat.csv). The project explores various aspects of students' academic, personal, and social life, aiming to understand the factors influencing their final grades and to predict academic outcomes.

# Overview
The project involves a detailed exploratory data analysis (EDA) and the application of machine learning techniques to predict students' final grades. We delve into aspects such as demographics, study habits, family background, and social behaviors, providing insights into how these factors correlate with academic performance.

# Features
* Data Preprocessing: Includes handling of missing values, encoding categorical variables, and feature scaling.
* Exploratory Data Analysis (EDA): Univariate, bivariate analysis, and visualization of data with custom features like age groups and total alcohol consumption.
* Feature Engineering: Creation of new features and transformation of existing ones to uncover hidden patterns.
# Machine Learning Models:
* Linear Regression: To understand the linear relationships between features and the target variable.
* Random Forest Regressor: To capture non-linear dependencies and feature importances.
* Model Evaluation: Using R² score and visualizations to evaluate and compare model performances.
* Visualization: Detailed visualizations including distribution plots, box plots, violin plots, and feature importance graphs.

# Tools and Libraries
* Python
* Pandas for data manipulation
* Matplotlib and Seaborn for data visualization
* Scikit-learn for machine learning and data preprocessing
 # Univariate Analysis / Distributions

 ![screenshot](images/histogram.png?raw=True)
![screenshot](images/distribg3.png?raw=True)

# Numerical Variables
##### Age: Most students are between 15 and 18 years old.
##### Medu (Mother's education): A wide range of educational backgrounds, with a notable number of mothers having a higher education (level 4).
##### Fedu (Father's education): Similar to mothers, fathers also show a diverse educational background.
##### Traveltime: Most students have a short travel time to school.
##### Studytime: Majority of students spend relatively few hours studying weekly.
# Categorical Variables
##### School: There are more students from the 'GP' school than the 'MS' school.
##### Sex: The dataset includes a fairly even distribution of male and female students.
##### Address: A larger proportion of students live in urban ('U') areas compared to rural ('R').
##### Famsize: More students come from families with more than 3 members ('GT3') than smaller families ('LE3').
##### Pstatus: Most students' parents are living together ('T').
![screenshot](images/categorical.png?raw=True)
![screenshot](images/3d.png?raw=True)
# Correlation Analysis Interpretation
#### The correlation matrix provides insights into how various numerical variables are related to each other. Key points:

#### Grades (G1, G2, G3): There are strong positive correlations between the grades in different periods (G1, G2, and G3). This suggests consistent performance across different grading periods.
#### Parental Education (Medu, Fedu): Mother's and Father's education levels (Medu and Fedu) are positively correlated with each other and with the student's final grade (G3). Higher parental education levels are associated with better student performance.
#### Failures: The number of class failures has a significant negative correlation with the grades, indicating that students who have failed more classes tend to have lower grades.
#### Study Time and Alcohol Consumption (Dalc, Walc): Study time has a slight positive correlation with grades but is negatively correlated with alcohol consumption. Students who study more tend to drink less.
#### Age: There's a slight negative correlation between age and final grade (G3), suggesting that older students in this group may have slightly lower grades.
#### Absences: Interestingly, absences have a very weak correlation with final grades, indicating that missing school does not have a strong direct impact on final grades in this dataset
![screenshot](images/heatmap.png?raw=True)
![screenshot](images/avggradebyabsences.png?raw=True)
# Feature Engineering Examples
```python
# Feature Engineering Examples

# 1. Total Alcohol Consumption
student_data['TotalAlc'] = student_data['Dalc'] + student_data['Walc']

# 2. Bin 'age' into age groups
student_data['AgeGroup'] = pd.cut(student_data['age'], bins=[0, 15, 17, 20], labels=['14-15', '16-17', '18+'])

# 3. Transform 'absences' into categorical bins
student_data['AbsenceCategory'] = pd.cut(student_data['absences'], bins=[-1, 5, 10, 30, float('inf')], labels=['Low', 'Medium', 'High', 'Very High'])

# 4. Interaction Term: Study Time and Failures
student_data['StudytimeXFailures'] = student_data['studytime'] * student_data['failures']

# 5. Logarithmic transformation of a skewed variable
# First, identify a skewed variable (using 'absences' as an example)
student_data['LogAbsences'] = np.log1p(student_data['absences'])

# Displaying the first few rows of the modified dataset
student_data[['TotalAlc', 'AgeGroup', 'AbsenceCategory', 'StudytimeXFailures', 'LogAbsences']].head()

```
#### Combining Features to Create New Insights:

#### Total Alcohol Consumption: Combine 'Dalc' (daily alcohol consumption) and 'Walc' (weekend alcohol consumption) to create a 'TotalAlc' feature, giving a more comprehensive view of a student's alcohol consumption.
#### Parental Education Level: Create a feature 'ParentEdu' by averaging 'Medu' and 'Fedu', to represent the overall educational level of the parents.
## Binning Numerical Variables:

#### age Groups: Convert 'age' into categorical bins (e.g., '14-16', '17-19'). This can help in identifying patterns across different age groups.
#### Absence Categories: Transform 'absences' into categories such as 'Low', 'Medium', 'High' based on defined thresholds, to simplify its impact analysis on grades.
#### Interaction Terms for Combined Effect:

##### Study Time and Failures: Create an interaction term between 'studytime' and 'failures' to explore if the combination of these two features has a different effect on grades than each feature individually.
## Polynomial Features:

#### Generate polynomial features (like squares or cubes) of significant variables (like 'studytime', 'age') to capture non-linear relationships.

# Model results:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd


student_data = pd.read_csv('data/student_data_encoded.csv')
# Dropping the original 'age' column
student_data.drop('age', axis=1, inplace=True)

# Define the target variable and features
target = 'G3'
features = [col for col in student_data.columns if col != target]

# Splitting the dataset
X = student_data[features]
y = student_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical and numerical columns
categorical_features = student_data.select_dtypes(include=['object']).columns.tolist()
# Adding 'AgeGroup' to categorical features
#categorical_features.append('AgeGroup')  

numerical_features = student_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove(target)  # Removing the target variable from numerical features

# Creating transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Applying the transformations
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

```

![screenshot](images/r2score.png?raw=True)
#### RandomForest outperfomed Linear Regression with slightly higher r2 scores.
```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Step 2: Training the Models

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_processed, y_train)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_processed, y_train)

# Step 3: Evaluating the Models

# Predicting and calculating R2 scores
y_pred_lr = lr.predict(X_test_processed)
y_pred_rf = rf.predict(X_test_processed)

r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

# Visualizing the scores
plt.bar(['Linear Regression', 'Random Forest'], [r2_lr, r2_rf])
plt.title('Model Performance Comparison')
plt.ylabel('R2 Score')
plt.show()

# Step 4: Visualizing Coefficients of Linear Regression

# Getting feature names after one-hot encoding
feature_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
feature_names += numerical_features

# Coefficients
coefficients = lr.coef_

# Visualizing the coefficients
plt.figure(figsize=(10, 15))
plt.barh(feature_names, coefficients)
plt.title('Coefficients of Linear Regression Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

```
#### Feature Importance
![screenshot](images/randomfc.png?raw=True)
#### Feature Importance
![screenshot](images/regressionc.png?raw=True)