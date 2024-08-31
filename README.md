# Churn Analysis of an Internet & Telephone Provider

## Project Overview
This repository contains a comprehensive analysis of customer churn for an Internet and Telephone Provider. The project integrates Cohort Analysis and Predictive Modeling to develop a robust model for identifying customers at risk of churning. The aim is to provide actionable insights for implementing proactive customer retention strategies.

## Key Features
### 1. Exploratory Data Analysis (EDA)
- Explored the dataset to understand the distribution of key variables.
- Conducted detailed analysis on customer tenure, contract types, and payment methods.
- Visualized trends and patterns in customer behavior over time, focusing on factors influencing churn.

### 2. Cohort Analysis
- Conducted cohort analysis to segment customers based on their signup date and tenure.
- Analyzed customer retention and churn patterns within cohorts, identifying key periods where churn is most likely.

### 3. Data Preprocessing
- Applied ordinal encoding for categorical variables, one-hot encoding for nominal variables, and imputation techniques for handling missing data.
- Streamlined preprocessing using a pipeline and column transformer to ensure efficiency and reproducibility.
- Data preprocessing improved model performance, especially with complex models like XGBoost.

### 4. Machine Learning Models
- Employed a variety of machine learning models for predictive analysis:
  - Logistic Regression
  - k-Nearest Neighbors (kNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Gradient Boosting
  - XGBoost
- XGBoost outperformed all other models, achieving an F1-Score of 89%, making it the final choice for deployment.

### 5. Model Evaluation & Deployment
- Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.
- Employed cross-validation techniques to ensure robustness and prevent overfitting.
- Saved the best-performing model using pickle for easy deployment.
- Deployed the model via Streamlit to create an interactive web application for predicting customer churn.

