[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e291f3d1ae8d4948b72376a2b216a8ff)](https://app.codacy.com/gh/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset/matplotlib)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset/numpy)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset/pandas)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset/seaborn)
![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset)
![GitHub repo size](https://img.shields.io/github/repo-size/Abhinav330/Classification-problem-using-ensambling-on-titanic-dataset)
# Code Summary

This Python script is a comprehensive example of a data analysis and machine learning project using the Titanic dataset. It covers various stages, including data loading, preprocessing, visualization, feature engineering, modeling, and generating predictions for a Kaggle competition.

## Data Loading and Preprocessing

The script starts by importing necessary libraries and loading the 'train.csv' and 'test.csv' datasets. It also combines both datasets for data preprocessing and exploration. Key steps in this section include:
- Handling missing values and visualizing them using a heatmap.
- Exploring and visualizing data distributions, especially 'Age' and 'Fare' columns.
- Creating new features like 'Ticket_Count' and 'Price' based on ticket counts and fares.
- Handling outliers in the 'Price' column.
- Extracting titles from names and grouping them into categories.
- Filling missing values in 'Age' and 'Embarked' columns.

## Data Visualization and Exploration

The script uses various plots to explore the data and relationships between variables, including:
- Bar plots to visualize the survival rate by different attributes.
- Kernel density estimate (KDE) plots for 'Age' and 'Price' by survival status.
- Heatmaps to visualize correlations between variables.
- FacetGrid for kernel density estimation of 'Age' and 'Price' by survival.

## Feature Engineering

Feature engineering involves:
- Creating new features like 'IsAlone' and 'IsCabin' based on family size and cabin information.
- Creating an 'Age_Range' feature by binning 'Age' values.
- Handling missing values in the 'Fare' column.
- Creating an interaction feature 'c*f' by multiplying 'Fare' and 'Pclass' columns.

## Model Building and Evaluation

The script builds a machine learning model using several classifiers and a voting ensemble method. Key steps include:
- Data encoding using the CatBoost encoder.
- Trying multiple classifiers and the voting ensemble method.
- Evaluating model performance with classification reports and confusion matrices.
- Using cross-validation to estimate model accuracy.

## Generating Predictions

The script generates predictions for the 'test.csv' dataset using the trained model. It also creates a submission file in the required format for the Kaggle competition.

## Referenes:
- kaggle.com. (n.d.). Titanic - Machine Learning from Disaster. [online] Available at: https://www.kaggle.com/competitions/titanic.
- Scikit-learn.org. (2012). 1.11. Ensemble methods — scikit-learn 0.22.1 documentation. [online] Available at: https://scikit-learn.org/stable/modules/ensemble.html.
- scikit-learn.org. (n.d.). sklearn.ensemble.AdaBoostClassifier — scikit-learn 0.22.1 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html.
- scikit-learn.org. (n.d.). sklearn.ensemble.BaggingClassifier — scikit-learn 0.23.1 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html.
- scikit-learn.org. (n.d.). 3.2.4.3.3. sklearn.ensemble.ExtraTreesClassifier — scikit-learn 0.22.2 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html.
- Scikit-learn.org. (2009). 3.2.4.3.5. sklearn.ensemble.GradientBoostingClassifier — scikit-learn 0.20.3 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html.
- Scikit-learn (2018). sklearn.ensemble.RandomForestClassifier — scikit-learn 0.20.3 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.
- Scikit-learn.org. (2019). sklearn.gaussian_process.GaussianProcessClassifier — scikit-learn 0.21.3 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html.
- scikit-learn. (2024). RidgeClassifierCV. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html [Accessed 18 Aug. 2024].
- scikit-learn. (n.d.). sklearn.naive_bayes.BernoulliNB. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html.
- scikit-learn (2019). sklearn.neighbors.KNeighborsClassifier — scikit-learn 0.22.1 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html.
- scikit-learn (2019). sklearn.svm.SVC — scikit-learn 0.22 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html.
- scikit learn (2018). 1.4. Support Vector Machines — scikit-learn 0.20.3 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/svm.html.
- Scikit-learn.org. (2019). sklearn.tree.DecisionTreeClassifier — scikit-learn 0.22.1 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html.
- scikit-learn (n.d.). sklearn.discriminant_analysis.LinearDiscriminantAnalysis — scikit-learn 0.24.1 documentation. [online] scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html. 
