# Problem Definition: Adult Income Prediction

## 1. Problem Statement

The goal of this project is to develop a machine learning system that predicts whether a person's annual income is above or below $50,000. The system uses information about a person's age, education, occupation, working hours, and other demographic and employment-related characteristics.

This is a binary classification problem because the model predicts one of two possible classes: `<=50K` or `>50K`.

## 2. Dataset and Inputs

The project uses the UCI Adult Income dataset, downloaded from OpenML (version 2). The dataset contains 48,842 records and 15 columns.

The input data consists of 14 features, including:
- Age
- Workclass
- Education
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Capital gain and capital loss
- Hours worked per week
- Native country

The target variable is `class`, which represents the income category.

## 3. Project Objectives

The main objectives are to:
- Load and validate the dataset.
- Identify missing values and possible data-quality problems.
- Prepare the data using a reproducible preprocessing pipeline.
- Prevent data leakage by fitting preprocessing steps only on the training data.
- Create additional features to investigate whether they improve model performance.
- Train and compare machine learning models.

## 4. Evaluation Metrics

The project uses two evaluation metrics:

- **Accuracy:** Measures the proportion of correct predictions.
- **F1-score:** Measures the balance between precision and recall.

F1-score is particularly important because the two income classes are not equally represented in the dataset.

## 5. Constraints and Reproducibility

The project must avoid data leakage. The dataset is split into training and testing sets before preprocessing. Imputation values, outlier limits, and other preprocessing parameters are learned from the training data only.

The pipeline uses a fixed random state to make the experiments reproducible. The final processed datasets, fitted preprocessor, validation reports, model comparison results, and SHA256 hashes are saved in the project results directory.

## 6. Expected Outcome

The expected outcome is a reproducible machine learning pipeline that predicts income categories and evaluates whether feature engineering improves classification performance. The project is intended for educational and research purposes, not for making decisions about individuals.