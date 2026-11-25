# Data Card – UCI Adult Income Dataset

## 1. Dataset Overview
- **Name:** UCI Adult Income Dataset
- **Source:** OpenML
- **Version:** 2
- **Number of records:** 48,842
- **Number of columns:** 15
- **Target variable:** `class`
- **Task:** Predict whether a person's annual income exceeds $50,000.

## 2. Dataset Source
The dataset is obtained using the Scikit-learn `fetch_openml()` function.

Source: https://www.openml.org/

## 3. License
The dataset is distributed through OpenML. The exact license and usage conditions should be verified on the dataset's OpenML page before redistribution.

## 4. Data Processing
The project applies:
- Stratified 70/30 train-test split.
- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Outlier clipping using training-set 1st and 99th percentiles.
- One-hot encoding for categorical features.
- Standard scaling for numerical features.
- Removal of exact duplicate training records.

The test set is not used to fit preprocessing transformations.

## 5. Known Biases and Limitations
The dataset contains demographic and employment information, including age, sex, and race.

It reflects historical US census data and may not represent current populations or other countries.

The dataset may contain social and demographic biases. Predictions should not be interpreted as objective judgments about individuals.

## 6. Privacy Concerns
The dataset contains demographic and socioeconomic information.

Although it is a public dataset, predictions based on sensitive demographic attributes can create fairness and discrimination concerns.

The project is intended for educational and research purposes, not real-world decisions about individuals.

## 7. Intended Use
This dataset is used to study machine learning preprocessing, data validation, feature engineering, and income classification.

## 8. Dataset Integrity
SHA256 hashes of the processed training features, testing features, and target labels are stored in:

`results/sha256_hashes.txt`
