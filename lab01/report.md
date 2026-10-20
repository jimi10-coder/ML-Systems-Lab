# Lab 1 — Machine Learning Baselines

## Goal

The goal of this lab was to train and compare two machine learning models using the Breast Cancer Wisconsin dataset. I compared Logistic Regression and Random Forest based on test accuracy, training time, inference time, model size, and process memory usage.

## Method

I used Python 3.11.8 and the Breast Cancer Wisconsin dataset from scikit-learn. I split the data into 70% training data and 30% testing data, using `random_state=42` and stratification.

I trained two models:

- Logistic Regression with `max_iter=1000` and `random_state=42`.
- Random Forest with 100 estimators and `random_state=42`.

I measured training time over five runs after a warm-up run. I measured single-sample inference time over 100 runs and used the median. I saved both trained models using joblib and measured their saved file sizes. I also recorded the current Python process memory usage.

## Results

| Measurement | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | 0.9415 | 0.9357 |
| Median training time (seconds) | 0.432092 | 0.198511 |
| Median inference time (ms) | 0.0561 | 2.8237 |
| Model size (KB) | 1.03 | 284.07 |
| Process memory usage (MB) | 103.83 | 105.09 |

The measurements are saved in `results/baseline_accuracy.csv`. The trained models are saved as joblib files in the results directory.

Note: Process memory usage is a current RSS reading, not peak memory during training.

## Conclusions

1. Logistic Regression achieved slightly higher test accuracy than Random Forest in this experiment.
2. Random Forest had a shorter median training time, but its saved model was larger.
3. Logistic Regression had lower median single-sample inference latency and a smaller saved model in this experiment.
