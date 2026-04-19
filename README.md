# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts customer churn using **Logistic Regression**, **Random Forest**, and **XGBoost** in Python. The project handles class imbalance using **SMOTE**, compares multiple models, and selects the best performing model based on evaluation metrics.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. By predicting churn in advance, companies can improve retention strategies and reduce customer loss.

This project uses a synthetic telecom customer dataset with 1000 records and applies multiple supervised learning algorithms for churn classification.

---

## Features

- End-to-end machine learning pipeline
- Synthetic telecom churn dataset generation
- Data preprocessing and feature engineering
- Class imbalance handling using **SMOTE**
- Trained multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Performance comparison using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Saved best model for future predictions

---

## Project Structure

```bash
customer_churn_prediction_project/
│── data/
│   └── customer_churn_sample.csv
│── models/
│   └── best_churn_model.pkl
│── reports/
│   └── metrics.json
│── src/
│   ├── data_generator.py
│   ├── train.py
│   └── predict.py
│── requirements.txt
└── README.md
