# 🏦 Loan Approval Prediction using Logistic Regression

This project predicts loan approval probabilities using a **Logistic Regression model** inside a scikit-learn **Pipeline**, with extensive hyperparameter tuning via **GridSearchCV**. The model outputs **class probabilities** via `predict_proba()` to support threshold-based decision making.

---

## 📌 Problem Statement

Predict whether a loan will be approved or not using applicant data. This is a binary classification task:
- `1` → Loan Approved
- `0` → Loan Not Approved

---

## 🧰 Tech Stack

- **Python**
- `pandas`, `numpy` – data preprocessing
- `scikit-learn` – modeling & pipeline:
  - `LogisticRegression`, `GridSearchCV`, `StandardScaler`, `Pipeline`
- `LabelEncoder` used separately for categorical encoding before pipeline

---

## ⚙️ Workflow Summary

1. **Label Encoding**:
   - Used `LabelEncoder` on categorical columns **outside** the pipeline.

2. **Pipeline**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```
---

## 📂 Dataset

- Taken from [Kaggle: Loan Prediction Challenge](https://www.kaggle.com/competitions/loan-approval-predictions)
