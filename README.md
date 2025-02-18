# Credit-Card-Approval-Prediction

This project predicts credit card approval decisions based on applicant and credit history data. Using advanced machine learning techniques, the goal is to assist financial institutions in assessing credit risks effectively.

---

## **Project Overview**
This project implements **Credit Risk Modeling** to predict **the Probability of Default (PD)** for loan applicants. Particularly, the project focuses on building a machine learning model to predict credit card approval and manage credit risk effectively. Using exploratory data analysis (EDA), feature engineering, feature selection, and machine learning techniques, we aim to derive insights and provide actionable predictions.

The models used include:
- **XGBoost**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Logistic Regression**

We focus on optimizing **recall**, as missing defaulters is riskier than misclassifying non-defaulters. The final analysis compares **PD across multiple models** and determines the best approach for **risk management and Expected Loss (EL) estimation**.

---

## **Data Description**
The dataset consists of **financial and demographic attributes** of loan applicants, including:
- **Income, Employment Stability, Debt-to-Income Ratio**
- **Past Loan Status, Payment History, Credit Utilization**
- **Demographic Details (Age, Gender, Family Size)**
- **Loan Details (Amount, Term, Purpose, Payment History)**

**Target Variable:**  
- **`Credit_Status`** (Binary: **1 = Default, 0 = No Default**)  

**Engineered Features:**
- **Debt-to-Income Ratio**
- **Missed Payment Frequency**
- **Maximum Consecutive Missed Payments**

---

## **Machine Learning Models**
To accurately predict **credit default**, we trained and compared the following models:

| **Model** | **Strengths** | **Weaknesses** |
|-----------|--------------|----------------|
| **XGBoost** | High accuracy, robust to missing values, works well with tabular data | Can be computationally expensive |
| **Random Forest** | Best recall (captures most defaulters), interpretable | Slightly slower than XGBoost |
| **SVM** | Works well for smaller datasets | Performs poorly for imbalanced datasets |
| **Logistic Regression** | Simple, interpretable benchmark model | Struggles with non-linearity |

**Hyperparameter Tuning:**  
We used **Bayesian Optimization** to fine-tune each model for the best performance.

---

## **Model Evaluation Metrics**
To assess performance, we evaluated each model using:

| **Metric** | **Definition** | **Importance in Credit Risk** |
|------------|--------------|------------------------------|
| **Accuracy** | Overall correct predictions | Misleading in imbalanced datasets |
| **Precision** | $\frac{TP}{TP+FP}$ (How many predicted defaulters are actual defaulters) | Important for minimizing false alarms |
| **Recall** | $\frac{TP}{TP+FN}$ (How many actual defaulters were detected) | Critical for identifying risky applicants |
| **F1 Score** | Harmonic mean of Precision & Recall | Best balance between false positives & false negatives |
| **ROC AUC** | Measures ability to distinguish defaults & non-defaults | Helps in selecting threshold |

**Final Model Selection:**  
- **Random Forest had the highest Recall** (best for default detection).
- **XGBoost provided stable PD estimations.**
- **SVM performed the worst (high false negatives).**

---

## **Probability of Default (PD) Analysis**
### **PD Definition**
The **Probability of Default (PD)** represents the likelihood that an applicant **defaults on their loan** within a given timeframe.

$PD = P(Y = 1 | X)$

where:
- $Y = 1$ (Default), $Y = 0$ (No Default)
- $X$ represents financial & demographic features

### **Risk Categorization**
To assess loan risk, we categorize PD into risk buckets:

| **Risk Category** | **PD Range** |
|------------------|--------------|
| **Low Risk** | $PD < 10\%$ |
| **Medium Risk** | 10\% $\leq PD$ < 40\% |
| **High Risk** | 40\% $\leq PD$ < 75\% |
| **Very High Risk** | PD $\geq$ 75\% |

### **PD Comparison Across Models**
- **Random Forest provided the widest range of PD values** (better separation of risky borrowers).
- **XGBoost had stable PD predictions** but slightly lower recall.
- **SVM performed poorly, assigning very low PD values to most applicants**.

**Mathematical Components in Credit Risk Modeling**:
1. **Loss Given Default (LGD):** $LGD = 1 - \text{Recovery Rate}$
---

## **Installation & Usage**
### **Requirements**
- Python 3.8+
- Libraries: `pandas`, `numpy`

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`
- **Tools**: Jupyter Notebook, Git, GitHub
  
## **Next Steps & Future Improvements**
1. Threshold Optimization
Adjust classification thresholds to balance Recall & Precision.

2. Incorporate Real LGD & EAD Data
Replace assumptions with actual financial data to improve accuracy.

3. Feature Engineering Enhancements
Include credit utilization, spending habits, and macroeconomic indicators to refine the predictive power of the model.

---

## **Setup Instructions**

## **Dataset Information**

The dataset  like `application_record.csv` is too large to be stored on GitHub.
You can download it from: [https://-google-drive-link](https://drive.google.com/drive/folders/15alTNIE2FTSc3YQUU8Nki3hOd8Q6eNEY?usp=drive_link)

### **Install Dependencies**
```bash
pip install -r requirements.txt


### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/credit-risk-prediction.git


