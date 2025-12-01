# 🏦 Bank Transaction Fraud Detection

## Project Overview
This project aims to **detect potential fraudulent transactions** using a **rule-based system** combined with **machine learning models**. Using a bank transaction dataset, the project performs:

- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Rule-Based Fraud Scoring  
- Machine Learning Modeling (Random Forest & Isolation Forest)  
- Evaluation and High-Quality Visualizations  

The project demonstrates skills in **data cleaning, feature engineering, anomaly detection, and predictive modeling**, making it suitable for a **data science portfolio**.

---

## Dataset
- **Source:** Kaggle Bank Transaction Dataset for Fraud Detection  
- **File:** `bank_transactions_data_2.csv`  
- **Features:**
  - `TransactionID` – Unique transaction ID  
  - `AccountID` – Customer account identifier  
  - `TransactionAmount` – Transaction amount  
  - `TransactionDate` – Date and time of transaction  
  - `TransactionType` – Debit or Credit  
  - `Location` – Transaction location  
  - `DeviceID` – Device used  
  - `IP Address` – IP of transaction origin  
  - `MerchantID` – Merchant ID  
  - `Channel` – ATM, Online, etc.  
  - `CustomerAge` – Age of customer  
  - `CustomerOccupation` – Occupation of customer  
  - `TransactionDuration` – Time taken to complete transaction  
  - `LoginAttempts` – Number of login attempts  
  - `AccountBalance` – Account balance at time of transaction  
  - `PreviousTransactionDate` – Date of previous transaction  

---

## Project Structure
The notebook follows these steps:

### 1. Setup & Libraries
Imports Python libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

### 2. Data Cleaning
### 3. Exploratory Data Analysis
Distribution of transaction types, channels, customer age, occupation

Monthly trends of transactions

Visualizations include countplots, histograms, and boxplots

## 💻 5. Feature Engineering

Creating meaningful features is crucial for improving the model's predictive power. The following features were engineered:

* **`TransactionCountPerAccount`**: The total number of transactions associated with a given `AccountID`. High frequency can be a fraud indicator.
* **`AmountOverBalance`**: The ratio of `TransactionAmount` to `AccountBalance`. A high ratio or amount exceeding the balance is a major red flag.
* **`DaysSinceLastTransaction`**: The time difference (in days) between the current transaction and the `PreviousTransactionDate`. Used to detect unusual transaction timing.

---

## 🎯 6. Fraud Scoring System

An initial **Rule-Based Scoring System** was implemented to create a pseudo-target variable (`FraudLabel`) for supervised learning. Each rule contributes a point to the total fraud score.

| # | Rule Description | Indicator |
| :--- | :--- | :--- |
| 1 | **High Transaction Amount** | Top 2% of all transaction amounts. |
| 2 | **Amount Deviation** | Transaction amount significantly deviates from the user's historical average. |
| 3 | **Multiple Login Attempts** | The `LoginAttempts` feature is $\geq 3$. |
| 4 | **Exceeds Balance** | `TransactionAmount` is greater than `AccountBalance`. |
| 5 | **Very Short Duration** | Very short `TransactionDuration` (potential bot or script). |
| 6 | **Night-time Transaction** | Transaction occurs between 2:00 AM and 5:00 AM. |
| 7 | **Rapid-fire Transactions** | Time since last transaction is $< 1.44$ minutes (approx. 86 seconds). |
| 8 | **High Frequency** | Transaction count per account is unusually high. |
| 9 | **Very Long Duration** | Very long `TransactionDuration` (indicates potential session takeover). |
| 10 | **Weekend Night** | Combination of night-time transaction and weekend day. |

**Fraud Labeling:** Transactions with a combined fraud score $\geq 3$ are assigned **`FraudLabel` = 1**.

---

## 🌲 7. Machine Learning: Random Forest

A **Random Forest Classifier** was trained to predict the **`FraudLabel`** created in the previous step.

* **Features:** Raw transaction details, customer information, and the engineered features were used.
* **Preprocessing:**
    * **Categorical Features** (e.g., `TransactionType`, `Channel`, `Location`) were one-hot encoded.
    * **Numeric Features** (e.g., `TransactionAmount`, engineered features) were scaled using `StandardScaler`.
* **Data Split:** The dataset was split into training (**80%**) and testing (**20%**) sets.
* **Model Training:** The Random Forest model was trained to predict the binary target (`FraudLabel`).

---

## 📊 8. Evaluation & Visualization

The model's performance was rigorously evaluated using standard classification metrics and visualizations.

* **Classification Report**: Displays **precision**, **recall**, and **f1-score** to assess model balance and performance.
* **Confusion Matrix**: Visualizes the counts of **True Positives, False Positives, True Negatives, and False Negatives**. 
* **ROC Curve**: Plots the True Positive Rate (Sensitivity) against the False Positive Rate (1-Specificity).
* **Feature Importance**: Identifies which features (both raw and engineered) contributed most to the model's predictions.

---

## ⚙️ 9. Isolation Forest for Anomaly Detection

To identify transactions that are statistically unusual outside of the predefined rules, an **unsupervised** **Isolation Forest** model was employed.

* **Objective:** Detects outliers (anomalies) in the transaction feature space.
* **Validation:** The Isolation Forest's predictions were compared against the rule-based **`FraudLabel`** to see if the unsupervised method confirms the rule-based approach.

---

## 📈 10. Summary Dashboard

The final results are presented through a dashboard containing high-quality visualizations to summarize findings and model performance.

* **FraudLabel Distribution**: Bar chart showing the count of fraudulent vs. normal transactions.
* **Transaction Amount vs. Fraud**: Box plot comparing transaction amount distributions across `FraudLabel` (1 vs. 0).
* **Feature Importance Plot**: Bar chart visualizing the top N features influencing the Random Forest model. 
* **Confusion Matrix Plot**: Heatmap visualization of the model's confusion matrix.
* **ROC Curve Plot**: Visual demonstration of the classifier's performance across various thresholds.
