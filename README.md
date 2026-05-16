# Loan-Default-Risk-Classification-ML-Model
Loan Default Risk Classification using Machine Learning
Project Title


Loan Default Risk Classification for Banking and Financial Risk Assessment


Team Members      

nived pd        253209

lubaba

nikhil as       253221

Course Details

Institution: Kerala University of Digital Sciences, Innovation and Technology

Programme: [Add Programme Name –  Masters program of datascience 

Academic Year:2026

1. Project Overview
2. 
1.1 Problem Statement
   

Financial institutions incur significant losses due to loan defaults. The objective of this project is to develop a predictive machine learning model capable of identifying high-risk applicants prior to loan disbursement. Early risk detection enables banks to minimize credit losses and improve portfolio quality.

1.2 Business Objective

The task is formulated as a binary classification problem:

0 — No Default
1 — Default

Given the asymmetric cost structure in banking:

A False Negative (approving a defaulter) results in direct financial loss.
A False Positive (rejecting a safe applicant) results only in opportunity loss.

Therefore, the model is optimized primarily for Recall on the Default class rather than overall accuracy.

2. Dataset Description

Source: Kaggle – Home Credit Default Risk Dataset

Total Records: 307,511 loan applications
Total Features: 122 raw features
Target Variable: Loan Default (0/1)
2.1 Class Distribution
No Default: 282,686 (91.9%)
Default: 24,825 (8.1%)

The dataset exhibits severe class imbalance (11.4:1 ratio). This was addressed using:

SMOTE oversampling on training data
scale_pos_weight parameter in LightGBM
2.2 Feature Overview

The dataset consists of:

65 Float variables
41 Integer variables
16 Categorical variables
0 duplicate records

Some property-related features contain high missingness (up to 70%).

Key financial features include:

AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
EXT_SOURCE_1/2/3
DAYS_BIRTH
DAYS_EMPLOYED
CNT_FAM_MEMBERS
3. Methodology

The project follows a complete machine learning lifecycle.

3.1 Data Preprocessing
Median imputation for numerical features
Mode imputation for categorical features
Label encoding for categorical variables
Removal of highly correlated features (|correlation| > 0.90)
StandardScaler applied for Logistic Regression
3.2 Exploratory Data Analysis (EDA)

Key findings:

Severe class imbalance (91.9% vs 8.1%)
EXT_SOURCE_1/2/3 are dominant predictive features
Income distribution is right-skewed
Credit amounts exhibit bimodal distribution
No single feature strongly predicts default; ensemble methods are necessary
3.3 Feature Engineering

Eight domain-driven financial features were engineered:

CREDIT_INCOME_RATIO
ANNUITY_INCOME_RATIO
GOODS_CREDIT_RATIO
CREDIT_TERM
AGE_YEARS
YEARS_EMPLOYED
EMPLOYED_TO_AGE_RATIO
INCOME_PER_PERSON

Feature selection was performed using Mutual Information, retaining the top 50 features for modeling.

3.4 Model Development

Three classification models were trained on SMOTE-balanced training data using an 80/20 stratified split.

Logistic Regression (Baseline Model)
class_weight = balanced
max_iter = 1000
Random Forest
n_estimators = 300
max_depth = 12
class_weight = balanced
LightGBM (Best Performing Model)
n_estimators = 500
learning_rate = 0.05
num_leaves = 63
scale_pos_weight = 11.39
4. Model Evaluation
4.1 Performance Comparison
Model	Accuracy	Recall	ROC-AUC
Logistic Regression	75.7%	0.373	0.631
Random Forest	86.2%	0.152	0.735
LightGBM	89.5%	0.540	0.768

LightGBM achieved the highest ROC-AUC and superior recall performance.

4.2 Threshold Optimization

Default classification threshold: 0.50
Optimized threshold: 0.1027

At optimized threshold:

Recall = 71.3%
Precision = 14.1%

Lowering the threshold significantly improved defaulter detection, aligning with banking risk priorities.

4.3 Cross Validation

5-Fold Cross Validation (LightGBM):

Mean ROC-AUC = 0.7585 ± 0.0043

This indicates stable and generalizable model performance.

5. Model Explainability

SHAP (SHapley Additive exPlanations) was applied to the LightGBM model to ensure interpretability and regulatory compliance.

Top contributing features:

EXT_SOURCE_3
EXT_SOURCE_2
EXT_SOURCE_1
DAYS_BIRTH
AMT_CREDIT

SHAP provides both global and local explanations, enabling transparent credit decision-making consistent with financial regulatory standards.

6. Streamlit Deployment

An interactive web-based loan risk assessment dashboard was developed using Streamlit.

Features include:

Applicant profile input (age, employment duration)
Financial details input (income, credit amount)
External credit score sliders
Real-time default probability prediction
Decision output (Approved / Rejected)
7. Application Screenshots

<img width="1600" height="720" alt="app_ss" src="https://github.com/user-attachments/assets/d001292b-b48a-4a45-b058-38d865f4e381" />

8. Project Setup Instructions
Step 1: Clone Repository
git clone https://github.com/nivedhpdds25/Loan-Default-Risk-Classification-ML-Model.git
cd Loan-Default-Risk-Classification-ML-Model
Step 2: Create Virtual Environment
python -m venv venv
venv\Scripts\activate   (Windows)
source venv/bin/activate   (Mac/Linux)
Step 3: Install Dependencies
pip install -r requirements.txt
Step 4: Run Streamlit Application
streamlit run app.py
9. Live Deployment

Repository:
https://github.com/nivedhpdds25/Loan-Default-Risk-Classification-ML-Model

Live Streamlit Application:
(https://loan-default-risk-classification-ml-model-mivjfwjinyvkabgkncyw.streamlit.app)

10. Conclusion

LightGBM emerged as the optimal model with ROC-AUC of 0.768 and significantly improved recall after threshold tuning. Feature engineering enhanced predictive capacity, while SMOTE effectively mitigated class imbalance. SHAP-based explainability ensures the model is suitable for regulated financial environments.

11. Future Work
Comparative evaluation with XGBoost and CatBoost
Integration of supplementary bureau datasets
Exploration of deep learning models such as TabNet
API-based production deployment using FastAPI
Fairness and bias auditing across demographic attributes
