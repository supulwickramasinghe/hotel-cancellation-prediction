# Guest Shield — Hotel Cancellation Prediction System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge\&logo=pandas)

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)

![Random Forest](https://img.shields.io/badge/Random%20Forest-Classification-green?style=for-the-badge)

![XGBoost](https://img.shields.io/badge/XGBoost-Classification-EC6B23?style=for-the-badge)

![SVM](https://img.shields.io/badge/SVM-Classification-blueviolet?style=for-the-badge)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Cancellation%20Prediction-orange?style=for-the-badge)

</p>

---

## 📌 Project Overview

**Guest Shield** is a machine learning–based hotel booking cancellation prediction system designed to identify reservations that are likely to be cancelled before the guest's arrival.

Hotel cancellations create significant operational and financial challenges. They affect occupancy planning, room allocation, pricing decisions, staffing, and revenue forecasting.

This project applies **exploratory data analysis, feature preprocessing, customer segmentation, class imbalance handling, classification modeling, and model evaluation** to historical hotel reservation data.

Multiple machine learning approaches are explored, including:

* Logistic Regression
* Support Vector Machines
* Random Forest
* XGBoost

The project demonstrates an end-to-end predictive analytics workflow:

```text
Hotel Reservation Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Customer Segmentation
        ↓
Class Imbalance Handling
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Cancellation Risk Prediction
```

---

# Business Problem

Booking cancellations are a major challenge in the hospitality industry.

When a customer cancels a reservation, hotels may experience:

* Lost room revenue
* Lower occupancy rates
* Inefficient room allocation
* Inaccurate demand forecasting
* Staffing inefficiencies
* Last-minute discounting
* Reduced operational profitability

Hotels therefore benefit from identifying **high-risk reservations before the cancellation occurs**.

The objective of Guest Shield is to build a predictive system capable of answering:

> **How likely is a hotel reservation to be cancelled?**

Accurate cancellation predictions can support proactive business decisions such as:

* Targeted confirmation campaigns
* Deposit requirements
* Dynamic overbooking strategies
* Personalized offers
* Revenue management
* Improved occupancy planning

---

# Key Features

 Exploratory Data Analysis

 Data Cleaning & Validation

 Feature Engineering

 Customer Booking Behaviour Analysis

 Reservation Pattern Analysis

 Customer Segmentation

 Classification Modeling

 Logistic Regression

 Support Vector Machine

 Random Forest

 XGBoost

 Class Imbalance Handling

 Model Comparison

 Confusion Matrix Analysis

 Precision / Recall / F1 Evaluation

 Cancellation Risk Prediction

 Business-Oriented Model Evaluation

---

# Solution Architecture

```text
                 Hotel Reservations Dataset
                            │
                            ▼
                    Data Validation
                            │
                    Missing / Duplicate
                         Checks
                            │
                            ▼
                 Exploratory Data Analysis
                            │
                            ▼
                   Feature Engineering
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Numerical       Categorical     Booking
         Features        Features       Behaviour
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    Data Preprocessing
                            │
                            ▼
                 Customer Segmentation
                            │
                            ▼
                Classification Modeling
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
      Logistic            SVM            Random Forest
      Regression                                │
                                                ▼
                                             XGBoost
                                                │
                                                ▼
                                      Model Evaluation
                                                │
                                  Precision • Recall • F1
                                      Accuracy • ROC-AUC
                                                │
                                                ▼
                                   Cancellation Prediction
                                                │
                                                ▼
                                     Business Intervention
```

---

# Dataset

The project uses the:

```text
Hotel Reservations.csv
```

dataset containing historical hotel booking information.

The repository stores the dataset under:

```text
data/Hotel Reservations.csv
```

Reservation data contains variables describing customer, room, booking, and reservation behaviour.

Typical predictors used in hotel cancellation modeling include information such as:

* Number of adults
* Number of children
* Weekend nights
* Week nights
* Meal plan
* Required car parking space
* Room type
* Lead time
* Arrival year
* Arrival month
* Arrival date
* Market segment
* Repeated guest status
* Previous cancellations
* Previous successful bookings
* Average room price
* Special requests
* Booking status

The target represents whether the reservation was:

```text
Canceled
or
Not Canceled
```

---

# Exploratory Data Analysis

The EDA stage investigates both customer behaviour and booking characteristics associated with hotel cancellations.

The analysis explores areas such as:

* Booking cancellation distribution
* Customer composition
* Lead time
* Room prices
* Market segments
* Number of special requests
* Arrival patterns
* Repeated customers
* Previous cancellation behaviour
* Stay duration
* Room types

The EDA notebook is located at:

```text
notebooks/EDA.ipynb
```

Understanding these patterns is important because different reservation characteristics may influence cancellation probability.

---

# Data Preprocessing

Before model training, reservation data is prepared for machine learning.

Typical preprocessing tasks include:

```text
Raw Reservation Data
        ↓
Data Quality Checks
        ↓
Categorical Encoding
        ↓
Numerical Feature Preparation
        ↓
Train/Test Split
        ↓
Class Imbalance Handling
        ↓
Model Training
```

The resulting dataset becomes suitable for supervised classification algorithms.

---

# Customer Segmentation

One of the more advanced components of Guest Shield is the analysis of heterogeneous booking behaviour.

Rather than assuming all hotel customers behave identically, customer reservations can be grouped according to booking characteristics.

Segmentation helps reveal groups with different:

* Cancellation behaviour
* Booking lead times
* Price sensitivity
* Reservation characteristics
* Customer behaviour

This allows predictive models to better capture differences between customer groups.

---

# Class Imbalance

Hotel cancellation datasets may contain unequal numbers of cancelled and non-cancelled reservations.

This means **accuracy alone may not provide a complete evaluation of model quality**.

For example, a model could achieve high accuracy while failing to identify many actual cancellations.

Guest Shield therefore considers classification metrics such as:

| Metric        | Meaning                                       |
| ------------- | --------------------------------------------- |
| **Precision** | How many predicted cancellations were correct |
| **Recall**    | How many actual cancellations were detected   |
| **F1-score**  | Balance between Precision and Recall          |
| **Accuracy**  | Overall classification correctness            |
| **ROC-AUC**   | Overall discrimination ability                |

From a hotel business perspective, **Recall can be particularly important** because missing a high-risk booking could result in lost revenue.

---

# Machine Learning Models

Multiple classification models are explored and compared.

---

## Logistic Regression

Logistic Regression provides an interpretable baseline model for estimating cancellation probability.

The repository includes multiple Logistic Regression experiments:

```text
notebooks/Binary_Logistic.ipynb
notebooks/Logistic_0.ipynb
notebooks/Logistic_1.ipynb
```

Logistic Regression is useful because the resulting probability can be interpreted as a reservation's cancellation risk.

---

# Support Vector Machine

Support Vector Machines are evaluated as an alternative classification approach.

The repository contains:

```text
notebooks/SupportVector_0.ipynb
notebooks/SupportVector_1.ipynb
```

SVM attempts to construct an optimal decision boundary separating:

```text
Cancelled Reservations
        from
Non-Cancelled Reservations
```

SVM models can capture more complex decision boundaries than standard linear models.

---

# Random Forest

Random Forest is used to capture nonlinear relationships and interactions within customer booking behaviour.

Random Forest combines multiple decision trees to create a robust ensemble prediction.

It can identify complex relationships involving factors such as:

* Lead time
* Average room price
* Market segment
* Special requests
* Previous cancellations
* Stay length
* Customer characteristics

The implementation is contained in:

```text
notebooks/RandomForest and XGboost.ipynb
```

---

# XGBoost

XGBoost is evaluated as a gradient boosting approach for hotel cancellation classification.

Rather than training independent trees, XGBoost builds trees sequentially, where each new tree attempts to correct errors produced by earlier trees.

This allows XGBoost to capture complex nonlinear relationships between booking characteristics and cancellation behaviour.

The XGBoost experiments are located in:

```text
notebooks/RandomForest and XGboost.ipynb
```

---

# Model Evaluation

The machine learning models are evaluated using classification metrics and confusion matrices.

A confusion matrix separates predictions into:

```text
                    Predicted
                 Stay     Cancel
Actual Stay       TN         FP
Actual Cancel     FN         TP
```

Where:

* **TP** — correctly predicted cancellations
* **TN** — correctly predicted completed bookings
* **FP** — bookings incorrectly predicted to cancel
* **FN** — actual cancellations that were missed

For hotel operations, False Negatives can be particularly costly because they represent customers who cancel despite being predicted to complete their booking.

---

# Model Comparison

The project compares several families of classification algorithms:

| Model                  | Main Strength                                    |
| ---------------------- | ------------------------------------------------ |
| Logistic Regression    | Interpretability                                 |
| Support Vector Machine | Flexible decision boundaries                     |
| Random Forest          | Nonlinear relationships and feature interactions |
| XGBoost                | High-performance boosted classification          |

The objective is not simply to identify the model with the highest accuracy.

The preferred model should provide a suitable balance between:

```text
Recall
Precision
F1-score
Accuracy
ROC-AUC
```

depending on the hotel's business objective.

---

# Business Interpretation

A predictive cancellation model becomes most valuable when predictions lead to business actions.

For example:

```text
Reservation
     ↓
Cancellation Model
     ↓
Predicted Risk
     ↓
┌──────────┬─────────────┬───────────┐
▼          ▼             ▼
Low       Medium         High
Risk       Risk          Risk
│           │              │
▼           ▼              ▼
Normal    Reminder      Deposit /
Process   Campaign      Confirmation
```

This transforms a machine learning prediction into an operational decision-support system.

---

# Example Business Use Cases

## High-Risk Booking Identification

Identify reservations with high cancellation probabilities and prioritize them for additional confirmation.

---

## Revenue Management

Use predicted cancellation risk when planning expected room occupancy and revenue.

---

## Overbooking Strategy

Predicted cancellation probabilities can support controlled overbooking decisions.

---

## Customer Communication

High-risk bookings could receive:

* Confirmation emails
* Reminder notifications
* Personalized offers
* Flexible rescheduling options

---

## Deposit Policies

Hotels could introduce different deposit policies based on predicted booking risk.

---

## Marketing Optimization

Customer booking behaviour can be used to identify segments that require different retention or promotional strategies.

---

# End-to-End Workflow

```text
Hotel Reservation Dataset
        ↓
Data Inspection
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Customer Segmentation
        ↓
Train / Test Split
        ↓
Class Imbalance Handling
        ↓
Model Development
        ↓
┌─────────────────────────────┐
│ Logistic Regression         │
│ Support Vector Machine      │
│ Random Forest               │
│ XGBoost                     │
└─────────────────────────────┘
        ↓
Model Evaluation
        ↓
Precision
Recall
F1-score
Accuracy
ROC-AUC
        ↓
Model Comparison
        ↓
Cancellation Risk Prediction
        ↓
Hotel Business Decisions
```

---

# 🛠 Technology Stack

| Layer                   | Technologies                                     |
| ----------------------- | ------------------------------------------------ |
| Programming             | Python                                           |
| Data Manipulation       | Pandas, NumPy                                    |
| Data Analysis           | Python, Jupyter Notebook                         |
| Visualization           | Matplotlib, Seaborn                              |
| Machine Learning        | Scikit-learn                                     |
| Classification          | Logistic Regression, SVM                         |
| Ensemble Learning       | Random Forest                                    |
| Gradient Boosting       | XGBoost                                          |
| Model Evaluation        | Confusion Matrix, Precision, Recall, F1, ROC-AUC |
| Development Environment | Jupyter Notebook                                 |
| Version Control         | Git & GitHub                                     |

---

# 📂 Repository Structure

```text
hotel-cancellation-prediction/
│
├── data/
│   └── Hotel Reservations.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Binary_Logistic.ipynb
│   ├── Logistic_0.ipynb
│   ├── Logistic_1.ipynb
│   ├── SupportVector_0.ipynb
│   ├── SupportVector_1.ipynb
│   └── RandomForest and XGboost.ipynb
│
├── Guest Shield_Report.pdf
│
└── README.md
```

---

# Project Files

## Exploratory Data Analysis

```text
notebooks/EDA.ipynb
```

Contains exploratory analysis and visualization of hotel reservation behaviour.

---

## Logistic Regression

```text
notebooks/Binary_Logistic.ipynb
notebooks/Logistic_0.ipynb
notebooks/Logistic_1.ipynb
```

Contains Logistic Regression experiments for hotel cancellation classification.

---

## Support Vector Machine

```text
notebooks/SupportVector_0.ipynb
notebooks/SupportVector_1.ipynb
```

Contains SVM-based hotel cancellation classification experiments.

---

## Random Forest & XGBoost

```text
notebooks/RandomForest and XGboost.ipynb
```

Contains ensemble and boosted tree models used to predict reservation cancellations.

---

## Project Report

```text
Guest Shield_Report.pdf
```

Contains the detailed project methodology, analysis, modeling results, and business interpretation.

---

# Prediction Concept

For a new reservation, the system can use booking characteristics such as:

```text
Lead Time
Number of Adults
Number of Children
Stay Duration
Room Type
Meal Plan
Market Segment
Average Room Price
Special Requests
Repeated Guest Status
Previous Cancellations
```

These features are transformed and passed into the trained classification model.

```text
New Hotel Reservation
        ↓
Feature Preparation
        ↓
Machine Learning Model
        ↓
Cancellation Probability
        ↓
Risk Classification
        ↓
Hotel Intervention
```

---

# Why Cancellation Prediction Matters

Traditional hotel systems record cancellations only **after they happen**.

Machine learning enables hotels to move from:

```text
Reactive Analytics
```

to:

```text
Predictive Analytics
```

Instead of asking:

> "How many bookings were cancelled last month?"

the hotel can ask:

> "Which current reservations are most likely to cancel?"

This allows hotels to act before revenue is lost.

---

# Future Improvements

Potential improvements include:

* Hyperparameter optimization
* CatBoost classification
* SHAP explainability
* Feature importance dashboards
* Probability calibration
* Cost-sensitive learning
* Automated class imbalance handling
* MLflow experiment tracking
* Flask prediction API
* Interactive web application
* Docker containerization
* AWS deployment
* API Gateway integration
* AWS Lambda inference
* Amazon ECR
* Amazon S3 model storage
* CloudWatch monitoring
* CI/CD with GitHub Actions
* Automated model retraining
* Model drift monitoring
* REST API deployment
* Real-time cancellation scoring

---

# ☁️ Potential Cloud Architecture

A production-oriented extension of Guest Shield could use a cloud architecture such as:

```text
Hotel Reservation System
          │
          ▼
      REST API
          │
          ▼
     API Gateway
          │
          ▼
     AWS Lambda
          │
          ▼
Cancellation Prediction Model
          │
          ▼
     Prediction Result
          │
          ▼
Hotel Reservation / Revenue System
```

Supporting components could include:

```text
Amazon S3
Amazon ECR
AWS Lambda
API Gateway
CloudWatch
IAM
GitHub Actions
```

This would transform the notebook-based machine learning project into a deployable prediction service.

---

# 🎓 Project Context

Guest Shield demonstrates the application of:

* Exploratory data analysis
* Data preprocessing
* Customer behaviour analysis
* Classification modeling
* Class imbalance handling
* Machine learning evaluation
* Predictive analytics
* Business decision support

to a real-world hospitality problem.

The project demonstrates the transition from:

```text
Business Problem
      ↓
Historical Data
      ↓
Exploratory Analysis
      ↓
Machine Learning
      ↓
Risk Prediction
      ↓
Business Action
```

---
