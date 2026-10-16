# Guest Shield — Hotel Cancellation Prediction System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Classification-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/AWS%20Lambda-Serverless-FF9900?style=for-the-badge&logo=awslambda&logoColor=white)
![API Gateway](https://img.shields.io/badge/API%20Gateway-HTTP%20API-FF4F8B?style=for-the-badge&logo=amazonapigateway&logoColor=white)
![AWS SAM](https://img.shields.io/badge/AWS%20SAM-Infrastructure%20as%20Code-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)

</p>

---

## 📌 Project Overview

**Guest Shield** is an end-to-end machine learning system for predicting hotel booking cancellations **before the guest's arrival**. It began as a comparative predictive analytics research project and has since been extended into a **containerized, serverless prediction API deployed on AWS**.

The research investigates booking behavior, customer segmentation, class imbalance, and alternative classification algorithms. The deployed inference system uses **KMeans clustering and cluster-specific Random Forest classifiers** to estimate cancellation risk for a new reservation.

The original research, experiments, and detailed findings are retained in [`Guest Shield_Report.pdf`](Guest%20Shield_Report.pdf) and the [`notebooks/`](notebooks/) directory.

**Current status:** Deployed and tested on **AWS Lambda + Amazon API Gateway**, with **AWS IAM (SigV4) authorization**. The prediction endpoint is intentionally **not public/anonymous**.

---

# Business Problem

Booking cancellations create uncertainty in occupancy planning, room allocation, staffing, pricing, and expected revenue. A reservation-level cancellation probability can help hotel teams identify bookings that may benefit from additional confirmation or other carefully designed interventions.

> **How likely is a hotel reservation to be cancelled?**

Guest Shield demonstrates how to convert historical reservations into a predictive model and expose that model through a secure API for use by downstream systems.

---

# Key Features

- **Exploratory analysis and research:** EDA and comparisons of Logistic Regression, SVM, Random Forest, and XGBoost.
- **Reproducible training:** A Python script loads the dataset, engineers features, trains models, evaluates predictions, and exports an inference bundle.
- **Customer segmentation:** KMeans assigns each processed reservation to one of two clusters.
- **Cluster-specific classification:** A dedicated Random Forest model predicts the outcome within each cluster.
- **Class imbalance handling:** SMOTE is applied to the training split only.
- **Risk output:** Predicted class, cancellation probability, percentage risk, cluster ID, and model version.
- **Containerized inference:** A Docker image packages the API handler, runtime dependencies, and trained model.
- **Serverless AWS deployment:** AWS SAM provisions Lambda and an HTTP API on API Gateway.
- **IAM-secured endpoint:** Requests require AWS Signature Version 4 signing; anonymous requests are rejected.
- **Observability:** Lambda execution logs and runtime metrics are available through Amazon CloudWatch.

---

# Solution Architecture

## Machine Learning Training

```text
Historical Hotel Reservations (CSV)
                │
                ▼
       Cleaning & Date Features
                │
                ▼
      Stratified 80/20 Split
                │
                ▼
  Encoder fitted on Training Data
                │
                ▼
        SMOTE (Train Only)
                │
                ▼
       KMeans (2 Clusters)
                │
          ┌─────┴─────┐
          ▼           ▼
    Random Forest  Random Forest
      Cluster 0      Cluster 1
          └─────┬─────┘
                ▼
      Held-out Test Evaluation
                │
                ▼
  guest_shield_model.joblib
 (Preprocessor + KMeans + RFs
        + Label Encoder)
```

## Deployed AWS Architecture

```text
    Authorized Client / Python Test
                 │
       POST /prod/predict
        AWS SigV4 Signed
                 │
                 ▼
      Amazon API Gateway
        HTTP API (AWS_IAM)
                 │
                 ▼
            AWS Lambda
     Python 3.13 Container Image
                 │
                 ▼
       Guest Shield Handler
                 │
                 ▼
      Feature Preprocessing
                 │
                 ▼
           KMeans Routing
                 │
          ┌──────┴──────┐
          ▼             ▼
     Random Forest  Random Forest
       Cluster 0      Cluster 1
          └──────┬──────┘
                 ▼
        JSON Risk Prediction

 Supporting AWS Services:
 ─ Amazon ECR: Lambda container image
 ─ AWS IAM: execution roles and API authorization
 ─ Amazon CloudWatch: logs and execution metrics
 ─ AWS SAM / CloudFormation: infrastructure as code
```

**Deployment region:** `ap-southeast-1` (Singapore). The API URL is generated as a CloudFormation stack output after deployment rather than hardcoded in this repository.

---

# Dataset and Research

The original dataset is stored at [`data/Hotel Reservations.csv`](data/Hotel%20Reservations.csv). Features include booking lead time, room price, stay duration, customer composition, meal plan, room type, market segment, past cancellations, and special requests.

The original research evaluates several modeling approaches:

| Method | Role in the project |
| --- | --- |
| Logistic Regression | Interpretable classification baseline |
| Support Vector Machine | Alternative classification experiments |
| Random Forest | Nonlinear classification and deployed cluster-specific models |
| XGBoost | Boosted-tree comparison in research notebooks |
| KMeans | Segmentation and routing in the deployed inference workflow |

The research report contains the detailed comparative findings: **[`Guest Shield_Report.pdf`](Guest%20Shield_Report.pdf)**.

---

# Data Preprocessing and Model Training

The training script [`scripts/export_model.py`](scripts/export_model.py) performs these operations:

1. Load reservation data and remove the booking identifier.
2. Correct invalid February 29, 2018 dates and derive arrival-day features.
3. Split observations into **80% training / 20% testing**, stratified by booking outcome.
4. Fit a `ColumnTransformer` with one-hot encoding on training data and transform the test split using the same encoder.
5. Encode the target and apply **SMOTE only to training data**.
6. Fit **KMeans with two clusters** to the resampled training features.
7. Train one **Random Forest** per cluster.
8. Route held-out test reservations to the appropriate classifier and evaluate predictions.
9. Save the full inference bundle to `models/guest_shield_model.joblib`.

The deployed training run achieved **89.91% held-out test accuracy**. This is a test-set result from the implemented pipeline, not a guarantee of future performance on live hotel reservations. Precision, recall, class-specific errors, and probability calibration also matter for operational decisions.

> **Modeling note:** The current training script applies standard SMOTE after one-hot encoding. Future experimentation should evaluate categorical-aware resampling (such as SMOTENC) and probability calibration.

---

# 🛠 Technology Stack

| Layer | Technologies |
| --- | --- |
| Programming | Python 3.13 |
| Data preparation | Pandas, NumPy |
| Machine learning | Scikit-learn, imbalanced-learn |
| Research experiments | Jupyter Notebooks, Logistic Regression, SVM, Random Forest, XGBoost |
| Model serialization | Joblib |
| API runtime | AWS Lambda Python container |
| Containerization | Docker |
| API management | Amazon API Gateway HTTP API |
| Container registry | Amazon Elastic Container Registry (ECR) |
| Infrastructure as code | AWS SAM, AWS CloudFormation |
| Security | AWS IAM authorization, SigV4 signed requests |
| Monitoring | Amazon CloudWatch |
| Development tools | Git, GitHub, AWS CLI, SAM CLI, PowerShell |

---

# 📂 Repository Structure

```text
hotel-cancellation-prediction/
│
├── app/
│   └── handler.py                 # Lambda prediction handler
├── data/
│   └── Hotel Reservations.csv    # Training dataset
├── models/
│   └── guest_shield_model.joblib # Generated locally; ignored by Git
├── notebooks/                    # Original research experiments
├── scripts/
│   └── export_model.py           # Train, evaluate, and export model
├── tests/
│   ├── test_local.py             # Local inference tests
│   └── test_aws_api.py           # Signed deployed-API test
│
├── Dockerfile                    # Lambda container build
├── template.yaml                 # AWS SAM infrastructure
├── requirements.txt             # Inference dependencies
├── requirements-training.txt    # Training dependencies
├── .dockerignore
├── .gitignore
├── Guest Shield_Report.pdf       # Detailed research report
└── README.md
```

`samconfig.toml`, `.venv/`, `.aws-sam/`, `.env`, and generated model binaries are excluded from version control. The trained `.joblib` bundle is not published as a Git-tracked file; **train it locally before building or deploying the container**.

---

# 🚀 Getting Started (Windows / PowerShell)

## Prerequisites

- Python **3.13** and `pip`
- Git
- Docker Desktop (running for container builds)
- AWS CLI and AWS SAM CLI (for AWS deployment)
- An AWS account and appropriately scoped permissions (for deployment and API invocation)

## 1. Clone the Repository

```powershell
git clone https://github.com/supulwickramasinghe/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
```

## 2. Set Up the Python Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Train and Export the Model

```powershell
python -m pip install -r requirements-training.txt
python scripts/export_model.py
```

This creates `models/guest_shield_model.joblib`, containing the trained preprocessing, clustering, classification, and label-encoding objects. The script prints evaluation results and overwrites an existing local model bundle.

## 4. Install Inference Dependencies and Test Locally

```powershell
python -m pip install -r requirements.txt
python tests/test_local.py
```

The local test should confirm that the handler can load the model and produce a prediction. Check the script output for its specific assertions/results.

---

# 🐳 Docker and Local Container Build

The Lambda container image is defined in [`Dockerfile`](Dockerfile). It uses the AWS Lambda Python 3.13 base image and includes the model bundle generated in the previous section.

```powershell
docker build -t guest-shield-api:local .
```

The `.dockerignore` file excludes notebooks, research data, training scripts, virtual environments, and other files that aren't needed inside the production inference container.

For a local runtime test, use `tests/test_local.py`. Docker Desktop must be running for image builds and SAM container-based builds.

---

# ☁️ Deploy to AWS with SAM

The AWS resources are declared in [`template.yaml`](template.yaml): an API Gateway **HTTP API**, a **container-image Lambda function**, the associated API integration and permissions, and stack outputs. The API's OpenAPI security definition enforces **AWS_IAM** authentication on `POST /predict`.

## 1. Configure AWS Credentials

Configure an AWS CLI profile using your preferred supported authentication method, and confirm the intended account and region:

```powershell
aws sts get-caller-identity --profile guest-shield
```

`guest-shield` is the profile name used in the example commands; substitute your own profile if different. **Never commit AWS credentials, tokens, or `.env` files.**

## 2. Validate and Build

```powershell
sam validate --template-file template.yaml --profile guest-shield --region ap-southeast-1
sam build --profile guest-shield --region ap-southeast-1
```

The build requires the generated model file and a working Docker installation.

## 3. Deploy

On a fresh machine or AWS account, run the guided deployment:

```powershell
sam deploy --guided --profile guest-shield --region ap-southeast-1
```

Review the prompts, CloudFormation changeset, IAM capabilities, and estimated resource implications before confirming. SAM can build/push the container image to an ECR repository managed or configured for the deployment. The generated local `samconfig.toml` is excluded from Git.

For subsequent deployments, after reviewing changes:

```powershell
sam deploy --profile guest-shield --region ap-southeast-1 --confirm-changeset
```

**Deployed stage:** `prod`
**Route:** `POST /predict`
**Full URL shape:** `https://<api-id>.execute-api.ap-southeast-1.amazonaws.com/prod/predict`

Get the deployed URL from the CloudFormation outputs:

```powershell
aws cloudformation describe-stacks `
  --stack-name guest-shield-api `
  --profile guest-shield `
  --region ap-southeast-1 `
  --query "Stacks[0].Outputs" `
  --output table `
  --no-cli-pager
```

---

# 🔐 API Security and Testing

The deployed endpoint **does not permit anonymous prediction requests**. API Gateway requires **AWS Signature Version 4**, using credentials belonging to a principal that is explicitly allowed to invoke the API.

| Test | Verified result |
| --- | --- |
| Unsigned POST request | **403 Forbidden** |
| Properly signed POST request | **200 OK** |
| SAM redeployment | IAM authorization preserved |

IAM authentication on a route does not automatically authorize every AWS identity. The caller also requires the appropriate `execute-api:Invoke` permission. Use least-privilege access and avoid embedding long-lived credentials in application code.

## Signed AWS API Test

Once your AWS profile can authenticate and has invoke permissions, run:

```powershell
python tests/test_aws_api.py
```

The test uses AWS credentials from the configured profile to sign the request and verify the deployed API response. If your local deployment has a different API URL, update the test configuration accordingly.

## Example Prediction Response

```json
{
  "prediction": "Not_Canceled",
  "cancellation_probability": 0.2943,
  "cancellation_risk_percent": 29.43,
  "cluster_id": 0,
  "model_version": "1.0.0"
}
```

This sample illustrates a reservation predicted **not to cancel**, with an estimated **29.43% cancellation probability**. Probabilities are model estimates; they should not be treated as calibrated financial-risk scores without further validation.

---

# 📊 Monitoring, Performance, and Costs

**Amazon CloudWatch** provides Lambda logs, errors, duration, and memory-use information. API Gateway and Lambda usage can also be reviewed through AWS service metrics and billing tools.

During initial testing, cold-start initialization was notably slower than warm execution because the runtime loads the ML dependencies and serialized models. The deployed configuration uses:

| Setting | Value |
| --- | --- |
| Lambda memory | **1,024 MB** |
| Lambda timeout | **30 seconds** |
| Reserved concurrency | Not explicitly configured |
| AWS region | `ap-southeast-1` |

The original deployment was tested with successful inference responses and AWS IAM authorization. Actual latency and charges depend on traffic, runtime conditions, and AWS pricing.

A budget alert can provide early warning, but **AWS Budgets alerts do not automatically stop spending**. Monitor Lambda, API Gateway, ECR image storage, CloudWatch logs, and any deployment artifacts regularly.

---

# 🧹 Remove AWS Resources

When you are finished experimenting, you can remove the CloudFormation stack:

```powershell
sam delete --stack-name guest-shield-api --profile guest-shield --region ap-southeast-1
```

**Destructive action:** This removes deployed stack resources. Review the deletion prompts carefully; any separately created ECR repositories, retained artifacts, logs, or other AWS resources may require additional cleanup. Verify the Billing console afterward.

---

# Business Interpretation

Guest Shield supplies a probability estimate that could support:

- Reservation confirmation campaigns for bookings identified as higher risk.
- Occupancy and revenue forecasting informed by cancellation likelihood.
- Review of cancellation patterns across booking segments.
- Further assessment of booking policies and customer communication strategies.

The current API is a **machine learning demonstration**, not a fully integrated hotel reservation system. Any business intervention should be tested for accuracy, fairness, customer experience, and financial impact before operational use.

---

# Future Improvements

The following are **future enhancements**, not features of the current AWS deployment:

- Probability calibration, feature attribution (e.g., SHAP), and threshold optimization.
- Evaluate categorical-aware resampling and compare alternative modeling strategies.
- CI/CD with GitHub Actions and automated test gates.
- Automated retraining, model registry/versioning, and drift monitoring.
- API request validation enhancements, throttling, and additional operational controls.
- Interactive booking-risk dashboard or hotel-system integration.

---

# 🎓 Project Context

Guest Shield combines **predictive analytics research** with a practical **ML deployment workflow**:

```text
Business Problem
      ↓
Historical Reservation Data
      ↓
Research & Feature Engineering
      ↓
Trained / Evaluated Model
      ↓
Docker Container
      ↓
AWS Lambda + API Gateway
      ↓
IAM-Secured Prediction API
      ↓
Potential Hotel Decision Support
```

For the comprehensive methodology, model comparisons, and original research results, see **[`Guest Shield_Report.pdf`](Guest%20Shield_Report.pdf)**.

---
