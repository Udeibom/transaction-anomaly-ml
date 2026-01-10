
# Transaction Anomaly Detection System

## Overview
This project implements an **end-to-end transaction anomaly detection system** for fraud and abuse detection using **unsupervised and semi-supervised machine learning** techniques.

The system is designed to operate in realistic production conditions where **fraud labels are rare, delayed, or unreliable**, and therefore **not used during model training**. Labels are introduced only for **offline evaluation**.

The project emphasizes **ML systems thinking**: modeling, thresholding, cost trade-offs, monitoring, drift detection, and deployment.

---

## Problem Framing

### Real-World Challenge
In real transaction systems:
- Fraud labels are scarce and often delayed
- Fraud patterns evolve over time
- False positives are costly and operationally disruptive

Therefore, the system must:
- Learn *normal transaction behavior*
- Flag *behavioral deviations*, not just large amounts
- Control alert volume and false positives
- Remain stable under distribution shift

### Problem Statement
> Detect anomalous credit card transactions in near-real time using historical transaction behavior, **without relying on labeled fraud data**.

This framing mirrors how production fraud systems are built.

---

## Dataset
**Credit Card Transactions Dataset** (anonymized)

Typical fields:
- `trans_date_trans_time`
- `amt`
- PCA-anonymized numerical features
- `Class` (fraud label — withheld during training)

**Important rule:**
- ❌ Labels are not used for training, feature engineering, or thresholding  
- ✅ Labels are used only for offline evaluation

---

## System Architecture

```
Raw Transactions
      ↓
Feature Engineering
      ↓
Anomaly Models
(Isolation Forest / Autoencoder)
      ↓
Scoring & Thresholding
      ↓
Alerts / Review Queue
      ↓
Monitoring & Drift Detection
```

---

## Feature Engineering

Fraud is **behavioral deviation**, not just high value.

Key features:
- Log-scaled transaction amount
- Time since last transaction
- Transaction frequency
- Hour-of-day / day-of-week encoding
- Rolling behavioral statistics (where applicable)

All features are normalized to ensure model stability.

---

## Models

### 1. Isolation Forest (Primary Model)
- Unsupervised
- Fast and scalable
- Industry-standard for tabular anomaly detection
- Selected as the **primary production model** due to stability

### 2. Autoencoder (Secondary Model)
- Neural network trained to reconstruct normal behavior
- Reconstruction error used as anomaly score
- Provides complementary signal
- Higher operational complexity

**Model Selection Rationale:**
Isolation Forest was selected as the primary model due to its robustness, stability over time, and ease of deployment.  
The autoencoder is retained as a secondary signal for future ensembling.

---

## Thresholding Strategy

Anomaly scores are **relative**, not absolute.

Thresholding is done using **percentile-based alerting**:
- Top 0.5% → Conservative
- Top 1% → Baseline (selected operating point)
- Top 2% → Aggressive

This ensures:
- Predictable alert volume
- Business-aligned decision making
- Independence from score scale

---

## Offline Evaluation (Labels Reintroduced)

Fraud labels are used **only for evaluation**.

Metrics:
- Precision@K
- Recall@K
- ROC-AUC (contextual, not operational)

**Key Insight:**
Isolation Forest achieves higher precision at low alert rates, making it more suitable for real-world deployment.

---

## Cost-Sensitive Analysis

To tie model decisions to business impact, a cost model is defined:

- Review cost per alert
- Fraud loss per missed fraud

A cost curve is computed across alert rates to select the **operating point that minimizes total expected cost**.

**Final Operating Threshold:**  
> **Top 1% most anomalous transactions**

This balances fraud capture and review effort.

---

## Monitoring & Drift Detection

The system includes post-deployment monitoring to detect silent failures.

### Monitored Metrics
- Mean and tail anomaly scores
- Alert rate stability
- Feature distribution drift

### Drift Detection Methods
- Rolling statistics
- KS-tests on key features
- Alert rate deviation checks
- Multi-signal drift decision logic

When drift is detected, the system triggers investigation rather than automatic retraining.

---

## Inference API

A FastAPI service exposes the trained model for real-time scoring.

### Endpoint
```
POST /score_transaction
```

### Response
- Anomaly score
- Alert flag
- Threshold
- Model version

The API is stateless and assumes features are computed upstream, reflecting real production architectures.

---

## Logging & Observability

The inference service includes structured logging for:
- Requests
- Predictions
- Alerts

Logs are stored in append-only JSON format and support:
- Monitoring jobs
- Drift detection
- Auditability

---

## Repository Structure

```
transaction-anomaly-ml/
├── api/
├── data/
├── features/
├── models/
├── evaluation/
├── monitoring/
├── notebooks/
├── logs/
└── README.md
```

---

## Limitations & Future Work

- Labels are static; delayed feedback loops could be simulated
- No online learning or automated retraining
- No ensemble deployment (rank fusion planned)
- Feature pipeline assumed upstream in API

Future improvements:
- Ensemble scoring
- Active learning with analyst feedback
- Automated retraining triggers
- Dashboard-based monitoring

---

## Interview Summary Statement

> “I built an end-to-end anomaly detection system for transactions where fraud labels were withheld during training, using Isolation Forest and autoencoders with percentile-based thresholding, cost-sensitive analysis, monitoring, drift detection, and a production-style inference API.”

---

## Author
**Caleb Udeibom**  
Machine Learning Engineer  
