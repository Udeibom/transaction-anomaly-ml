# Feature Validation Report

## Overview
This report validates behavioral features engineered for unsupervised transaction anomaly detection.

## Sanity Checks
- Feature ranges are reasonable
- Heavy-tailed distributions observed for time and amount-based features
- No impossible values detected

## Leakage Analysis
- No future information used in rolling features
- Fraud labels were excluded from feature computation
- Correlation with fraud labels is weak, as expected

## Correlation Analysis
- Some correlation between amount-based features
- No perfect multicollinearity observed
- Features retained for complementary signal coverage

## Conclusion
Feature set is stable, non-leaky, and suitable for anomaly modeling.
