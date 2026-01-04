# Transaction Anomaly Detection System

## Problem Statement

This project aims to detect anomalous credit card transactions in near-real time by learning normal transaction behavior from historical data.

In real-world fraud detection systems, fraud labels are often rare, delayed, or unreliable. As a result, this system is designed to operate without using labeled fraud data during training. Instead, it models normal transaction patterns and flags transactions that deviate significantly from expected behavior.

Each transaction is assigned an anomaly score, and the most unusual transactions are surfaced for further review while controlling false positives.

## What Is Considered an Anomaly?

An anomaly is defined as a transaction that significantly deviates from a cardholder’s historical behavior. This deviation may be related to transaction amount, timing, frequency, or changes in spending patterns over time.

Fraud is treated as a behavioral shift rather than simply a high transaction amount.

## Near Real-Time Assumption

Near real-time refers to scoring transactions within seconds to minutes after they occur. In this project, near real-time behavior is simulated using batch-based processing while preserving transaction order and historical context.

## Constraints and Assumptions

- Fraud labels are **not used during model training**
- Labels are used **only for offline evaluation**
- The system starts with batch scoring and can be extended to streaming inference
- Focus is on unsupervised and semi-supervised anomaly detection methods
