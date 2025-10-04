# Exoplanet Hunter AI Overview

Exoplanet Hunter AI is an advanced machine learning system designed to detect and classify exoplanets using data from NASA's Kepler mission. It automates the analysis of vast datasets to identify confirmed planets, promising candidates, and false positives, accelerating discoveries and reducing human error.

## Purpose
The system tackles the challenge of manually processing thousands of potential exoplanet candidates from Kepler data, which is time-intensive and error-prone. Its mission is to speed up classification, enabling astronomers to focus on high-potential targets that may support life.

## Technical Architecture
- **CNN-Transformer Hybrid**: Integrates convolutional neural networks (CNNs) for extracting local patterns with transformer models for long-range dependencies in stellar light curves, delivering high accuracy.
- **Ensemble Learning**: Aggregates outputs from CNN-Transformer, LightGBM, and XGBoost via a voting classifier for robust performance across varied data scenarios.
- **Constitutional AI**: Incorporates safety mechanisms like uncertainty quantification, explainable AI outputs, and flagging for ambiguous predictions to uphold scientific standards.

## How It Works
Users upload stellar observation data (e.g., light curves from Kepler). The AI preprocesses the data, applies the hybrid model and ensemble for classification, and outputs predictions with confidence scores and explanations. Uncertain cases are flagged for manual review.

## Training Data
- Based on NASA's full Kepler cumulative dataset.
- Preprocessed with techniques like SMOTE to address class imbalance, ensuring balanced performance for different exoplanet types (e.g., confirmed, candidates, false positives).

## Performance Metrics

| Metric          | Value    |
|-----------------|----------|
| Accuracy        | 83.55%  |
| Macro F1-Score  | 0.8142  |
| F1 (Confirmed)  | 0.8672  |
| F1 (False Positive) | 0.8985 |

## Key Benefits
- **Accelerated Discovery**: Analyzes thousands of candidates in seconds, shortening the path from raw data to actionable insights.
- **Uncertainty Quantification**: Uses Monte Carlo Dropout to generate confidence intervals, ensuring reliable flagging of edge cases.
- **Interpretable Results**: Provides feature importance rankings (e.g., transit depth, period) to explain predictions and foster trust.
- **Scalability**: Handles diverse exoplanet signals with minimal bias due to comprehensive training.

## Safety and Reliability Features
- **Uncertainty Estimation**: Flags low-confidence predictions for expert review.
- **Explainability**: Visualizes key influencing factors in decisions.
- **Integrity Checks**: Automated validation to prevent false discoveries or overlooked candidates.

## Get Started
Ready to classify your data? Visit the [Exoplanet Hunter AI dashboard](https://exoplanet-hunter-ai-frontend.onrender.com/) to upload stellar observations and start discovering exoplanets with state-of-the-art AI.
