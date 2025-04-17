# Network Anomaly Detection using Machine Learning

This project investigates and compares the performance of three powerful machine learning models—**XGBoost**, **Autoencoders**, and **Generative Adversarial Networks (GANs)**—for detecting anomalies and intrusions in network traffic using the **KDD Cup dataset**. The aim is to build a robust intrusion detection system capable of identifying both known and unknown threats while ensuring high accuracy and efficiency.

---

## 📌 Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Scope](#scope)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [XGBoost](#xgboost)
  - [Autoencoder](#autoencoder)
  - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## 🧠 Introduction

This project explores machine learning approaches to detect intrusions in network traffic by analyzing the **KDD Cup dataset**. It evaluates the performance of:
- **XGBoost**: For supervised classification
- **Autoencoders**: For unsupervised anomaly detection
- **GANs**: For modeling normal behavior and identifying deviations

---

## 🎯 Motivation

The need for **intelligent, adaptive, and real-time intrusion detection systems** has never been more critical. Traditional systems fail to detect emerging threats and generate many false positives. Machine learning can address these shortcomings by learning patterns in network traffic and adapting to new threats.

---

## ❗ Problem Statement

- High false positive rates in current IDS
- Difficulty detecting zero-day or novel attacks
- Imbalance in network traffic datasets
- High computational requirements for processing large volumes of traffic

---

## 🎯 Objectives

- Develop a comparative analysis of ML-based anomaly detection systems
- Achieve high accuracy in binary and multi-class intrusion detection
- Evaluate supervised vs. unsupervised learning for anomaly detection
- Minimize false positives and maximize recall and precision

---

## 📌 Scope

✅ Included:
- Dataset preprocessing
- Feature engineering
- Implementation of XGBoost, Autoencoders, and GANs
- Model training and evaluation
- Binary and multi-class classification

❌ Excluded:
- Real-time production deployment
- Encrypted traffic handling
- Hardware-specific optimizations

---

## 📁 Dataset

- **Name**: KDD Cup 1999 Dataset
- **Features**: 41 (categorical and numerical)
- **Size**: Large, with extreme class imbalance (only 1% anomalies)

---

## ⚙️ Methodology

### XGBoost
- Gradient-boosted trees for fast and accurate classification
- Works on both binary and multi-class labels
- GPU-accelerated for faster training

### Autoencoder
- Deep neural network for reconstructing input
- Detects anomalies based on reconstruction error
- Useful for unsupervised detection of novel attacks

### Generative Adversarial Networks (GANs)
- Generator learns to replicate normal network behavior
- Discriminator detects deviations (anomalies)
- Powerful approach for learning complex distributions

---

## 🔧 Preprocessing Pipeline

- Encoding of categorical variables
- Feature scaling and normalization
- Handling imbalanced data via sampling
- Latent space encoding (for Autoencoders)
- Noise filtering and outlier reduction

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## ✅ Results

| Model        | Accuracy (%) | Recall (%) | F1 Score (%) |
|--------------|--------------|------------|--------------|
| XGBoost      | **99.98**    | 99.85      | 99.92        |
| Autoencoder  | 97.45        | 96.92      | 97.18        |
| GAN          | 99.87        | **93.43**  | **93.77**    |

- **XGBoost** is highly effective for supervised tasks with labeled data.
- **Autoencoders** are useful for identifying unknown anomalies via reconstruction loss.
- **GANs** excel at recognizing complex, novel patterns and modeling normal behavior.

---

## 💡 Conclusion

Each model demonstrates unique strengths in network anomaly detection:
- **XGBoost**: Best for high-accuracy classification of known attacks.
- **Autoencoders**: Ideal for discovering new threats without labeled data.
- **GANs**: Combines generative modeling and detection for robust, adaptive systems.

The combination or ensemble of these models can lead to an **advanced intrusion detection system** capable of handling diverse cyber threats.

---

## 🛠 Installation

```bash
git clone https://github.com/teja85053/network-anomaly-detection.git
cd network-anomaly-detection
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# For XGBoost training
python train_xgboost.py

# For Autoencoder training
python train_autoencoder.py

# For GAN training
python train_gan.py

# For evaluation and result visualization
python evaluate_models.py
```

Make sure to place the KDD dataset in the appropriate data/ folder and preprocess it accordingly.

## 🔑 Keywords

Network Intrusion Detection, XGBoost, Autoencoder, GAN, Anomaly Detection, KDD Cup, Cybersecurity, Machine Learning, Supervised Learning, Unsupervised Learning, Deep Learning