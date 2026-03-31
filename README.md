# 🔩 Intelligent Bearing Fault Diagnosis & Predictive Maintenance Platform

A production-grade machine learning platform for classifying fault types in rotating machinery (ball bearings), robust transferring of models from laboratory environments to real-world deployment, and providing actionable prognostic intelligence.

![System Overview](https://img.shields.io/badge/Status-In%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-green)

---

## 📖 Overview

Bearings are crucial components of most rotating machinery, and their failure is responsible for the majority of severe machine breakdowns. While Machine Learning (ML) can accurately diagnose these faults, models trained on clean laboratory datasets frequently fail when deployed to real-world machines due to domain shifts (different operating speeds, unique machine resonance, background noise).

This project solves both problems:
1. **High-Precision Classification:** Achieving near-perfect fault isolation on the CWRU lab dataset using classical ML ensembles and Deep Learning.
2. **Cross-Machine Robustness:** Leveraging advanced Domain Adaptation (DA) techniques to successfully transfer learned representations to the highly noisy, real-world Paderborn University dataset, drastically reducing the need for labeled real-world data.

We abstract the ML complexity behind a **deployable API** and a **real-time alerting dashboard** to bring real value to maintenance engineers.

## ✨ Key Features & Differentiators

Unlike standard university ML projects, this platform implements production-tier methodologies:

* **Advanced Domain Adaptation:** Implements Domain-Adversarial Neural Networks (DANN), Maximum Mean Discrepancy (MMD), and Test-Time Adaptation to achieve robustness across different sensor environments.
* **Stacking Ensembles:** Combines strengths of classical algorithms (SVM, Random Forest) trained on handcrafted multi-domain features (Time, Frequency, Envelope) with raw-signal 1D-CNNs using a meta-learner.
* **Prognostic RUL Capabilities:** Moves beyond "what is broken?" to estimate the Remaining Useful Life (RUL) — telling engineers "when will it break?".
* **Explainable AI (XAI):** Demystifies the black box using SHAP values and 1D Grad-CAM, highlighting the exact frequencies or signal time-steps that trigger fault alerts.
* **Anomaly/Novelty Detection:** Utilizes One-Class SVMs and Autoencoders to raise flags when the machine encounters entirely unseen "novel" faults not present in the training data.
* **Real-time Edge Dashboard:** A Streamlit-based interface that ingests live vibration data streams, scores degradation continuously, and triggers maintenance Slack/Email alerts.

## 🗂️ Datasets Required

1. **Source Domain:** [Case Western Reserve University (CWRU) Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
   * High-quality, lab-controlled vibration signals.
2. **Target Domain:** [Paderborn University (PU) Bearing Dataset](https://github.com/Kat-Center/Paderborn-Bearing-Dataset)
   * Real-world noise profiles, artificial and real damages.

## 🏗️ Architecture

1. **Data Pipeline:** Standardized ingestion of `.mat` and `.csv` sensor signals, followed by segmentation, noise augmentation, and multi-domain feature extraction.
2. **Offline ML Pipeline:** Training models (SVM, RF-SVM, 1D-CNN, ResNet-1D), establishing domain adaptation losses, and saving best artifacts via MLflow.
3. **Online Inference (API):** A FastAPI microservice exposing `/predict`, `/stream`, and `/explain` endpoints.
4. **Presentation Layer:** Interactive Streamlit dashboard visualizing waveform, RUL countdown, health index, and model explainability charts.

*(For detailed system architecture, component breakdown, and technology choices, please refer to [SYSTEM_DESIGN.md](./SYSTEM_DESIGN.md))*

## 🚀 Getting Started

*(This section will be updated as the codebase is populated)*

### Prerequisites

* Python 3.10+
* Docker (Optional, for containerized fast-startup)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bearing-fault-diagnosis.git
   cd bearing-fault-diagnosis
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Preparation:**
   * Download the CWRU dataset into `data/raw/cwru/`
   * Download the Paderborn dataset into `data/raw/paderborn/`
   * *(Alternatively, run the future `src/data/download_datasets.py` script)*

### Execution (Coming Soon)

* **Run ML Training:** `python scripts/train_all.py`
* **Start API Server:** `uvicorn api.main:app --reload`
* **Launch UI Dashboard:** `streamlit run dashboard/app.py`

## 📊 Evaluation & Metrics

The system will report comprehensive metrics including:
* Macro-F1 and per-class precision/recall on the CWRU Source Test Set.
* **Cross-Domain Accuracy Gap** (Accuracy on PU *without* vs *with* domain adaptation).
* Confusion matrices, ROC curves, and Feature Space (t-SNE) projections comparing pre- and post-adaptation feature alignment.

## 💼 Future Applications

This methodology is not solely restricted to bearings. The anomaly detection and domain adaptation pipeline built here directly translates to monitoring:
- Gearboxes in Wind Turbines
- Compressors and Pumps in Oil & Gas
- Jet Engine Turbines

---
**License:** MIT License
