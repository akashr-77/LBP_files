# 🔩 Intelligent Bearing Fault Diagnosis & Predictive Maintenance System

## System Design & Workflow Document

---

## 1. Project Vision

Build an **end-to-end, production-grade bearing fault diagnosis platform** that goes beyond research-level classification. The system will:

1. Train high-accuracy models on CWRU lab data
2. Achieve **cross-domain robustness** via transfer learning & domain adaptation to Paderborn real-world data
3. Provide a **real-time monitoring dashboard** with live signal processing
4. Generate **Remaining Useful Life (RUL) predictions** — moving from *classification* to *prognostics*
5. Deliver **explainable AI** insights so maintenance engineers trust the model
6. Package everything as a **deployable, containerised microservice**

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                         │
│                                                                           │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐  │
│  │  CWRU     │   │  Paderborn   │   │  Simulated /  │   │  Live Sensor │  │
│  │  Dataset  │   │  Dataset     │   │  Augmented    │   │  Stream      │  │
│  └────┬─────┘   └──────┬───────┘   └──────┬────────┘   └──────┬───────┘  │
│       └────────────┬────┴──────────────────┘                   │          │
│                    ▼                                           │          │
│          ┌─────────────────┐               ┌──────────────────┐│          │
│          │ Offline Pipeline│               │ Online Pipeline  ││          │
│          └────────┬────────┘               └────────┬─────────┘│          │
└───────────────────┼─────────────────────────────────┼──────────┘          │
                    ▼                                 ▼                     │
┌─────────────────────────────────────────────────────────────────┐         │
│                   SIGNAL PROCESSING LAYER                       │         │
│                                                                 │         │
│  ┌────────────┐  ┌────────────────┐  ┌────────────────────────┐ │         │
│  │ Windowing  │  │ Noise Injection│  │ Multi-Domain Feature   │ │         │
│  │ & Overlap  │  │ & Augmentation │  │ Extraction             │ │         │
│  └─────┬──────┘  └───────┬────────┘  │ • Time-domain stats    │ │         │
│        │                 │           │ • Frequency (FFT, PSD)  │ │         │
│        │                 │           │ • Time-Freq (STFT, CWT) │ │         │
│        │                 │           │ • Envelope Analysis     │ │         │
│        │                 │           └───────────┬─────────────┘ │         │
└────────┼─────────────────┼───────────────────────┼──────────────┘         │
         └─────────────────┴───────────────────────┘                        │
                              │                                             │
                              ▼                                             │
┌─────────────────────────────────────────────────────────────────────────┐  │
│                      MODEL LAYER                                        │  │
│                                                                         │  │
│  ┌─────────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │  │
│  │ Classical ML Branch │  │ Deep Learning    │  │ Ensemble / Meta   │   │  │
│  │  • SVM              │  │  • 1D-CNN        │  │  • Stacking       │   │  │
│  │  • RF               │  │  • ResNet-1D     │  │  • Voting         │   │  │
│  │  • RF-SVM Hybrid    │  │  • LSTM / GRU    │  │  • Confidence     │   │  │
│  │  • XGBoost          │  │  • CNN-LSTM      │  │    Weighted Avg   │   │  │
│  └──────────┬──────────┘  └────────┬─────────┘  └────────┬──────────┘   │  │
│             │                      │                     │              │  │
│             ▼                      ▼                     │              │  │
│  ┌────────────────────────────────────────────────────┐   │              │  │
│  │         DOMAIN ADAPTATION MODULE                   │   │              │  │
│  │  • Fine-tuning on Paderborn                        │◄──┘              │  │
│  │  • MMD / CORAL loss alignment                      │                  │  │
│  │  • Adversarial Domain Adaptation (DANN)            │                  │  │
│  │  • Test-Time Adaptation (TTA)                      │                  │  │
│  └───────────────────────┬────────────────────────────┘                  │  │
└──────────────────────────┼──────────────────────────────────────────────┘  │
                           ▼                                                │
┌─────────────────────────────────────────────────────────────────────────┐  │
│                   INTELLIGENCE LAYER                                    │  │
│                                                                         │  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────┐ │  │
│  │ Fault Classifier  │  │ RUL Estimator     │  │ Anomaly Detector     │ │  │
│  │ (multi-class)     │  │ (regression /     │  │ (one-class SVM /     │ │  │
│  │                   │  │  survival model)  │  │  autoencoder)        │ │  │
│  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────────┘ │  │
│           └──────────────────────┼───────────────────────┘               │  │
│                                  ▼                                       │  │
│                   ┌──────────────────────────┐                           │  │
│                   │ Explainability Engine     │                           │  │
│                   │ • SHAP / LIME            │                           │  │
│                   │ • Grad-CAM (for 1D-CNN)  │                           │  │
│                   │ • Feature importance      │                           │  │
│                   └──────────┬───────────────┘                           │  │
└──────────────────────────────┼───────────────────────────────────────────┘  │
                               ▼                                             │
┌─────────────────────────────────────────────────────────────────────────────┘
│                    APPLICATION LAYER
│
│  ┌───────────────────────────────────────────────────────────────────────┐
│  │                 Real-Time Dashboard (Streamlit / Plotly Dash)         │
│  │                                                                       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  │ Live Signal │  │ Fault Class │  │ RUL      │  │ SHAP / Feature │  │
│  │  │ Waveform    │  │ & Confidence│  │ Countdown│  │ Explanation    │  │
│  │  └─────────────┘  └─────────────┘  └──────────┘  └────────────────┘  │
│  │                                                                       │
│  │  ┌──────────────────┐  ┌──────────────────────────────────────────┐   │
│  │  │ Alert System     │  │ Maintenance Recommendation Engine        │   │
│  │  │ (Email/SMS/Slack)│  │ (Rule-based + AI severity scoring)       │   │
│  │  └──────────────────┘  └──────────────────────────────────────────┘   │
│  └───────────────────────────────────────────────────────────────────────┘
│
│  ┌───────────────────────────────────────────────────────────────────────┐
│  │                  REST API (FastAPI)                                    │
│  │   POST /predict   POST /stream   GET /health   GET /model-info       │
│  └───────────────────────────────────────────────────────────────────────┘
│
│  ┌───────────────────────────────────────────────────────────────────────┐
│  │        MLOps: MLflow Tracking │ Docker │ Model Registry               │
│  └───────────────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────────

```

---

## 3. What Makes This Project Impressive — Key Differentiators

### 3.1 🧠 Advanced Domain Adaptation (Beyond Simple Transfer Learning)

| Technique | Description | Why It's Impressive |
|---|---|---|
| **Fine-Tuning** | Freeze early CNN layers, retrain final layers on Paderborn data | Baseline transfer approach |
| **MMD Loss (Maximum Mean Discrepancy)** | Add a domain alignment loss that minimises distribution divergence between CWRU and Paderborn feature spaces | Statistically rigorous cross-domain alignment |
| **CORAL (Correlation Alignment)** | Align second-order statistics (covariance) of source and target domains | Simple yet effective, publishable |
| **DANN (Domain-Adversarial Neural Network)** | Add a gradient reversal layer and domain classifier so the feature extractor learns domain-invariant representations | State-of-the-art; directly addresses the core problem |
| **Test-Time Adaptation (TTA)** | Adapt batch-norm statistics at inference time using incoming test data | Zero-label adaptation; works on completely unseen machines |

> **Recommendation**: Implement at least **DANN + Fine-Tuning + TTA** for a comprehensive adaptation study. Compare all methods in a single results table.

---

### 3.2 📊 Multi-Domain Feature Engineering

Extract features from **four signal domains** to give ML models the richest possible input:

#### Time Domain (14 features)
- Mean, Standard Deviation, RMS, Peak-to-Peak
- Skewness, Kurtosis, Crest Factor, Shape Factor
- Impulse Factor, Margin Factor, Clearance Factor
- Zero-Crossing Rate, Entropy, Autocorrelation peaks

#### Frequency Domain (8+ features)
- FFT magnitude spectrum → spectral centroid, bandwidth, rolloff
- Power Spectral Density (PSD) → dominant frequencies, spectral energy bands
- Bearing characteristic frequencies (BPFO, BPFI, BSF, FTF) energy ratios

#### Time-Frequency Domain (advanced)
- **Short-Time Fourier Transform (STFT)** → spectrogram-based features
- **Continuous Wavelet Transform (CWT)** → scalogram images (can be fed as 2D to CNN!)
- **Empirical Mode Decomposition (EMD)** → Intrinsic Mode Functions for non-stationary analysis

#### Envelope Analysis
- Hilbert transform → envelope spectrum
- Extract bearing defect frequencies from demodulated signal

---

### 3.3 🏗️ Model Ensemble — Stacking Architecture

Instead of picking the single best model, build a **stacking ensemble** that combines all model families:

```
                    ┌──────────────────────────────────────┐
                    │         META-LEARNER (Layer 2)       │
                    │    Logistic Regression / XGBoost     │
                    └─────────┬────────┬───────┬───────────┘
                              │        │       │
                 ┌────────────┘        │       └────────────┐
                 ▼                     ▼                    ▼
         ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
         │   SVM         │    │   RF-SVM      │    │   1D-CNN      │
         │ (handcrafted  │    │ (handcrafted  │    │ (raw signal)  │
         │  features)    │    │  features)    │    │               │
         └───────────────┘    └───────────────┘    └───────────────┘
                   Layer 1 Base Learners
```

- Use **5-fold cross-validated predictions** as meta-features (no data leakage)
- The meta-learner combines diverse model perspectives → higher accuracy, better calibrated confidence
- Report both individual model results AND ensemble results

---

### 3.4 🔍 Explainable AI (XAI) — Build Trust

Maintenance engineers won't trust a black-box. Add explainability:

| Tool | Applies To | What It Shows |
|---|---|---|
| **SHAP (SHapley Additive exPlanations)** | SVM, RF, XGBoost | Which features contributed most to this specific prediction |
| **LIME** | Any model | Local interpretable approximation of the model's decision |
| **Grad-CAM (1D adaptation)** | 1D-CNN | Which time-steps in the raw vibration signal activated the fault decision |
| **Feature Importance** | RF, XGBoost | Global ranking of which features matter most overall |
| **t-SNE / UMAP Visualization** | All | 2D projection of learned feature space showing class clusters |

> Display SHAP waterfall plots and Grad-CAM heatmaps in the dashboard alongside each prediction.

---

### 3.5 ⏳ Remaining Useful Life (RUL) Estimation — Predictive Maintenance

Go **beyond classification** (what fault?) to **prognostics** (when will it fail?):

- **Approach 1 — Degradation Trend Modelling**: Track how features (e.g., RMS, Kurtosis) evolve over time → fit a degradation curve → extrapolate to failure threshold
- **Approach 2 — Survival Analysis**: Use Cox Proportional Hazards or a Weibull model with extracted features as covariates to estimate probability of failure over time
- **Approach 3 — Deep Learning**: LSTM/GRU trained on sequential feature windows to predict cycles-to-failure

> Even a simple degradation trend model with a health index adds massive real-world value.

---

### 3.6 🚨 Anomaly Detection — Unseen Fault Types

Real machinery can develop faults never seen during training. Add an **anomaly/novelty detection** layer:

- **One-Class SVM** trained only on "Normal" class → flags anything outside the normal envelope
- **Autoencoder** trained on normal signals → high reconstruction error = anomaly
- **Confidence Thresholding** → if the classifier's max softmax probability < threshold, flag as "Unknown Fault"

This makes the system **open-set** — it doesn't blindly misclassify unseen faults.

---

### 3.7 📈 Data Augmentation for Robustness

Augment training data to simulate real-world noise variability:

| Technique | Description |
|---|---|
| **Gaussian Noise Injection** | Add random noise at varying SNR levels |
| **Time Shifting** | Shift the signal window randomly |
| **Signal Scaling** | Randomly scale amplitude ±20% |
| **Speed Perturbation** | Resample signal to simulate RPM variation |
| **Mixup** | Linearly interpolate between two samples and their labels |
| **GAN-based Synthesis** | Train a WGAN-GP to generate synthetic vibration signals for minority classes |

---

## 4. Detailed Module-Level Workflow

### Phase 1: Data Preparation & Feature Engineering

```
Step 1.1 — Load & Parse Datasets
├── CWRU: Parse .mat files (DE, FE, BA channels)
│   ├── 10 classes: Normal + 3 fault types × 3 severity levels
│   └── Sampling rate: 12 kHz (fan end), 48 kHz (drive end)
├── Paderborn: Download and parse vibration CSV files
│   ├── Healthy, Inner Race, Outer Race damage types
│   └── Artificial + Real damage conditions
└── Standardise both datasets to uniform format:
    → DataFrame with columns: [signal_array, fault_type, severity, dataset_source, rpm]

Step 1.2 — Windowing & Segmentation
├── Window size: 2048 samples (configurable)
├── Overlap: 50% (configurable)
├── Label inheritance: each window inherits parent signal's label
└── Store as: segmented_cwru.pkl, segmented_paderborn.pkl

Step 1.3 — Feature Extraction Pipeline
├── Time-domain: 14 statistical features per window
├── Frequency-domain: FFT → 8 spectral features per window
├── Time-frequency: CWT scalograms (saved as images for 2D CNN variant)
├── Envelope: Hilbert transform → 4 envelope features
└── Output: feature_matrix.csv (N_windows × 26+ features)

Step 1.4 — Data Augmentation
├── Apply noise injection, time shifting, scaling to CWRU data
├── Generate synthetic minority samples via SMOTE / GAN
└── Output: augmented_feature_matrix.csv
```

### Phase 2: Model Training (Source Domain — CWRU)

```
Step 2.1 — Classical ML Pipeline
├── Preprocessing: StandardScaler + PCA (optional dimensionality reduction)
├── Train SVM (RBF kernel) with GridSearchCV
├── Train Random Forest + extract top features → train SVM on selected features (RF-SVM)
├── Train XGBoost (additional strong baseline)
├── Evaluate: 5-fold stratified CV, confusion matrix, per-class F1
└── Save models: svm_model.pkl, rf_svm_model.pkl, xgboost_model.pkl

Step 2.2 — Deep Learning Pipeline
├── 1D-CNN Architecture:
│   ├── Input: (2048, 1) raw vibration signal
│   ├── Conv1D blocks: [64, 128, 256] filters, kernel=7, BatchNorm, ReLU, MaxPool
│   ├── Global Average Pooling → Dense(128) → Dropout(0.3) → Dense(num_classes)
│   └── Output: Softmax probabilities
├── ResNet-1D variant (residual connections for deeper networks)
├── CNN-LSTM Hybrid: CNN feature extractor → LSTM sequence modelling
├── Training: Adam, lr=1e-3, cosine annealing, early stopping
├── Evaluate: same metrics + Grad-CAM visualisations
└── Save: best_cnn.keras, best_resnet.keras

Step 2.3 — Ensemble (Stacking)
├── Generate 5-fold OOF (out-of-fold) predictions from each base model
├── Stack OOF predictions as features → train meta-learner (Logistic Regression)
├── Evaluate ensemble on held-out CWRU test set
└── Save: ensemble_meta_model.pkl
```

### Phase 3: Domain Adaptation (CWRU → Paderborn)

```
Step 3.1 — Direct Transfer (Baseline)
├── Evaluate CWRU-trained models directly on Paderborn test set
└── Record accuracy drop → this is the "domain gap"

Step 3.2 — Fine-Tuning
├── Freeze CNN layers 1-4, unfreeze final dense layers
├── Fine-tune on 10-20% labeled Paderborn data
└── Evaluate improvement over direct transfer

Step 3.3 — MMD / CORAL Alignment
├── Add MMD loss term to CNN training objective
│   └── MMD(source_features, target_features) → minimise distribution gap
├── Alternative: CORAL loss (align covariance matrices)
└── Retrain CNN with combined classification + alignment loss

Step 3.4 — DANN (Domain-Adversarial)
├── Architecture:
│   ├── Shared Feature Extractor (CNN backbone) ──┬──→ Fault Classifier
│   │                                              └──→ Domain Classifier
│   └── Gradient Reversal Layer between extractor and domain classifier
├── Training: adversarial min-max game
│   ├── Classifier minimises fault classification loss
│   └── Feature extractor maximises domain classifier loss (via GRL)
├── Result: features become domain-invariant
└── Evaluate on Paderborn test set

Step 3.5 — Test-Time Adaptation
├── At inference, update BatchNorm running stats using incoming batches
├── No labels needed — purely unsupervised adaptation
└── Combine with DANN for best results

Step 3.6 — Comparison Table
├── Create comprehensive table: Method × Accuracy × F1 × Adaptation Data Required
└── Statistical significance tests (McNemar's test or paired t-test on folds)
```

### Phase 4: Anomaly Detection Module

```
Step 4.1 — Train One-Class SVM on normal samples only
Step 4.2 — Train Autoencoder on normal vibration windows
├── Encoder: Conv1D layers → bottleneck (dim=16)
├── Decoder: Transposed Conv1D layers → reconstruct input
├── Anomaly score = reconstruction error (MSE)
└── Threshold: mean + 3σ of normal reconstruction errors

Step 4.3 — Confidence-based rejection
├── If softmax max probability < 0.7 → flag as "Uncertain / Potential Novel Fault"
└── Combine with anomaly detector output for final decision
```

### Phase 5: Explainability Engine

```
Step 5.1 — SHAP Analysis
├── Compute SHAP values for SVM/RF/XGBoost on test set
├── Generate: summary plot, waterfall plot (per-sample), dependence plots
└── Save as interactive HTML

Step 5.2 — Grad-CAM for 1D-CNN
├── Compute gradient of predicted class w.r.t. last conv layer activations
├── Overlay importance heatmap on raw vibration signal
└── Identify which signal regions trigger fault detection

Step 5.3 — Feature Space Visualisation
├── Extract penultimate layer features from CNN
├── Apply t-SNE / UMAP → 2D scatter plot coloured by fault class
├── Compare CWRU vs Paderborn clusters (before/after adaptation)
└── This visually demonstrates domain adaptation effectiveness
```

### Phase 6: Real-Time Dashboard & API

```
Step 6.1 — FastAPI Backend
├── POST /predict
│   ├── Input: raw vibration signal (JSON array or .wav/.csv upload)
│   ├── Pipeline: window → extract features → run ensemble → explain
│   └── Output: {fault_class, confidence, top_features, rul_estimate, anomaly_flag}
├── POST /predict/batch — batch prediction for uploaded files
├── GET /health — service health check
├── GET /model-info — current model version, training metrics
└── WebSocket /stream — streaming predictions for live sensor data

Step 6.2 — Streamlit Dashboard
├── Page 1: Live Monitor
│   ├── Real-time vibration waveform (animated line chart)
│   ├── Fault class indicator (colour-coded: green/yellow/red)
│   ├── Confidence gauge (0-100%)
│   ├── RUL countdown bar
│   └── Anomaly alert banner
├── Page 2: Diagnostics
│   ├── SHAP waterfall for current prediction
│   ├── Grad-CAM overlay on signal
│   ├── Feature values vs. normal baseline comparison
│   └── Historical trend of health index
├── Page 3: Upload & Analyse
│   ├── Upload .mat / .csv vibration files
│   ├── Run full analysis pipeline
│   └── Download PDF report
├── Page 4: Model Comparison
│   ├── Confusion matrices side-by-side
│   ├── ROC/PR curves per class
│   ├── Domain adaptation results table
│   └── t-SNE / UMAP feature space plots
└── Page 5: Settings & Alerts
    ├── Configure alert thresholds
    ├── Email / SMS / Slack notification setup
    └── Model selection (swap between trained models)

Step 6.3 — Alert & Recommendation Engine
├── Alert Rules:
│   ├── Confidence > 80% AND fault class ≠ Normal → ⚠️ WARNING
│   ├── Anomaly score > threshold → 🔴 CRITICAL ALERT
│   └── RUL < 500 cycles → 🟡 Schedule Maintenance
├── Maintenance Recommendations (rule-based + severity):
│   ├── Inner Race fault → "Inspect inner raceway; replace bearing if pitting visible"
│   ├── Outer Race fault → "Check housing alignment; inspect outer raceway"
│   ├── Ball fault → "Inspect rolling elements; check lubricant contamination"
│   └── Severity-dependent: minor → "Monitor closely" / severe → "Immediate replacement"
└── Notification dispatch: email via SMTP, Slack webhook, browser push
```

### Phase 7: MLOps & Reproducibility

```
Step 7.1 — Experiment Tracking with MLflow
├── Log all hyperparameters, metrics, and model artifacts
├── Compare runs visually in MLflow UI
└── Model registry: promote best model to "Production" stage

Step 7.2 — Docker Containerisation
├── Dockerfile for FastAPI + model serving
├── docker-compose.yml: API + Dashboard + MLflow
└── Single command deployment: docker-compose up

Step 7.3 — CI/CD Pipeline (optional GitHub Actions)
├── On push: run unit tests → lint → retrain if data changes → deploy
└── Model performance monitoring: alert if accuracy drops below threshold
```

---

## 5. Project Directory Structure

```
LBP/
├── README.md                          # Project overview, setup instructions
├── SYSTEM_DESIGN.md                   # This document
├── requirements.txt                   # All dependencies
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Multi-service orchestration
│
├── config/
│   ├── config.yaml                    # All hyperparameters, paths, thresholds
│   └── logging.yaml                   # Logging configuration
│
├── data/
│   ├── raw/
│   │   ├── cwru/                      # Original .mat files
│   │   └── paderborn/                 # Paderborn vibration files
│   ├── processed/
│   │   ├── segmented_cwru.pkl
│   │   ├── segmented_paderborn.pkl
│   │   ├── features_cwru.csv
│   │   └── features_paderborn.csv
│   └── augmented/
│       └── augmented_features.csv
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py                  # Dataset loading (CWRU + Paderborn)
│   │   ├── windowing.py               # Signal segmentation
│   │   ├── augmentation.py            # Data augmentation techniques
│   │   └── feature_extraction.py      # Multi-domain feature extraction
│   │
│   ├── models/
│   │   ├── svm_model.py               # SVM training & evaluation
│   │   ├── rf_svm_model.py            # RF-SVM hybrid
│   │   ├── xgboost_model.py           # XGBoost baseline
│   │   ├── cnn_1d.py                  # 1D-CNN architecture
│   │   ├── resnet_1d.py               # ResNet-1D architecture
│   │   ├── cnn_lstm.py                # CNN-LSTM hybrid
│   │   ├── ensemble.py                # Stacking ensemble
│   │   └── anomaly_detector.py        # One-Class SVM + Autoencoder
│   │
│   ├── adaptation/
│   │   ├── fine_tuning.py             # Layer-wise fine-tuning
│   │   ├── mmd_coral.py               # MMD and CORAL loss functions
│   │   ├── dann.py                    # Domain-Adversarial Neural Network
│   │   └── tta.py                     # Test-Time Adaptation
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py          # SHAP analysis for ML models
│   │   ├── gradcam_1d.py              # Grad-CAM adapted for 1D signals
│   │   ├── feature_importance.py      # Global feature rankings
│   │   └── visualisation.py           # t-SNE, UMAP projections
│   │
│   ├── prognostics/
│   │   ├── health_index.py            # Composite health indicator
│   │   ├── degradation_model.py       # Degradation trend fitting
│   │   └── rul_estimator.py           # RUL prediction models
│   │
│   └── utils/
│       ├── metrics.py                 # Custom metrics (per-class F1, etc.)
│       ├── plotting.py                # Standardised plot functions
│       └── config_parser.py           # YAML config loader
│
├── api/
│   ├── main.py                        # FastAPI application
│   ├── schemas.py                     # Pydantic request/response models
│   ├── inference.py                   # Model loading & prediction pipeline
│   └── websocket_stream.py            # Live streaming predictions
│
├── dashboard/
│   ├── app.py                         # Streamlit main app
│   ├── pages/
│   │   ├── live_monitor.py
│   │   ├── diagnostics.py
│   │   ├── upload_analyse.py
│   │   ├── model_comparison.py
│   │   └── settings.py
│   └── components/
│       ├── signal_plot.py
│       ├── gauge.py
│       └── alert_banner.py
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb    # Feature extraction deep-dive
│   ├── 03_classical_ml.ipynb          # SVM, RF-SVM, XGBoost experiments
│   ├── 04_deep_learning.ipynb         # CNN, ResNet, CNN-LSTM training
│   ├── 05_domain_adaptation.ipynb     # Transfer learning experiments
│   ├── 06_ensemble.ipynb              # Stacking ensemble construction
│   ├── 07_explainability.ipynb        # SHAP, Grad-CAM analysis
│   └── 08_anomaly_detection.ipynb     # Novelty detection experiments
│
├── tests/
│   ├── test_feature_extraction.py
│   ├── test_models.py
│   ├── test_adaptation.py
│   └── test_api.py
│
├── results/
│   ├── models/                        # Saved model artifacts
│   ├── figures/                       # Generated plots and visualisations
│   ├── reports/                       # Auto-generated PDF reports
│   └── mlflow/                        # MLflow tracking data
│
└── scripts/
    ├── train_all.py                   # Master training script
    ├── evaluate_cross_domain.py       # Cross-domain evaluation
    ├── generate_report.py             # Auto-generate results PDF
    └── download_paderborn.py          # Download Paderborn dataset
```

---

## 6. Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core language |
| **Data** | NumPy, Pandas, SciPy | Signal processing, data manipulation |
| **Signal Processing** | PyWavelets, librosa | CWT, STFT, envelope analysis |
| **ML** | scikit-learn, XGBoost | SVM, RF, RF-SVM, Stacking |
| **Deep Learning** | TensorFlow / Keras | 1D-CNN, ResNet, LSTM, Autoencoders |
| **Domain Adaptation** | Custom (TF/Keras) | DANN, MMD, CORAL, TTA |
| **Explainability** | SHAP, LIME, tf-keras-vis | Model interpretability |
| **Visualisation** | Matplotlib, Seaborn, Plotly | Plots and interactive charts |
| **Dimensionality Reduction** | UMAP, t-SNE (scikit-learn) | Feature space visualisation |
| **Dashboard** | Streamlit | Interactive web dashboard |
| **API** | FastAPI, Uvicorn | REST API + WebSocket |
| **Experiment Tracking** | MLflow | Hyperparameter & metric logging |
| **Containerisation** | Docker, docker-compose | Deployment |
| **Report** | FPDF / ReportLab | Auto-generated PDF reports |

---

## 7. Evaluation Framework

### 7.1 Metrics

| Metric | Purpose |
|---|---|
| **Overall Accuracy** | High-level performance |
| **Per-Class Precision, Recall, F1** | Identifies weak fault classes |
| **Macro-F1** | Balanced performance across classes |
| **Cohen's Kappa** | Agreement beyond chance |
| **Confusion Matrix** | Detailed misclassification patterns |
| **ROC-AUC (One-vs-Rest)** | Discriminative ability per class |
| **Cross-Domain Accuracy Gap** | Measures domain shift impact |
| **Adaptation Improvement (%)** | Quantifies transfer learning benefit |

### 7.2 Experimental Comparisons

| Experiment | What It Demonstrates |
|---|---|
| CWRU → CWRU (same domain) | Baseline upper bound |
| CWRU → Paderborn (no adaptation) | Domain gap quantification |
| CWRU → Paderborn (fine-tuned) | Benefit of labelled target data |
| CWRU → Paderborn (DANN) | Unsupervised domain adaptation |
| CWRU → Paderborn (DANN + TTA) | Best-case adaptation |
| Individual models vs Ensemble | Value of model combination |
| With vs Without augmentation | Robustness improvement |

---

## 8. Implementation Timeline (Suggested)

| Week | Phase | Key Deliverables |
|---|---|---|
| **1** | Data & EDA | Load both datasets, EDA notebook, windowing pipeline |
| **2** | Feature Engineering | Multi-domain extraction pipeline, augmentation module |
| **3** | Classical ML | SVM, RF-SVM, XGBoost trained & evaluated |
| **4** | Deep Learning | 1D-CNN, ResNet-1D, CNN-LSTM trained on CWRU |
| **5** | Domain Adaptation | Fine-tuning, MMD, DANN, TTA implemented & compared |
| **6** | Ensemble & Anomaly | Stacking ensemble, anomaly detection module |
| **7** | Explainability & RUL | SHAP, Grad-CAM, health index, RUL estimation |
| **8** | Dashboard & API | Streamlit dashboard, FastAPI backend, alert system |
| **9** | Integration & Docker | End-to-end pipeline, containerisation, MLflow |
| **10** | Documentation & Polish | Report generation, README, demo video, final testing |

---

## 9. Potential Future Extensions

- **Edge Deployment**: Convert model to TensorFlow Lite / ONNX for running on IoT edge devices (Raspberry Pi, NVIDIA Jetson)
- **Multi-Sensor Fusion**: Incorporate acoustic emission, temperature, and current signals alongside vibration
- **Federated Learning**: Train models across multiple factory sites without sharing raw data (privacy-preserving)
- **Physics-Informed ML**: Inject bearing physics equations (defect frequency relationships) as constraints into the neural network
- **Digital Twin**: Create a digital twin of the bearing system for simulation and what-if analysis
- **Continuous Learning**: Auto-retrain models as new labelled data arrives from the field

---

## 10. Key Research References

1. **CWRU Bearing Dataset** — Case Western Reserve University Bearing Data Center
2. **Paderborn Bearing Dataset** — KAt Data Center, Paderborn University
3. Ganin et al., "Domain-Adversarial Training of Neural Networks" (DANN), JMLR 2016
4. Long et al., "Learning Transferable Features with Deep Adaptation Networks" (MMD), ICML 2015
5. Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
6. Wang et al., "Batch Normalization Test-Time Adaptation," NeurIPS 2020
7. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017

---

> **Summary**: This design transforms a standard ML classification project into a **comprehensive, production-grade predictive maintenance platform**. The combination of domain adaptation (DANN), ensemble learning, anomaly detection, explainability (SHAP/Grad-CAM), RUL estimation, and a real-time dashboard makes this project stand out as both academically rigorous and industrially applicable.
