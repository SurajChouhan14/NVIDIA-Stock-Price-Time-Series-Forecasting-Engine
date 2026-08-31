# NVIDIA Stock Price Time-Series Forecasting Engine
> **Econometric Time-Series Forecasting Suite implementing Augmented Dickey-Fuller (ADF) Unit-Root Diagnostics, Box-Jenkins ARIMA(2,1,1) Modeling, and Holt's Linear Exponential Smoothing**  
> *Time-Series Econometrics · Augmented Dickey-Fuller · Box-Jenkins ARIMA · statsmodels · Holt Linear Smoothing*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SurajChouhan14/NVIDIA-Stock-Price-Time-Series-Forecasting-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajChouhan14/NVIDIA-Stock-Price-Time-Series-Forecasting-Engine/actions)
[![Observations](https://img.shields.io/badge/dataset-1%2C390%20Daily%20Bars-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)]()

---

## 🎯 Executive Overview & Econometric Architecture
Time-series forecasting of high-beta equity prices requires rigorous statistical diagnostics to establish stationarity before parametric modeling. This repository establishes an end-to-end econometric pipeline for NVIDIA (`NVDA`) equity:

### 1. Augmented Dickey-Fuller (ADF) Unit-Root Diagnostics
Tests the null hypothesis $H_0: \gamma = 0$ (unit root non-stationarity) against $H_1: \gamma < 0$:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \varepsilon_t$$
* **Raw Daily Prices:** Non-stationary ($t = +1.0816 > -2.86$, $p = 0.997 \implies$ fail to reject $H_0$).
* **Differenced Log Returns:** Strictly stationary ($t = -37.8420 < -2.86$, $p < 0.001 \implies$ reject $H_0$).

### 2. Box-Jenkins ARIMA(2,1,1) Specification
Formulated on log price series with order $(p=2, d=1, q=1)$:
$$(1 - \phi_1 B - \phi_2 B^2)(1 - B) \ln(y_t) = c + (1 + \theta_1 B) \varepsilon_t$$
Multi-step recursive forecasts are projected over a 60-day out-of-sample holdout and exponentiated back to price levels.

```
  ┌────────────────────────────────────────────────────────┐
  │ NVIDIA Historical Daily Bars (1,390 Observations)      │
  │ • Train Window: 1,330 Days | Test Window: 60 Days       │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Statistical Diagnostics: Augmented Dickey-Fuller (ADF) │
  │ • Raw Prices: Non-Stationary (t = +1.08, p = 0.997)     │
  │ • Log Differenced: Strictly Stationary (t = -37.84)     │
  └───────────────────────────┬────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
  ┌──────────────────────────┐              ┌──────────────────────────┐
  │ Box-Jenkins ARIMA(2,1,1) │              │ Holt Linear Smoothing    │
  │ (Log Transform + MLE)    │              │ (Trend State-Space)      │
  └─────────────┬────────────┘              └─────────────┬────────────┘
                │                                         │
                ▼                                         ▼
         • MAPE: 10.19%                            • MAPE: 11.55%
         • RMSE: $94.37                            • RMSE: $106.02
         • MAE: $85.56                             • MAE: $95.47
```

---

## 📊 Benchmark Execution & Validation Matrix

### 60-Day Out-of-Sample Holdout Window

| Model Architecture | Specification / Parameters | RMSE ($) | MAE ($) | MAPE (%) | Directional Acc (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **ARIMA (2, 1, 1)** | Log-scale recursive multi-step (MLE) | **$94.37** | **$85.56** | **10.19%** | **52.54%** |
| **Holt Linear Smoothing (Opt)** | Automated MLE parameter estimation | $106.02 | $95.47 | 11.55% | 50.85% |
| **Holt Linear Smoothing (Fixed)** | Fixed smoothing ($\alpha=0.30, \beta=0.10$) | $164.18 | $142.08 | 16.33% | 49.15% |

> **Methodological Disclosure:** Directional accuracy of ~52.54% over a 60-day recursive horizon is consistent with the weak-form Efficient Market Hypothesis (near-random-walk behavior of recursive point forecasts over extended horizons).

---

## 📁 Repository Structure

```text
NVIDIA-Stock-Price-Time-Series-Forecasting-Engine/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated CI test & benchmark workflow
├── .gitignore                          # Git exclusions (pycache, results, logs)
├── NVIDIA_Stock_Price_Time_Series_Forecasting.ipynb # Jupyter notebook
├── README.md                           # Documentation & econometric diagnostics
├── data/
│   └── nvda_stock_prices.csv           # 1,390 daily historical trading bars
├── requirements.txt                    # Dependencies (statsmodels, pandas, numpy, scipy)
├── run_pipeline.py                     # Pipeline execution runner
├── src/
│   ├── data_loader.py                  # Ingestion, feature prep & train/test splits
│   └── time_series_forecaster.py       # ADF diagnostics & ARIMA/Holt forecasters
└── test_nvidia_forecasting.py          # 5 automated unit & econometric invariant tests
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/NVIDIA-Stock-Price-Time-Series-Forecasting-Engine.git
cd NVIDIA-Stock-Price-Time-Series-Forecasting-Engine
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
python run_pipeline.py
```

### 3. Run Test Suite
```bash
python test_nvidia_forecasting.py
```
