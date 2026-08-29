# 📈 NVIDIA Stock Price Time-Series Forecasting Engine
### Augmented Dickey-Fuller (ADF) Unit-Root Tests | Box-Jenkins ARIMA(2,1,1) | Holt Smoothing | statsmodels

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Time Series](https://img.shields.io/badge/Econometrics-statsmodels%20ARIMA-success.svg)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An econometric and time-series forecasting engine implementing **Augmented Dickey-Fuller (ADF)** stationarity diagnostics and **Box-Jenkins ARIMA(2,1,1)** models on 5-year daily equity price series using `statsmodels`.

---

## 📌 Methodology & Econometric Formulations

### 1. Augmented Dickey-Fuller (ADF) Unit-Root Test:
$$\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^k \beta_i \Delta y_{t-i} + \epsilon_t$$
* Raw Price Series: $t = +1.08 > -2.86 \implies \text{Non-Stationary}$.
* **First-Differenced Log Return Series:** $\mathbf{t = -37.84 \; (p < 0.001)} \implies \mathbf{\text{Stationary (Reject Unit Root)}}$.

### 2. Box-Jenkins ARIMA(2,1,1) Multi-Step Forward Forecasting:
$$\Delta \ln(P_t) = \phi_1 \Delta \ln(P_{t-1}) + \phi_2 \Delta \ln(P_{t-2}) + \epsilon_t + \theta_1 \epsilon_{t-1}$$

---

## 📊 60-Day Forward Backtesting Performance
* **Historical Dataset:** 1,390 trading days (1,330 in-sample training, 60 out-of-sample forward test).
* **Evaluation Metrics:**
  * **ARIMA (2,1,1):** $\mathbf{\text{MAPE} = 10.19\%}$, $\text{RMSE} = \$94.37$, $\text{MAE} = \$85.56$, $\text{Directional Acc} = 52.54\%$.
  * **Holt Linear Smoothing:** $\text{MAPE} = 16.33\%$, $\text{RMSE} = \$164.18$.
  * **Outperformance:** ARIMA outperforms Holt linear smoothing by **6.14% MAPE margin**.

---

## 📂 Repository Structure
```
NVIDIA-Stock-Price-Time-Series-Forecasting-Engine/
├── src/
│   ├── time_series_forecaster.py   # statsmodels ADF, ARIMA & Holt forecaster
│   └── data_loader.py              # Equity historical price loader
├── NVIDIA_Stock_Price_Time_Series_Forecasting.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_nvidia_forecasting.py      # Unit testing suite (4/4 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/NVIDIA-Stock-Price-Time-Series-Forecasting-Engine.git
cd NVIDIA-Stock-Price-Time-Series-Forecasting-Engine
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_nvidia_forecasting.py
```
