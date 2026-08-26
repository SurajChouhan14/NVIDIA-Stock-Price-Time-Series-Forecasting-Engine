# NVIDIA (NVDA) Stock Price Time-Series Forecasting Engine

An econometric and quantitative time-series forecasting system implementing **Augmented Dickey-Fuller (ADF) Unit-Root Diagnostics**, **ARIMA$(p,d,q)$ Modeling**, and **Holt's Linear Exponential Smoothing** on 5 years of daily trading data for **NVIDIA Corporation (`NVDA`)**.

---

## 1. System Architecture

```
                                 +-------------------------------------+
                                 | NVIDIA Daily Historical Stock Data  |
                                 | (1,390 Daily OHLCV Observations)    |
                                 +------------------+------------------+
                                                    |
                                                    v
                                 +-------------------------------------+
                                 | Statistical Diagnostics             |
                                 | • Trend & Seasonality Decomposition |
                                 | • Augmented Dickey-Fuller (ADF) Test|
                                 | • Stationarity Differencing (d=1)   |
                                 +------------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
              +--------------------+                                +--------------------+
              | ARIMA (2, 1, 1)    |                                | Holt's Linear      |
              | Autoregressive Model                                | Exponential Smooth |
              +----------+---------+                                +---------+----------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                 +-------------------------------------+
                                 | Out-of-Sample Backtesting (60 Days) |
                                 | (RMSE, MAE, MAPE, Directional Acc)  |
                                 +-------------------------------------+
```

---

## 2. Mathematical Formulation

### **1. Augmented Dickey-Fuller (ADF) Unit-Root Test**:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$
* Null Hypothesis $H_0: \gamma = 0$ (Unit root / Non-stationary).
* Differencing order $d=1$ establishes strict stationarity with $t = -37.84$ ($p < 0.001$).

### **2. $\text{ARIMA}(p, d, q)$ Model**:
$$\left(1 - \sum_{i=1}^{p} \phi_i L^i\right) (1 - L)^d y_t = c + \left(1 + \sum_{j=1}^{q} \theta_j L^j\right) \epsilon_t$$

### **3. Holt's Linear Exponential Smoothing**:
$$\ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1}$$
$$\hat{y}_{t+h} = \ell_t + h \cdot b_t$$

---

## 3. Exact Computed Benchmark Results (60-Day Out-of-Sample Horizon)

```
===============================================================================================
NVIDIA TIME-SERIES FORECASTING PERFORMANCE MATRIX
===============================================================================================
Forecasting Model                   | RMSE ($)     | MAE ($)      | MAPE (%)     | Directional Acc
-----------------------------------------------------------------------------------------------
ARIMA (2, 1, 1) Multi-Step          | $94.37       | $85.56       | 10.19 %      | 52.54 %
Holt Linear Exponential Smoothing   | $164.18      | $142.08      | 16.33 %      | 49.15 %
===============================================================================================
```

---

## 4. Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run time-series forecasting pipeline
python run_pipeline.py

# 3. Run automated unit tests
python test_nvidia_forecasting.py
```

---

## 5. Master Placement Resume Description

> **NVIDIA Stock Price Time-Series Forecasting Engine (ARIMA / Holt)**
> * Developed an econometric time-series forecasting suite on 1,390 daily trading observations of NVIDIA (`NVDA`) equity.
> * Formulated Augmented Dickey-Fuller (ADF) unit-root stationarity tests ($t = -37.84, p < 0.001$) confirming first-order differencing ($d=1$).
> * Benchmarked multi-step ahead ARIMA$(2,1,1)$ against Holt's Linear Exponential Smoothing, achieving a **10.19% out-of-sample MAPE** over a 60-day testing horizon.

---

## License
MIT License. Open for academic research and portfolio demonstration.
