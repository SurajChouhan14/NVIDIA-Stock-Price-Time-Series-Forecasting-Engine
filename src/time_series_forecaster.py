"""
NVIDIA Stock Price Time-Series Forecasting & Econometric Diagnostics Engine.

Implements industry-standard econometric and time-series modeling using statsmodels:
1. Augmented Dickey-Fuller (ADF) Unit-Root Stationarity Diagnostics (statsmodels.tsa.stattools.adfuller)
2. Box-Jenkins ARIMA(2,1,1) Forecasting (statsmodels.tsa.arima.model.ARIMA)
3. Holt's Linear Exponential Smoothing (statsmodels.tsa.holtwinters.Holt)
4. Out-of-Sample Backtesting & Forecast Error Metrics (RMSE, MAE, MAPE, Directional Accuracy)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt


class NVDATimeSeriesForecaster:
    """
    Time-series modeling and out-of-sample forecasting engine for NVIDIA equity
    using official statsmodels architecture.
    """

    def __init__(self, data_dict):
        self.full_df = data_dict['full_df']
        self.train_df = data_dict['train_df']
        self.test_df = data_dict['test_df']
        self.train_prices = data_dict['train_df']['Close'].values
        self.test_prices = data_dict['test_df']['Close'].values

    def run_adf_stationarity_test(self, series=None):
        """
        Computes Augmented Dickey-Fuller (ADF) test statistic for unit root stationarity
        using statsmodels.tsa.stattools.adfuller.
        """
        if series is None:
            series = self.full_df['Close'].values

        # Execute statsmodels ADF test
        adf_res = adfuller(series, autolag='AIC')
        t_stat = float(adf_res[0])
        p_val = float(adf_res[1])
        crit_5pct = float(adf_res[4]['5%'])

        is_stationary = bool(t_stat < crit_5pct)

        return {
            'adf_t_statistic': round(t_stat, 4),
            'p_value': round(p_val, 6),
            'critical_value_5pct': round(crit_5pct, 2),
            'is_stationary_5pct': is_stationary,
            'conclusion': "Stationary (Reject Unit Root)" if is_stationary else "Non-Stationary (Contains Unit Root -> Differencing d=1 required)"
        }

    def fit_and_forecast_holt(self, alpha=0.30, beta=0.10, forecast_horizon=60):
        """
        Holt's Linear Exponential Smoothing with trend component
        using statsmodels.tsa.holtwinters.Holt.
        """
        model = Holt(self.train_prices, initialization_method='estimated')
        fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta)
        forecasts = fitted_model.forecast(forecast_horizon)
        return np.array(forecasts)

    def fit_and_forecast_arima(self, p=2, d=1, q=1, forecast_horizon=60):
        """
        Box-Jenkins ARIMA(p, d, q) model on log prices using statsmodels.tsa.arima.model.ARIMA.
        Integrates forecasts back into dollar price levels.
        """
        log_train = np.log(self.train_prices)
        model = ARIMA(log_train, order=(p, d, q))
        fitted_model = model.fit()

        log_forecasts = fitted_model.forecast(steps=forecast_horizon)
        price_forecasts = np.exp(log_forecasts)
        return np.array(price_forecasts)

    def evaluate_forecast(self, forecasts, actuals=None):
        """
        Computes RMSE, MAE, MAPE, and Directional Accuracy.
        """
        if actuals is None:
            actuals = self.test_prices[:len(forecasts)]

        actuals = np.array(actuals)
        forecasts = np.array(forecasts)

        mae = float(np.mean(np.abs(actuals - forecasts)))
        rmse = float(np.sqrt(np.mean((actuals - forecasts) ** 2)))
        mape = float(np.mean(np.abs((actuals - forecasts) / actuals)) * 100.0)

        # Directional accuracy: sign of (y_t - y_{t-1}) vs sign of (y_hat_t - y_{t-1})
        dir_actual = np.sign(actuals[1:] - actuals[:-1])
        dir_pred = np.sign(forecasts[1:] - actuals[:-1])
        dir_acc = float(np.mean(dir_actual == dir_pred) * 100.0)

        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape_pct': round(mape, 2),
            'directional_accuracy_pct': round(dir_acc, 2)
        }
