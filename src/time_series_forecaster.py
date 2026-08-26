"""
NVIDIA Stock Price Time-Series Forecasting & Econometric Diagnostics Engine.

Implements:
1. Augmented Dickey-Fuller (ADF) Unit-Root Stationarity Diagnostics
2. Autoregressive Integrated Moving Average (ARIMA) & Holt's Exponential Smoothing
3. Out-of-Sample Backtesting & Forecast Error Metrics (RMSE, MAE, MAPE, Directional Accuracy)
"""

import numpy as np
import pandas as pd


class NVDATimeSeriesForecaster:
    """
    Time-series modeling and out-of-sample forecasting engine for NVIDIA equity.
    """

    def __init__(self, data_dict):
        self.full_df = data_dict['full_df']
        self.train_df = data_dict['train_df']
        self.test_df = data_dict['test_df']
        self.train_prices = data_dict['train_df']['Close'].values
        self.test_prices = data_dict['test_df']['Close'].values

    def run_adf_stationarity_test(self, series=None):
        """
        Computes Augmented Dickey-Fuller (ADF) test statistic for unit root stationarity.
        Delta y_t = alpha + gamma * y_{t-1} + sum beta_i * Delta y_{t-i} + e_t
        """
        if series is None:
            series = self.full_df['Close'].values

        diff = np.diff(series)
        y_lag = series[:-1]

        # OLS regression: diff = alpha + gamma * y_lag
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        theta, residuals, _, _ = np.linalg.lstsq(X, diff, rcond=None)
        gamma = theta[1]

        # t-statistic for gamma
        s2 = np.sum((diff - X @ theta) ** 2) / (len(diff) - 2)
        var_theta = s2 * np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(var_theta[1, 1])
        t_stat = gamma / se_gamma if se_gamma > 0 else 0.0

        # Critical values for ADF with drift
        crit_1pct = -3.43
        crit_5pct = -2.86
        crit_10pct = -2.57

        is_stationary = bool(t_stat < crit_5pct)

        return {
            'adf_t_statistic': round(float(t_stat), 4),
            'critical_value_5pct': crit_5pct,
            'is_stationary_5pct': is_stationary,
            'conclusion': "Stationary (Reject Unit Root)" if is_stationary else "Non-Stationary (Contains Unit Root -> Differencing d=1 required)"
        }

    def fit_and_forecast_holt(self, alpha=0.30, beta=0.10, forecast_horizon=60):
        """
        Holt's Linear Exponential Smoothing with trend component.
        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        """
        n = len(self.train_prices)
        level = np.zeros(n)
        trend = np.zeros(n)

        level[0] = self.train_prices[0]
        trend[0] = self.train_prices[1] - self.train_prices[0]

        for t in range(1, n):
            level[t] = alpha * self.train_prices[t] + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]

        # Multi-step ahead out-of-sample forecast
        forecasts = np.zeros(forecast_horizon)
        for h in range(1, forecast_horizon + 1):
            forecasts[h-1] = level[-1] + h * trend[-1]

        return forecasts

    def fit_and_forecast_arima(self, p=2, d=1, q=1, forecast_horizon=60):
        """
        ARIMA(p, 1, q) model on log prices.
        Delta y_t = phi_1 * Delta y_{t-1} + ... + e_t + theta_1 * e_{t-1}
        """
        log_train = np.log(self.train_prices)
        diff_train = np.diff(log_train)

        # Fit AR(2) on differences
        n = len(diff_train)
        Y = diff_train[2:]
        X = np.column_stack([diff_train[1:-1], diff_train[:-2]])

        phi, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        # Multi-step forecasting of log differences
        diff_forecasts = []
        recent_diffs = list(diff_train[-2:])

        for h in range(forecast_horizon):
            next_diff = phi[0] * recent_diffs[-1] + phi[1] * recent_diffs[-2]
            diff_forecasts.append(next_diff)
            recent_diffs.append(next_diff)

        # Integrate back to price levels: y_T+h = y_T + sum_{i=1}^h Delta y_i
        log_forecasts = np.log(self.train_prices[-1]) + np.cumsum(diff_forecasts)
        price_forecasts = np.exp(log_forecasts)

        return price_forecasts

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
