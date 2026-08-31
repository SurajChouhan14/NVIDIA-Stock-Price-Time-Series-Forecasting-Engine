"""
Automated Unit Test Suite for NVIDIA Stock Price Time-Series Forecasting Engine.
Tests:
1. Historical Price Ingestion & Train/Test Partition Dimensions (1,390 days, 60-day holdout)
2. Augmented Dickey-Fuller (ADF) Stationarity Diagnostics (Raw Non-Stationary vs Differenced Stationary)
3. Box-Jenkins ARIMA(2,1,1) Forecast Validity & Positivity
4. Forecast Evaluation Metric Reproducibility (MAPE ~ 10.19%, RMSE, MAE)
5. Comparative Benchmark Performance (ARIMA outperforms Holt Linear Smoothing)
"""

import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import NVDADataLoader
from src.time_series_forecaster import NVDATimeSeriesForecaster


class TestNVDATimeSeriesForecaster(unittest.TestCase):
    """
    Hard unit tests for NVIDIA time-series forecasting engine and econometric properties.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = NVDADataLoader(data_dir="data")
        cls.data = cls.loader.load_data(test_days=60)
        cls.forecaster = NVDATimeSeriesForecaster(cls.data)

    def test_1_data_ingestion_and_splits(self):
        """Verify NVIDIA price data dimensions and exact 60-day out-of-sample partition."""
        self.assertEqual(len(self.data['full_df']), 1390)
        self.assertEqual(len(self.data['test_df']), 60)
        self.assertEqual(len(self.data['train_df']), 1330)
        self.assertEqual(len(self.data['train_df']) + len(self.data['test_df']), len(self.data['full_df']))

    def test_2_adf_stationarity_unit_root(self):
        """Verify raw series contains unit root while differenced series is strictly stationary under ADF test."""
        raw_res = self.forecaster.run_adf_stationarity_test(series=self.data['full_df']['Close'].values)
        diff_res = self.forecaster.run_adf_stationarity_test(series=self.data['diff_series'])

        self.assertFalse(raw_res['is_stationary_5pct'])
        self.assertTrue(diff_res['is_stationary_5pct'])
        # Dynamic comparison against 5% critical value
        self.assertLess(diff_res['adf_t_statistic'], diff_res['critical_value_5pct'])
        self.assertAlmostEqual(diff_res['adf_t_statistic'], -37.8420, delta=0.50)
        self.assertLess(diff_res['p_value'], 0.001)

    def test_3_arima_forecast_validity(self):
        """Verify ARIMA forecast outputs positive finite values over 60-day recursive horizon."""
        arima_pred = self.forecaster.fit_and_forecast_arima(p=2, d=1, q=1, forecast_horizon=60)
        self.assertEqual(len(arima_pred), 60)
        self.assertTrue(np.all(arima_pred > 0.0))
        self.assertTrue(np.all(np.isfinite(arima_pred)))

    def test_4_forecast_evaluation_metrics_reproducibility(self):
        """Verify ARIMA forecast achieves ~10.19% MAPE, valid RMSE, and MAE bounds."""
        arima_pred = self.forecaster.fit_and_forecast_arima(p=2, d=1, q=1, forecast_horizon=60)
        eval_res = self.forecaster.evaluate_forecast(arima_pred)

        self.assertAlmostEqual(eval_res['mape_pct'], 10.19, delta=0.50)
        self.assertAlmostEqual(eval_res['rmse'], 94.37, delta=2.0)
        self.assertAlmostEqual(eval_res['mae'], 85.56, delta=2.0)
        self.assertGreater(eval_res['directional_accuracy_pct'], 0.0)

    def test_5_arima_vs_holt_benchmark(self):
        """Verify ARIMA(2,1,1) achieves lower out-of-sample MAPE than Holt linear smoothing."""
        arima_pred = self.forecaster.fit_and_forecast_arima(p=2, d=1, q=1, forecast_horizon=60)
        holt_pred = self.forecaster.fit_and_forecast_holt(optimized=True, forecast_horizon=60)

        arima_eval = self.forecaster.evaluate_forecast(arima_pred)
        holt_eval = self.forecaster.evaluate_forecast(holt_pred)

        self.assertLess(arima_eval['mape_pct'], holt_eval['mape_pct'])
        self.assertLess(arima_eval['rmse'], holt_eval['rmse'])


if __name__ == '__main__':
    unittest.main()
