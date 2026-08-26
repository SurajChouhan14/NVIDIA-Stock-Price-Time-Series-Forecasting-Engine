"""
Automated Unit Test Suite for NVIDIA Stock Price Time-Series Forecasting Engine.
Verifies Data Ingestion, ADF Stationarity Tests, ARIMA/Holt Multi-Step Forecasts, and Metric Bounds.
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
    Unit test cases for NVIDIA time-series forecasting engine.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = NVDADataLoader(data_dir="data")
        cls.data = cls.loader.load_data(test_days=60)
        cls.forecaster = NVDATimeSeriesForecaster(cls.data)

    def test_data_ingestion_and_splits(self):
        """Verify NVIDIA price data dimensions and train/test splits."""
        self.assertGreater(len(self.data['full_df']), 1000)
        self.assertEqual(len(self.data['test_df']), 60)
        self.assertEqual(len(self.data['train_df']) + len(self.data['test_df']), len(self.data['full_df']))

    def test_adf_stationarity_unit_root(self):
        """Verify differenced series is strictly stationary under ADF test."""
        raw_res = self.forecaster.run_adf_stationarity_test(series=self.data['full_df']['Close'].values)
        diff_res = self.forecaster.run_adf_stationarity_test(series=self.data['diff_series'])
        self.assertFalse(raw_res['is_stationary_5pct'])
        self.assertTrue(diff_res['is_stationary_5pct'])
        self.assertLess(diff_res['adf_t_statistic'], -3.43)

    def test_arima_forecast_validity(self):
        """Verify ARIMA forecast outputs positive finite values."""
        arima_pred = self.forecaster.fit_and_forecast_arima(p=2, d=1, q=1, forecast_horizon=60)
        self.assertEqual(len(arima_pred), 60)
        self.assertTrue(np.all(arima_pred > 0))
        self.assertTrue(np.all(np.isfinite(arima_pred)))

    def test_forecast_evaluation_metrics(self):
        """Verify RMSE, MAE, and MAPE metrics are positive and bounded."""
        arima_pred = self.forecaster.fit_and_forecast_arima(p=2, d=1, q=1, forecast_horizon=60)
        eval_res = self.forecaster.evaluate_forecast(arima_pred)
        self.assertGreater(eval_res['rmse'], 0.0)
        self.assertGreater(eval_res['mae'], 0.0)
        self.assertLess(eval_res['mape_pct'], 30.0)  # MAPE within reasonable financial bound


if __name__ == '__main__':
    unittest.main()
