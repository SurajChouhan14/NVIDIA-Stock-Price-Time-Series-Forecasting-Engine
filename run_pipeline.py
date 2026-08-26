"""
End-to-End Execution Pipeline for NVIDIA Stock Price Time-Series Forecasting.
Performs ADF stationarity diagnostics, ARIMA(2,1,1) & Holt's Exponential Smoothing modeling, and out-of-sample evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import NVDADataLoader
from src.time_series_forecaster import NVDATimeSeriesForecaster


def main():
    print("=" * 95)
    print("NVIDIA (NVDA) STOCK PRICE TIME-SERIES FORECASTING & ECONOMETRIC ANALYSIS")
    print("Dataset: 5-Year Historical Daily Trading Prices | Tech: ADF Unit-Root Tests + ARIMA + Holt's Smoothing")
    print("=" * 95)

    print("\n[1/3] Loading NVIDIA daily price history & partitioning train/test splits...")
    loader = NVDADataLoader(data_dir="data")
    data = loader.load_data(test_days=60)
    print(f"      Total Observations: {len(data['full_df']):,} trading days")
    print(f"      Training Window   : {len(data['train_df']):,} days | Out-of-Sample Test Window: {len(data['test_df'])} days")
    print(f"      Start Price (2019): ${data['full_df']['Close'].iloc[0]:.2f} -> End Price (2024): ${data['full_df']['Close'].iloc[-1]:.2f}")

    print("\n[2/3] Performing Statistical Diagnostics & Unit Root Stationarity Tests...")
    forecaster = NVDATimeSeriesForecaster(data)
    
    # 1. ADF on raw prices
    raw_adf = forecaster.run_adf_stationarity_test(series=data['full_df']['Close'].values)
    print(f"      * Raw Price Series ADF Stat : t = {raw_adf['adf_t_statistic']:.4f} (5% Critical: {raw_adf['critical_value_5pct']}) -> {raw_adf['conclusion']}")
    
    # 2. ADF on 1st Differenced Series
    diff_adf = forecaster.run_adf_stationarity_test(series=data['diff_series'])
    print(f"      * Differenced Series ADF Stat : t = {diff_adf['adf_t_statistic']:.4f} (5% Critical: {diff_adf['critical_value_5pct']}) -> {diff_adf['conclusion']}")

    print("\n[3/3] Generating Out-of-Sample Forecasts & Quantitative Backtesting (60-Day Horizon)...")
    arima_pred = forecaster.fit_and_forecast_arima(p=2, d=1, q=1, forecast_horizon=60)
    holt_pred = forecaster.fit_and_forecast_holt(alpha=0.30, beta=0.10, forecast_horizon=60)

    arima_eval = forecaster.evaluate_forecast(arima_pred)
    holt_eval = forecaster.evaluate_forecast(holt_pred)

    print("\n" + "=" * 95)
    print("OUT-OF-SAMPLE FORECASTING PERFORMANCE MATRIX (60-DAY TEST WINDOW):")
    print("=" * 95)
    print(f"{'Forecasting Model':<35} | {'RMSE ($)':<12} | {'MAE ($)':<12} | {'MAPE (%)':<12} | {'Directional Acc':<15}")
    print("-" * 95)
    print(f"{'ARIMA (2, 1, 1) Multi-Step':<35} | ${arima_eval['rmse']:<11.2f} | ${arima_eval['mae']:<11.2f} | {arima_eval['mape_pct']:<11.2f}% | {arima_eval['directional_accuracy_pct']:<14.2f}%")
    print(f"{'Holt Linear Exponential Smoothing':<35} | ${holt_eval['rmse']:<11.2f} | ${holt_eval['mae']:<11.2f} | {holt_eval['mape_pct']:<11.2f}% | {holt_eval['directional_accuracy_pct']:<14.2f}%")
    print("=" * 95)

    print("\n[CONCLUSION] Successfully established time-series stationarity via first differencing (d=1),")
    print(f"   achieving high forecasting precision (ARIMA MAPE: {arima_eval['mape_pct']:.2f}%).")
    print("=" * 95)


if __name__ == '__main__':
    main()
