"""
NVIDIA Stock Historical Daily Price Data Ingestion Module.
Loads OHLCV stock history, calculates log returns, moving averages, and train/test splits.
"""

import os
import pandas as pd
import numpy as np


class NVDADataLoader:
    """
    Data ingestion and feature preparation engine for NVIDIA stock price time-series modeling.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, "nvda_stock_prices.csv")

    def load_data(self, test_days=60):
        """
        Loads NVIDIA daily price history and creates train/test partitions.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Log_Close'] = np.log(df['Close'])
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        df['Diff_Log_Close'] = df['Log_Close'].diff().fillna(0)

        train_df = df.iloc[:-test_days].copy()
        test_df = df.iloc[-test_days:].copy()

        return {
            'full_df': df,
            'train_df': train_df,
            'test_df': test_df,
            'close_series': df['Close'].values,
            'log_close_series': df['Log_Close'].values,
            'diff_series': df['Diff_Log_Close'].values[1:]
        }
