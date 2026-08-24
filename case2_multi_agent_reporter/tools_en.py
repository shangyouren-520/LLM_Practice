"""
=============================================================================
Case 2: Multi-Agent Multimodal Financial Research Report System - DataTools
(Deterministic Tooling Layer)
-----------------------------------------------------------------------------
[Teaching Objective]:
This module implements the DataTools wrapper introduced on Slide 45.
Agents are not universally capable, so deterministic tasks such as data
retrieval and visualization are delegated to reliable Python scientific
computing libraries (Pandas / Matplotlib). This combines AI-driven cognition
with conventional deterministic programming.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd

# Force the non-interactive Agg backend to avoid Tkinter GUI thread conflicts
# when ThreadPoolExecutor is used, which can otherwise trigger:
# RuntimeError: main thread is not in main loop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Configure a robust font fallback chain for Matplotlib.
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
    'Arial Unicode MS',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


class DataTools:
    """Collection of data-processing and visualization utilities."""

    @staticmethod
    def get_stock_data(
        symbol: str = "300750",
        stock_name: str = "CATL",
        days: int = 60
    ) -> pd.DataFrame:
        """
        Tool 1: Retrieve historical stock-trading data
        (corresponds to Tool 1 on Slide 45).

        :param symbol: Stock ticker/code (for example, 300750)
        :param stock_name: Stock/company name (for example, CATL)
        :param days: Number of historical days to retrieve (default: 60)
        :return: DataFrame containing Date, Open, High, Low, Close, and Volume
        """
        logging.info(
            f"🛠️ [Tool Call] Retrieving the most recent {days} days of trading data "
            f"for {stock_name} ({symbol})..."
        )

        # Generate a high-fidelity synthetic financial time series resembling
        # the teaching-slide trend, using approximately CNY 97.33-115.00 as the range.
        np.random.seed(42)
        end_date = datetime(2025, 11, 25)
        dates = [end_date - timedelta(days=i) for i in range(days)]
        dates.reverse()

        # Generate a smoothly varying closing-price series.
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.02, days)

        # Shape the path to resemble the slide: an early rise toward 115,
        # followed by a pullback toward 97.33.
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        # Normalize and scale the series to the target range.
        prices = np.array(prices)
        prices = 97.0 + (
            (prices - prices.min()) / (prices.max() - prices.min())
        ) * (115.0 - 97.0)
        prices[-1] = 97.33  # Align the final close with the teaching slide.

        df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Open": np.round(
                prices * (1 + np.random.uniform(-0.01, 0.01, days)),
                2
            ),
            "High": np.round(
                prices * (1 + np.random.uniform(0.005, 0.02, days)),
                2
            ),
            "Low": np.round(
                prices * (1 - np.random.uniform(0.005, 0.02, days)),
                2
            ),
            "Close": np.round(prices, 2),
            "Volume": np.random.randint(150000, 350000, days)
        })

        return df

    @staticmethod
    def draw_chart(df: pd.DataFrame, title: str, filename: str) -> str:
        """
        Tool 2: Generate a professional stock trend chart
        (corresponds to Tool 2 on Slide 45).

        :param df: Historical stock-data DataFrame
        :param title: Main chart title
        :param filename: Output image filename
        :return: Absolute or relative path to the saved image
        """
        logging.info(
            f"🛠️ [Tool Call] Rendering a high-resolution market trend chart "
            f"with Matplotlib: {title}..."
        )

        output_dir = "output_charts"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

        # Plot the closing-price series.
        x_indices = range(len(df))
        ax1.plot(
            x_indices,
            df["Close"],
            color="#1F77B4",
            linewidth=2.0,
            label="Close (CNY)"
        )
        ax1.set_ylabel("Price (CNY)", fontsize=11, color="#1F77B4")
        ax1.tick_params(axis='y', labelcolor="#1F77B4")
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)

        # Configure sparse x-axis ticks, showing approximately one label every eight days.
        step = max(1, len(df) // 8)
        ax1.set_xticks(x_indices[::step])
        ax1.set_xticklabels(df["Date"].iloc[::step], rotation=25, fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Add moving-average reference lines (MA5 / MA20).
        ma5 = df["Close"].rolling(5).mean()
        ma20 = df["Close"].rolling(20).mean()
        ax1.plot(
            x_indices,
            ma5,
            color="#FF7F0E",
            linestyle=":",
            linewidth=1.2,
            label="MA5"
        )
        ax1.plot(
            x_indices,
            ma20,
            color="#2CA02C",
            linestyle="--",
            linewidth=1.2,
            label="MA20"
        )

        ax1.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        # Save a high-resolution image (teaching-slide standard: approximately 100-150 dpi).
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        logging.info(f"✅ Chart saved to: {save_path}")
        return save_path
