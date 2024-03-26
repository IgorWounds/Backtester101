"""Base class for trading strategies."""

from typing import Any

import pandas as pd


class Strategy:
    """Base class for trading strategies."""

    def __init__(self, indicators: dict, signal_logic: Any):
        """Initialize the strategy with indicators and signal logic."""
        self.indicators = indicators
        self.signal_logic = signal_logic

    def generate_signals(
        self, data: pd.DataFrame | dict[str, pd.DataFrame]
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Generate trading signals based on the strategy's indicators and signal logic."""
        if isinstance(data, dict):
            for _, asset_data in data.items():
                self._apply_strategy(asset_data)
        else:
            self._apply_strategy(data)
        return data

    def _apply_strategy(self, df: pd.DataFrame) -> None:
        """Apply the strategy to a single dataframe."""
        for name, indicator in self.indicators.items():
            df[name] = indicator(df)

        df["signal"] = df.apply(lambda row: self.signal_logic(row), axis=1)
        df["positions"] = df["signal"].diff().fillna(0)
