from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Strategy(ABC):
    """
    Abstract Base Class for all trading strategies.

    This class defines the interface that any strategy must implement to be compatible
    with the backtesting engine. It ensures that every strategy provides a way to
    calculate the required historical data lookback and a method to generate trading signals.
    """

    def __init__(self):
        """
        Initializes the strategy with an empty weights dictionary.
        """
        self.weights = {}

    @property
    @abstractmethod
    def required_lookback(self) -> int:
        """
        Defines the minimum amount of historical data (in bars/days) required
        for the strategy to make its first decision.

        Returns:
            int: The number of historical periods needed.
        """
        pass
    
    @property
    def window_type(self) -> str:
        """
        Specifies the type of data windowing used by the strategy.

        Returns:
            str: "rolling" for a fixed-size moving window, or "expanding" for a growing window.
                 Defaults to "rolling".
        """
        return "rolling"

    @abstractmethod
    def generate_signals(self, historical_data: pd.DataFrame) -> dict:
        """
        Generates trading signals (target weights) based on historical data.

        Args:
            historical_data (pd.DataFrame): A DataFrame containing the historical market data
                                            needed for analysis.

        Returns:
            dict: A dictionary mapping ticker symbols to their target portfolio weights (e.g., {'AAPL': 0.5}).
        """
        pass
      


class PairsTradingStrategy(Strategy):
    """
    A statistical arbitrage strategy trading a pair of co-integrated assets (Long/Short).
    
    This strategy monitors the spread between two assets (Y and X). When the spread deviates
    significantly from its mean (Z-score), it bets on mean reversion.
    """
    
    def __init__(self, tickers: list[str],
                 window: int = 60, 
                 entry_z: float = 2.0, 
                 exit_z: float = 0.0):
        """
        Initializes the Pairs Trading strategy.

        Args:
            tickers (list[str]): A list of two tickers [Asset Y, Asset X]. 
                                 Asset Y is the dependent variable, Asset X is the hedge.
            window (int, optional): The rolling window size for calculating spread statistics. Defaults to 60.
            entry_z (float, optional): The Z-score threshold to open a position. Defaults to 2.0.
            exit_z (float, optional): The Z-score threshold to close a position. Defaults to 0.0.
        """
        super().__init__()
        self.ticker_y, self.ticker_x = tickers
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        
        # État interne (State Machine)
        # 0 = Cash, 1 = Long Spread (Long Y / Short X), -1 = Short Spread (Short Y / Long X)
        self.current_state = 0 

    @property
    def required_lookback(self) -> int:
        return self.window
    
    @property
    def window_type(self) -> str:
        return "rolling" # Fenêtre glissante stricte

    def generate_signals(self, historical_data: pd.DataFrame) -> dict:
        """
        Calculates the spread, Z-score, and determines positions based on a state machine.

        The process involves:
        1. Performing a Rolling OLS to find the dynamic Beta (Hedge Ratio).
        2. Calculating the spread: Spread = Price_Y - (Beta * Price_X).
        3. Normalizing the spread into a Z-Score.
        4. Updating the internal state (Neutral, Long Spread, Short Spread).
        5. Calculating weights to ensure dollar-neutrality based on the hedge ratio.

        Args:
            historical_data (pd.DataFrame): MultiIndex DataFrame with levels [Date, Ticker]
                                            and at least a 'Close' column.

        Returns:
            dict: Target weights for Asset Y and Asset X. Returns empty dict if data is insufficient.
        """
        
        try:
            # get closing prices
            y_series = historical_data.xs(self.ticker_y, level=1)['Close']
            x_series = historical_data.xs(self.ticker_x, level=1)['Close']
        except KeyError:
            return {}

        # return an empty dict if not enough data
        if len(y_series) < self.window:
            return {}

        # Rolling OLS
        # Price_Y = beta * Price_X + alpha
        # Beta represents the coverage ratio: How many shares of X for 1 share of Y
        X_const = sm.add_constant(x_series)
        model = sm.OLS(y_series, X_const).fit()
        beta = model.params.iloc[1]
        
        # spread et Z-score calculation
        # Spread = Y - beta * X
        spread = y_series - (beta * x_series)
        
        mean_spread = spread.mean()
        std_spread = spread.std()
        
        current_spread = spread.iloc[-1]
        
        if std_spread == 0:
            return {}
        
        z_score = (current_spread - mean_spread) / std_spread

        # State machine
        
        # neutral
        if self.current_state == 0:
            if z_score < -self.entry_z:
                self.current_state = 1  # cheap spread -> long spread
            elif z_score > self.entry_z:
                self.current_state = -1 # expensive spread -> short spread
        
        # long spread
        elif self.current_state == 1:
            if z_score >= -self.exit_z: # mean reversion (e.g.: > -0.5 ou > 0)
                self.current_state = 0
                
        # short spread
        elif self.current_state == -1:
            if z_score <= self.exit_z: # mean reversion (e.g.: > -0.5 ou > 0)
                self.current_state = 0

        # Determining the weights
        
        if self.current_state == 0:
            return {self.ticker_y: 0.0, self.ticker_x: 0.0}

        # Direction:
        # long spread = long Y / short X
        # short spread = short Y / long X
        direction_y = 1.0 if self.current_state == 1 else -1.0
        direction_x = -1.0 * direction_y

        # we trade 'beta' stocks X for each stock Y
        last_price_y = y_series.iloc[-1]
        last_price_x = x_series.iloc[-1]
        
        # Notional value of an unit of spread = (1 * Py) + (|beta| * Px)
        exposure_y = last_price_y
        exposure_x = abs(beta) * last_price_x
        total_exposure = exposure_y + exposure_x
        
        # standardization to get a sum equals to 1
        w_y_raw = exposure_y / total_exposure
        w_x_raw = exposure_x / total_exposure
        
        # we apply the directions
        return {
            self.ticker_y: w_y_raw * direction_y,
            self.ticker_x: w_x_raw * direction_x
        }
        



class SMACrossStrategy(Strategy):
    """
    A classic Trend Following strategy based on Simple Moving Average (SMA) crossovers.

    This strategy generates a buy signal when a short-term moving average crosses above
    a long-term moving average (Golden Cross) and exits/sells when it crosses below (Death Cross).
    """

    def __init__(self, ticker: list[str], short_window: int = 20, long_window: int = 60):
        """
        Initializes the SMA Cross strategy.

        Args:
            ticker (list[str]): A list containing a single ticker symbol (e.g., ['AAPL']).
            short_window (int, optional): The lookback period for the fast moving average. Defaults to 20.
            long_window (int, optional): The lookback period for the slow moving average. Defaults to 60.
        """
        self.ticker = ticker[0]
        self.short_window = short_window
        self.long_window = long_window

    @property
    def required_lookback(self) -> int:
        return self.long_window
    
    @property
    def window_type(self) -> str:
        return "rolling" 
    
    def generate_signals(self, historical_data: pd.DataFrame) -> dict:
        """
        Calculates the Short and Long SMAs and determines the allocation weight.

        Args:
            historical_data (pd.DataFrame): MultiIndex DataFrame containing 'Close' prices.

        Returns:
            dict: Target weight for the ticker. {ticker: 1.0} for Long, {ticker: 0.0} for Neutral.
        """
        
        try:
            prices = historical_data.xs(self.ticker, level=1)['Close']
        except KeyError:
            return {}

        if len(prices) < self.long_window:
            return {}
        
        # MA calculation
        short_mavg = prices.rolling(window=self.short_window).mean().iloc[-1]
        long_mavg = prices.rolling(window=self.long_window).mean().iloc[-1]
        
        # Trend Following
        target_weight = 0.0

        if short_mavg > long_mavg:
            # we buy (long)
            target_weight = 1.0 
        elif short_mavg < long_mavg:
            # we close our position or stay still
            target_weight = 0.0
            
        return {self.ticker: target_weight}
