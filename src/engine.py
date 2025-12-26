import pandas as pd

from src.strategies import Strategy
from src.portofolio import Portfolio



class BacktestEngine:
    """
    Core execution engine for backtesting trading strategies.

    This class orchestrates the interaction between market data, the trading strategy,
    and the portfolio management system. It iterates through historical data day-by-day
    to simulate trading decisions and execution.
    """

    def run(self, data_feed: pd.DataFrame, strategy: Strategy, portfolio: Portfolio, liquidate_at_end: bool = True):
        """
        Executes the backtest simulation.

        Iterates through the provided `data_feed` chronologically. At each step:
        1. Extracts the relevant historical window for the strategy.
        2. Generates trading signals via `strategy.generate_signals()`.
        3. Executes rebalancing via `portfolio.rebalance()` using current market prices.
        4. Records a daily snapshot of the portfolio state.

        Args:
            data_feed (pd.DataFrame): Historical market data. Must be a MultiIndex DataFrame 
                                      with levels (Date, Ticker) containing OHLCV columns.
            strategy (Strategy): The trading strategy instance containing the logic for generating signals.
            portfolio (Portfolio): The portfolio instance handling positions, cash, fees, and P&L tracking.
            liquidate_at_end (bool, optional): If True, closes all open positions at the last available 
                                               market price to realize final P&L. Defaults to True.

        Returns:
            dict: A dictionary containing performance metrics and history, returned by 
                  `portfolio.get_performance_metrics()`.
        
        Raises:
            ValueError: If the `data_feed` does not contain enough historical data to satisfy 
                        the strategy's `required_lookback`.
        """

        # Data preprocessing
        data_feed = data_feed.sort_index(level=0)
        unique_dates = data_feed.index.get_level_values(0).unique()
        
        lookback = strategy.required_lookback

        if len(unique_dates) < lookback:
             raise ValueError(f"Not enough data. Needed: {lookback}, Available: {len(unique_dates)}")


        for i, current_date in enumerate(unique_dates):
            
            # Warm-up
            if i < lookback:
                continue

            # Prevent look-ahead bias
            end_date_window = unique_dates[i - 1] 
            
            if strategy.window_type == "rolling":
                start_date_window = unique_dates[i - lookback]
            elif strategy.window_type == "expanding":
                start_date_window = unique_dates[0]
            else:
                raise ValueError(f"Unknown window_type: {strategy.window_type}")

            historical_window = data_feed.loc[start_date_window:end_date_window]

            # generates trading signals
            target_weights = strategy.generate_signals(historical_window)

            try:
                # Retrieval of data from current day for execution and valuation
                daily_data = data_feed.loc[current_date]
                
                # execution at the opening prices
                execution_prices = daily_data['Open']
                portfolio.rebalance(target_weights, execution_prices, current_date)
                
                # save prices for reporting
                closing_prices = daily_data['Close']
                
                # record snapshot for reporting
                portfolio.record_snapshot(current_date, closing_prices)
                
            except KeyError:
                # handle missing days
                continue

        # closes all open positions at the last available market price to realize final P&L
        if liquidate_at_end and portfolio.positions:
            print("\nClosing of all positions...")
            
            # last date available
            last_date = unique_dates[-1]
            
            # we use closing prices to liquidate at the very end
            try:
                closing_prices = data_feed.loc[last_date]['Close']
                
                # we send to the portfolio an empty dict to force the closing of all positions, all weights = 0
                empty_targets = {} 
                
                portfolio.rebalance(empty_targets, closing_prices, last_date)
                
                # last snapshot to calculate the final cash & equity value
                portfolio.record_snapshot(last_date, closing_prices)
                
            except KeyError:
                print("No prices found to liquidate.")
        
        # we ask the portfolio to calculate all the statistics for reporting
        return portfolio.get_performance_metrics()

