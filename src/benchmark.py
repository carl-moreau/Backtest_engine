import pandas as pd

class Benchmark:
    """
    Generates an equity curve for a composite benchmark (Asset Basket).
    
    This class is agnostic, accepting any weighting scheme (e.g., 100% SPY or 50/50 split).
    It assumes that the provided data ('data') is already aligned to the desired period.
    """
    
    def __init__(self, data: pd.DataFrame, composition: dict[str, float]):
        """
        Initializes the Benchmark with market data and a defined composition.

        Args:
            data (pd.DataFrame): A DataFrame containing price data (either MultiIndex or Wide format).
                                 This dataset must cover the backtest period.
            composition (dict[str, float]): A dictionary mapping tickers to their respective weights.
                                            Example: {'SPY': 1.0} or {'KO': 0.5, 'PEP': 0.5}.
        """
        self.data = data
        self.composition = composition
        
        # The sum of the weights must equal 1
        total_weight = sum(composition.values())
        if not (0.99 <= total_weight <= 1.01):
            print(f"Warning Benchmark : Somme des poids = {total_weight:.2f} (au lieu de 1.0)")

    def calculate(self, initial_capital: float) -> pd.Series:
        """
        Calculates the value of the passive portfolio (Buy & Hold) based on the provided data.

        Args:
            initial_capital (float): The starting capital for the benchmark portfolio.

        Returns:
            pd.Series: A Series representing the daily equity value of the benchmark portfolio.
                       Returns an empty Series if calculation fails or no data is found.
        """
        benchmark_equity = None
        
        for ticker, weight in self.composition.items():
            
            # --- 1. Price Extraction ---
            try:
                if isinstance(self.data.index, pd.MultiIndex):
                    # # Handle MultiIndex (Date, Ticker) structure
                    prices = self.data.xs(ticker, level=1)['Close']
                else:
                    # Handle Wide format (Tickers as Columns) or Simple DataFrame
                    if ticker in self.data.columns:
                        prices = self.data[ticker]
                    elif 'Close' in self.data.columns:
                        prices = self.data['Close']
                    else:
                        raise KeyError
            except KeyError:
                print(f"Benchmark: Ticker '{ticker}' not found in provided data.")
                continue
            
            # continue if no prices
            if prices.empty:
                continue

            # Logic: We calculate the performance relative to the start (Price_t / Price_0)
            # and multiply it by the specific capital allocated to this asset (Capital * Weight)
            start_price = prices.iloc[0]
            if start_price == 0: continue # Prevent division by zero errors
            
            allocated_amt = initial_capital * weight
            component_equity = (prices / start_price) * allocated_amt
            
            # Aggregation
            if benchmark_equity is None:
                benchmark_equity = component_equity
            else:
                # adds the equity value of the current component to the total
                benchmark_equity = benchmark_equity + component_equity
        
        # Return empty Series if no valid components were processed (e.g., all tickers missing)
        return benchmark_equity if benchmark_equity is not None else pd.Series()