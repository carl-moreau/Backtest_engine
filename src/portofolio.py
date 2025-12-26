from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np



class FeeModel(ABC):
    """
    Abstract Base Class for calculating transaction fees.
    """

    @abstractmethod
    def calculate(self, trade_value) -> float:
        """
        Calculates the fee for a given trade value.

        Args:
            trade_value (float): The monetary value of the trade.

        Returns:
            float: The calculated fee amount.
        """
        pass


class SlippageModel(ABC):
    """
    Abstract Base Class for estimating execution price slippage.
    """

    @abstractmethod
    def calculate(self, price: float, quantity: float) -> float:
        """
        Calculates the slippage per share/unit.

        Args:
            price (float): The intended execution price.
            quantity (float): The number of shares/units to be traded.

        Returns:
            float: The execution price including slippage.
        """
        pass


class PercentageFeeModel(FeeModel):
    """
    Fee model that charges a percentage of the trade value.
    """

    def __init__(self, fee_rate: float = 0.001, min_fee: float = 0.0) -> None:
        """
        Initializes the percentage fee model.

        Args:
            fee_rate (float, optional): The fee rate as a decimal (e.g., 0.001 for 0.1%). Defaults to 0.001.
            min_fee (float, optional): The minimum fee per transaction. Defaults to 0.0.
        """
        self.fee_rate = fee_rate
        self.min_fee = min_fee
    
    def calculate(self, trade_value: float) -> float:
        """
        Calculates the fee based on a fixed percentage.

        Args:
            trade_value (float): The monetary value of the trade.

        Returns:
            float: The calculated fee, respecting the minimum fee constraint.
        """
        
        abs_value = abs(trade_value)
        
        fees = abs_value * self.fee_rate

        return max(fees, self.min_fee)
    

class LinearSlippageModel(SlippageModel):
    """
    Slippage model assuming a linear market impact based on trade size.
    """

    def __init__(self, base_bps: float = 5.0, impact_per_share: float = 0.0001) -> None:
        """
        Initializes the linear slippage model.

        Args:
            base_bps (float, optional): The base bid-ask spread in basis points. Defaults to 5.0.
            impact_per_share (float, optional): The market impact coefficient (Lambda).
                                                How much the price moves per share traded. Defaults to 0.0001.
        """
        self.base_pct = base_bps / 10000 # bps to decimal (ex: 0.0005)
        self.impact_per_share = impact_per_share

    def calculate(self, price: float, quantity: float):
        """
        Calculates slippage including a fixed spread and a linear market impact.

        Args:
            price (float): The current market price.
            quantity (float): The quantity traded.

        Returns:
            float: The execution price including slippage.
        """
        # base bps
        base_slippage = price * self.base_pct
        
        # linear impact
        volume_impact = abs(quantity) * self.impact_per_share
        
        # total slippage
        total_slippage = base_slippage + volume_impact
        
        # adapt to trade direction (long or short)
        direction = 1 if quantity > 0 else -1
        
        # If I buy, I buy at a higher price
        # If I sell, I sell at a lower price
        execution_price = price + (total_slippage * direction)
        
        return execution_price
        
    
class RebalanceMode(Enum):
    """
    Enumeration defining different portfolio rebalancing strategies.

    Attributes:
        CONTINUOUS_WITH_BUFFER: Rebalance continuously but ignore small deviations defined by a buffer.
        SIGNAL_CHANGE: Rebalance only when the target signal changes significantly.
    """
    CONTINUOUS_WITH_BUFFER = "continuous_buffer"
    SIGNAL_CHANGE = "signal_change"              


class Portfolio:
    """
    Manages the portfolio state, including cash, positions, and P&L tracking.
    Handles order execution, fees, and slippage.
    """

    def __init__(self, initial_capital: float, fee_model: FeeModel, slippage_model: SlippageModel,
                 rebalance_mode: RebalanceMode = RebalanceMode.CONTINUOUS_WITH_BUFFER, 
                 trade_buffer: float = 0.05, 
                 max_allocation_pct: float = 0.2) -> None:
        """
        Initializes the portfolio manager.

        Args:
            initial_capital (float): The starting capital.
            fee_model (FeeModel, optional): The model used to calculate transaction fees.
            slippage_model (SlippageModel, optional): The model used to estimate slippage.
            rebalance_mode (RebalanceMode, optional): The logic determining when to rebalance.
            max_allocation_pct (float, optional): The maximum allowed allocation per asset (0.0 to 1.0). Defaults to 20%.
            trade_buffer (float, optional): The minimum weight change required to trigger a trade in continuous mode.
        """
        
        # static parameters
        self.initial_capital = initial_capital
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.rebalance_mode = rebalance_mode
        self.trade_buffer = trade_buffer
        self.max_allocation_pct = max_allocation_pct
        
        # state description parameters
        self.cash = initial_capital
        self.positions = {}         # {Ticker: quantity}
        self.avg_entry_prices = {}  # {Ticker: avg_entry_price}
        self.history = []
        self.realized_pnl_history = []

        self.closed_trades = []  # list[dict] for all completed trades
        self.total_fees_paid = 0.0 
        
        # memory for “Signal Change” mode
        self.previous_target_weights = {}
    
    def get_unrealized_pnl(self, current_prices: pd.DataFrame) -> float:
        """
        Calculates the unrealized Profit and Loss (P&L) for all open positions.

        Args:
            current_prices (pd.DataFrame): Current market prices (usually closing prices).

        Returns:
            float: The total unrealized P&L of the portfolio.
        """
        pnl = 0
        for ticker, qty in self.positions.items():
            if ticker in current_prices.index:
                current_price = current_prices.loc[ticker]
                entry_price = self.avg_entry_prices.get(ticker, 0)
                
                pnl += (current_price - entry_price) * qty
        return pnl
    
    def record_snapshot(self, date, closing_prices):
        """
        Records the daily state of the portfolio (Equity, Cash, Positions, etc.).

        Args:
            date (datetime): The current date.
            current_prices (pd.Series): Current market prices to calculate unrealized P&L.
        """
        # unrealized P&L with closing prices
        unrealized_pnl = self.get_unrealized_pnl(closing_prices)
        
        # mark-to-market of the positions
        mv_positions = 0
        for t, qty in self.positions.items():
            if t in closing_prices.index:
                mv_positions += qty * closing_prices.loc[t]
        
        total_equity = self.cash + mv_positions
        
        self.history.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
        })

    def get_performance_metrics(self) -> dict:
        """
        Computes and returns a comprehensive summary of the portfolio's performance.

        Calculates metrics such as Total Return, Sharpe Ratio, and Max Drawdown based on 
        the recorded history.

        Returns:
            dict: A dictionary containing scalar metrics (Sharpe, MaxDD, etc.), 
                  the full daily history DataFrame, and the list of closed trades.
        """
        if not self.history:
            return {}
            
        # dict to pd.DataFrame
        df = pd.DataFrame(self.history).set_index('date')
        
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        # DD calculation
        df['cum_max'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']
        

        total_ret = (df['equity'].iloc[-1] / self.initial_capital) - 1
        
        std_dev = df['returns'].std()
        sharpe = (df['returns'].mean() / std_dev * np.sqrt(252)) if std_dev != 0 else 0
        
        max_dd = df['drawdown'].min()
        
        # return dict
        return {
            "Total Return": total_ret,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Final Equity": df['equity'].iloc[-1],
            "Initial Capital": self.initial_capital,
            "Total Fees Paid": self.total_fees_paid,
            
            "History": df,
            "Closed Trades": self.closed_trades 
        }
    
    def _should_trade(self, ticker: str, target_weight: float, current_val: float, total_equity: float) -> bool:
        """
        Determines if a trade should be executed for a specific asset based on the rebalance mode.

        Args:
            ticker (str): The symbol of the asset.
            target_weight (float): The desired weight for the asset.
            current_val (float): The current market value of the position in dollars.
            total_equity (float): The total equity of the portfolio.

        Returns:
            bool: True if a trade is required, False otherwise.
        """
        
        # Rebalance on Signal Change
        if self.rebalance_mode == RebalanceMode.SIGNAL_CHANGE:
            prev_target = self.previous_target_weights.get(ticker, 0)

            # we trade if signal changed significantly
            if not np.isclose(target_weight, prev_target, atol=1e-5):
                return True
            return False
        
        elif self.rebalance_mode == RebalanceMode.CONTINUOUS_WITH_BUFFER:
            if target_weight == 0 and current_val != 0:
                return True
            
            target_val = target_weight * total_equity

            # percentage gap in terms of total portfolio
            weight_diff_pct = abs(target_val - current_val) / total_equity if total_equity > 0 else 0

            if weight_diff_pct > self.trade_buffer:
                return True
            return False
        
        return False

    def rebalance(self, target_weights: dict, current_prices: pd.DataFrame, date):
        """
        Adjusts portfolio positions to match the target weights provided by the strategy.

        This method calculates the necessary trades to move from the current allocation
        to the target allocation, accounting for cash constraints, max allocation limits,
        execution costs (fees + slippage), and rebalancing rules.

        Args:
            target_weights (dict): A dictionary mapping tickers to target portfolio weights.
            current_prices (pd.DataFrame): Current market prices for the assets.
            date (datetime): The current date of the simulation.
        """
        
        # mark-to-market of all positions
        mv_positions = 0
        for t, qty in self.positions.items():
            if t in current_prices.index:
                mv_positions += qty * current_prices.loc[t]
        total_equity = self.cash + mv_positions
        
        daily_realized_pnl = 0

        all_tickers = list(self.positions.keys() | target_weights.keys())

        # main loop
        for ticker in all_tickers:
            if ticker not in current_prices.index: continue
            
            price = current_prices.loc[ticker]
            current_qty = self.positions.get(ticker, 0)
            target_w_raw = target_weights.get(ticker, 0.0) # ex: 0.5
            
            # exposure constraint
            # e.g.: 0.5 * 0.2 = 0.1 (10% of total capital invested in this asset)
            target_w_adjusted = target_w_raw * self.max_allocation_pct
            
            current_val = current_qty * price
            
            # decision to whether trade or not
            if not self._should_trade(ticker, target_w_adjusted, current_val, total_equity):
                self.previous_target_weights[ticker] = target_w_adjusted
                continue

            # determining position size
            target_val_dollars = target_w_adjusted * total_equity
            
            target_qty_raw = target_val_dollars / price
            target_qty_int = np.floor(target_qty_raw) if target_qty_raw >= 0 else np.ceil(target_qty_raw)
            
            delta_final = target_qty_int - current_qty
            
            if delta_final == 0:
                self.previous_target_weights[ticker] = target_w_adjusted
                continue

            # execution and pnl
            execution_price = self.slippage_model.calculate(price, delta_final)
            trade_val = delta_final * execution_price
            fees = self.fee_model.calculate(trade_val)
            self.total_fees_paid += fees 
            
            avg_price = self.avg_entry_prices.get(ticker, 0)
            
            # 1. we increase our exposure, update avg_entry_price
            if (current_qty * delta_final) > 0: 
                old_cost = current_qty * avg_price
                new_cost = delta_final * execution_price
                new_avg = (old_cost + new_cost) / (current_qty + delta_final)
                self.avg_entry_prices[ticker] = new_avg
            
            # 2. we whether reduce our exposure or liquidate the position
            elif (abs(delta_final) <= abs(current_qty)) and (current_qty * delta_final < 0):
                direction = 1 if current_qty > 0 else -1
                trade_pnl = abs(delta_final) * (execution_price - avg_price) * direction
                daily_realized_pnl += trade_pnl
                self.closed_trades.append({
                    'date': date, 'ticker': ticker, 'pnl': trade_pnl, 'type': 'Partial'
                })
                
            # 3. from long to short (flip)
            else:
                direction = 1 if current_qty > 0 else -1
                trade_pnl = abs(current_qty) * (execution_price - avg_price) * direction
                daily_realized_pnl += trade_pnl
                self.closed_trades.append({
                    'date': date, 'ticker': ticker, 'pnl': trade_pnl, 'type': 'Close'
                })
                # new avg_entry_price for the new opposite position
                self.avg_entry_prices[ticker] = execution_price

            # update cash and inventory
            self.cash -= (trade_val + fees)
            self.positions[ticker] = current_qty + delta_final
            
            # if position == 0, remove ticker from inventory
            if self.positions[ticker] == 0:
                del self.positions[ticker]
                if ticker in self.avg_entry_prices: del self.avg_entry_prices[ticker]
            
            # save the current signal
            self.previous_target_weights[ticker] = target_w_adjusted

        # end of the main loop
        self.realized_pnl_history.append(daily_realized_pnl)
