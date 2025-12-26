import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ReportGenerator:
    """
    Generates a performance report (Landscape Dashboard 16:9).
    
    This class consolidates backtest results, trade history, and benchmark data to create
    an interactive HTML dashboard. The dashboard includes:
    1. Equity Curve comparison (Strategy vs Benchmark).
    2. Comprehensive performance statistics (CAGR, Sharpe, Sortino, Drawdown).
    3. Trade analysis (Win rate, Avg PnL, etc.).
    4. Underwater Plot (Drawdown over time).
    """

    def __init__(self, metrics: dict, benchmark_equity: pd.Series = None, strategy_name: str = "Strategy"): #type: ignore
        """
        Initializes the ReportGenerator.

        Args:
            metrics (dict): The output dictionary from the Portfolio.get_performance_metrics() method.
                            Must contain 'History' (pd.DataFrame), 'Closed Trades' (list),
                            'Initial Capital' (float), and 'Final Equity' (float).
            benchmark_equity (pd.Series, optional): A daily series of the benchmark's equity value
                                                    to be plotted alongside the strategy.
            strategy_name (str, optional): The display name of the strategy for the report title.
                                           Defaults to "Strategy".
        """
        self.metrics = metrics
        self.strategy_name = strategy_name
        self.benchmark_equity = benchmark_equity 
        self.df = metrics.get('History', pd.DataFrame())
        self.closed_trades = metrics.get('Closed Trades', [])
        
        self.init_capital = metrics['Initial Capital']
        self.net_pnl = metrics['Final Equity'] - self.init_capital
        
        self.stats = {}
        self.trade_stats = {}

        if not self.df.empty:
            self._compute_advanced_stats()
            self.trade_stats = self._calculate_trade_metrics(self.closed_trades)

    def _compute_advanced_stats(self):
        """
        Calculates advanced financial metrics based on the daily returns series.

        Computes:
        - CAGR (Compound Annual Growth Rate)
        - Annualized Volatility
        - Sortino Ratio (Downside risk-adjusted return)
        - Calmar Ratio (Return vs Max Drawdown)
        
        Updates the internal `self.stats` dictionary.
        """
        df = self.df
        returns = df['returns']
        
        if len(df) < 2: return

        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        
        self.stats['CAGR'] = (self.metrics['Final Equity'] / self.init_capital) ** (1/years) - 1 if years > 0 else 0
        self.stats['Volatility'] = returns.std() * np.sqrt(252)
        
        downside = returns[returns < 0]
        sortino_denom = downside.std() * np.sqrt(252)
        self.stats['Sortino'] = (returns.mean() * 252) / sortino_denom if sortino_denom != 0 else 0
        
        max_dd = abs(self.metrics['Max Drawdown'])
        self.stats['Calmar'] = self.stats['CAGR'] / max_dd if max_dd != 0 else 0

    def _calculate_trade_metrics(self, closed_trades: list) -> dict:
        """
        Analyzes the list of closed trades to extract trading-specific metrics.

        Args:
            trades (list[dict]): A list of dictionaries representing individual closed trades.
                                 Each dict should contain 'pnl' (float).

        Returns:
            dict: A dictionary containing:
                  - Total Trades
                  - Win Rate (%)
                  - Average Trade PnL
                  - Profit Factor (Gross Win / Gross Loss)
                  - Largest Win / Largest Loss
        """
        if not closed_trades:
            return {"Total Trades": 0, "Win Rate": 0.0, "Profit Factor": 0.0, "Avg Win": 0.0, "Avg Loss": 0.0, "Risk/Reward": 0.0, "Expectancy": 0.0}

        df_trades = pd.DataFrame(closed_trades)
        
        # remove flat pnl trades
        real_trades = df_trades[df_trades['pnl'] != 0]
        
        winning_trades = real_trades[real_trades['pnl'] > 0]
        losing_trades = real_trades[real_trades['pnl'] < 0]

        n_total = len(real_trades)
        n_wins = len(winning_trades)
        n_losses = len(losing_trades)

        if n_total == 0: return {"Total Trades": 0}

        win_rate = n_wins / n_total
        avg_win = winning_trades['pnl'].mean() if n_wins > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if n_losses > 0 else 0.0
        
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "Total Trades": n_total,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Risk/Reward": risk_reward_ratio,
            "Expectancy": expectancy,
            "Largest Win": winning_trades['pnl'].max() if n_wins > 0 else 0.0,
            "Largest Loss": losing_trades['pnl'].min() if n_losses > 0 else 0.0
        }

    def generate(self, filename: str = "backtest_dashboard.html", output_folder: str = "reports"):
        """
        Builds and saves the interactive Plotly dashboard as an HTML file.

        Constructs a 3-row layout:
        - Row 1: Equity Curve (Strategy vs Benchmark).
        - Row 2: Performance Statistics Table & Trade Metrics Table.
        - Row 3: Drawdown Chart (Underwater plot).

        Args:
            filename (str, optional): The path/name of the output HTML file.
                                      Defaults to "report.html".
        """
        print(f"Generating the Dashboard for {self.strategy_name}...")
        
        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.15, 0.50, 0.35],
            specs=[
                [{"type": "domain", "colspan": 2}, None], 
                [{"type": "xy"}, {"type": "table"}],      
                [{"type": "xy", "colspan": 2}, None] # full width for drawdown
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.03,
            subplot_titles=("", "Equity Curve vs Benchmark", "Trade Statistics", "Drawdown", "")
        )

        # row 1: KPIs
        indicators = [
            ("Total Return", self.metrics['Total Return'], ".2%", "#2c3e50"),
            ("CAGR", self.stats.get('CAGR', 0), ".2%", "#2c3e50"),
            ("Sharpe", self.metrics['Sharpe Ratio'], ".2f", "#2980b9"),
            ("Max DD", self.metrics['Max Drawdown'], ".2%", "#c0392b"),
            ("Fees Paid", self.metrics.get('Total Fees Paid', 0), ".2f", "#e67e22"),
            ("Win Rate", self.trade_stats.get('Win Rate', 0), ".1%", "#27ae60"),
            ("Profit Factor", self.trade_stats.get('Profit Factor', 0), ".2f", "#27ae60")
        ]
        
        for i, (title, val, fmt, col) in enumerate(indicators):
            fig.add_trace(go.Indicator(
                mode="number", value=val,
                number={'valueformat': fmt, 'font': {'size': 20, 'color': col}},
                title={'text': title, 'font': {'size': 12, 'color': 'gray'}},
                domain={'x': [i/7, (i+1)/7 - 0.02], 'y': [0.85, 1]}
            ))

        # row 2: Equity Curve & stats
        
        # Benchmark (Buy & Hold)
        if self.benchmark_equity is not None:
             fig.add_trace(go.Scatter(
                x=self.benchmark_equity.index, y=self.benchmark_equity, 
                mode='lines', name='Buy & Hold',
                line=dict(color='gray', width=1.5, dash='dash'), # Gray dotted line
                opacity=0.7
            ), row=2, col=1)

        # Strategy Equity
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['equity'], 
            mode='lines', name='Strategy',
            line=dict(color='#0047AB', width=2), 
            fill='tozeroy', fillcolor='rgba(0, 71, 171, 0.1)'
        ), row=2, col=1)

        # stats
        table_data = [
            ["<b>Total Trades</b>", f"{self.trade_stats.get('Total Trades',0)}"],
            ["<b>Win Rate</b>", f"{self.trade_stats.get('Win Rate',0):.1%}"],
            ["<b>Profit Factor</b>", f"{self.trade_stats.get('Profit Factor',0):.2f}"],
            ["<b>Expectancy</b>", f"{self.trade_stats.get('Expectancy',0):,.2f} €"],
            ["<b>Avg Win</b>", f"{self.trade_stats.get('Avg Win',0):,.2f} €"],
            ["<b>Avg Loss</b>", f"{self.trade_stats.get('Avg Loss',0):,.2f} €"],
            ["<b>Largest Win</b>", f"{self.trade_stats.get('Largest Win',0):,.2f} €"],
            ["<b>Largest Loss</b>", f"{self.trade_stats.get('Largest Loss',0):,.2f} €"],
        ]
        headers, cells = ["Metric", "Value"], list(zip(*table_data))

        fig.add_trace(go.Table(
            header=dict(values=[f"<b>{h}</b>" for h in headers], fill_color='#2c3e50', font=dict(color='white'), align='left'),
            cells=dict(values=cells, fill_color=[['whitesmoke', 'white']*5], align='left', height=25, font=dict(color='#2c3e50'))
        ), row=2, col=2)

        # row 3: drawdown
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['drawdown'], 
            mode='lines', name='Drawdown',
            line=dict(color='#c0392b', width=1),
            fill='tozeroy', fillcolor='rgba(192, 57, 43, 0.1)'
        ), row=3, col=1)

        fig.update_layout(
            width=1450, height=800,
            title_text=f"<b>Backtest Report</b> | {self.strategy_name}",
            template="plotly_white",
            margin=dict(l=50, r=50, t=100, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        current_working_dir = os.getcwd()
        target_dir = os.path.join(current_working_dir, output_folder)
        os.makedirs(target_dir, exist_ok=True)
        full_path = os.path.join(target_dir, filename)
        fig.write_html(full_path)
        print(f"Dashboard saved : {full_path}")