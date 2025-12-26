# Quantitative Backtesting Engine (Python)

An event-driven, object-oriented backtesting framework designed to simulate trading strategies with a focus on architectural robustness and realistic market frictions.

This project was built as an educational initiative to bridge the gap between academic theory and professional engineering standards, implementing core concepts like **Separation of Concerns (SoC)** and **SOLID** principles.

## Key Features

* **Event-Driven Architecture:** Simulates the passage of time bar-by-bar to strictly prevent *Look-Ahead Bias*.
* **Realistic Frictions:**
    * **Transaction Costs:** Configurable Fee Models (Fixed, Percentage, etc.).
    * **Slippage Modeling:** Linear price impact simulation based on order size.
* **Risk Management:**
    * **Dynamic Sizing:** Allocates positions based on strict capital percentage limits.
    * **Integer Constraints:** Handles realistic share counts (no fractional shares).
* **Institutional Reporting:** Generates interactive HTML dashboards (via Plotly) featuring Equity Curves, Drawdown plots, and advanced KPIs (Sortino, Calmar, CAGR).
* **Strategy Agnostic:** Designed to handle **N-assets**. The engine accepts any strategy inheriting from the abstract base class, whether it's a Single-Asset Trend Following or a Multi-Asset Statistical Arbitrage.

## Architecture

The system is decoupled into three main layers:

1.  **Data Layer:** Handles ingestion and alignment of OHLCV data (MultiIndex Pandas DataFrames).
2.  **Strategy Layer:** Pure logic components. They receive historical windows ($t_{-N}$ to $t_{-1}$) and output target weights.
3.  **Portfolio/Execution Layer:** The "Brain". It manages cash, calculates realized P&L (VWAP basis), applies fees/slippage, and tracks exposure.
