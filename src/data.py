from dataclasses import dataclass
import pandas as pd
import yfinance as yf
from curl_cffi import requests
session = requests.Session(impersonate="chrome")


@dataclass
class DataHandler:
    """
    Handles data acquisition from external sources (Yahoo Finance).

    This class is responsible for downloading and formatting historical market data
    for a specified list of tickers within a given date range. It uses a custom
    session to handle requests robustly.

    Attributes:
        tickers (list[str]): A list of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOG']).
        start_date (str): The start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): The end date for data retrieval in 'YYYY-MM-DD' format.
    """
    tickers: list[str]
    start_date: str
    end_date: str

    def download(self) -> pd.DataFrame:
        """
        Downloads historical market data for the initialized tickers.

        Fetches OHLCV data from Yahoo Finance using yfinance. The data is then
        restructured into a MultiIndex DataFrame (Date, Ticker) to be compatible
        with the backtesting engine.

        Returns:
            pd.DataFrame: A MultiIndex DataFrame containing the market data.
                          The index levels are (Date, Ticker).
                          Columns typically include 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        raw_data = yf.download(tickers=self.tickers, start=self.start_date, end=self.end_date,
                           auto_adjust=False, progress=False, session=session)
        data_feed = raw_data.stack(level=1, future_stack=True).sort_index() # type: ignore
        return data_feed # type: ignore