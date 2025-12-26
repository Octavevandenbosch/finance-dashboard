import yfinance as yf
import pandas as pd
import numpy as np
from threading import Thread

class TickerScraper:
    def __init__(self, ticker_list, num_threads=5) -> None:
        self.ticker_list = ticker_list
        self.num_threads = num_threads
        self.data = self.download_tickers_threaded()
        # We process the data into a list of dictionaries
        self.parsed_data = [self.parse_ticker(ticker, self.data[ticker]) for ticker in self.data if ticker in self.data]

    def download_tickers_threaded(self):
        def download_one_ticker(ticker_name, output):
            try:
                # yfinance often works better without .US suffix for US stocks
                # but for international stocks (e.g. AIR.PA), the suffix is needed.
                # We try as-is first.
                search_ticker = ticker_name
                if ticker_name.endswith(".US"):
                    search_ticker = ticker_name.replace(".US", "")
                
                ticker_obj = yf.Ticker(search_ticker)
                
                # Fetch info and fast_info
                # Note: fast_info is newer and sometimes more reliable for price/mcap
                output[ticker_name] = (ticker_obj.info, ticker_obj.fast_info)
                # print(f"✅ Done downloading {ticker_name} (YF)")
            except Exception as e:
                print(f"⚠️ Failed to fetch {ticker_name} from YF: {e}")

        output = {}
        threads = []
        
        # Batch threads to avoid overwhelming logic
        # Simple implementation: start all threads in batches of size num_threads
        for i in range(0, len(self.ticker_list), self.num_threads):
            batch = self.ticker_list[i:i + self.num_threads]
            batch_threads = []
            for ticker in batch:
                t = Thread(target=download_one_ticker, args=(ticker, output))
                t.start()
                batch_threads.append(t)
            
            for t in batch_threads:
                t.join()

        return output

    def load_string_value(self, stock, key):
        try:
            x = stock.get(key)
            if x is None:
                return None
            return str(x)
        except:
            return None

    def load_number_value(self, stock, key):
        try:
            # Handle fast_info object which acts like a dict but isn't exactly one sometimes
            if hasattr(stock, key):
                x = getattr(stock, key)
            else:
                x = stock.get(key)
                
            if x is None:
                return None
            return float(x)
        except:
            return None

    def parse_ticker(self, ticker_name, ticker_data):
        if not ticker_data:
            return {}
            
        ticker_info, ticker_fast_info = ticker_data

        # Mapping to EODHD columns expected by the app
        current_price = (
            self.load_number_value(ticker_info, "currentPrice")
            or self.load_number_value(ticker_info, "regularMarketPrice")
            or self.load_number_value(ticker_fast_info, "last_price")
        )

        dividend_yield = self.load_number_value(ticker_info, "dividendYield")

        row = {
            "ticker": ticker_name,
            "name": self.load_string_value(ticker_info, "shortName") or self.load_string_value(ticker_info, "longName"),
            "industry": self.load_string_value(ticker_info, "industry"),
            "country": self.load_string_value(ticker_info, "country"),
            "currency": self.load_string_value(ticker_info, "currency"),
            
            # Price: Prefer regularMarketPrice from info, fallback to fast_info last_price
            "current price": current_price,
            
            "market cap": self.load_number_value(ticker_info, "marketCap") or self.load_number_value(ticker_fast_info, "market_cap"),
            "enterprise value": self.load_number_value(ticker_info, "enterpriseValue"),
            "total debt": self.load_number_value(ticker_info, "totalDebt"),
            "price to book": self.load_number_value(ticker_info, "priceToBook"),
            "peg ratio": self.load_number_value(ticker_info, "pegRatio"),
            "book value per share": self.load_number_value(ticker_info, "bookValue"),
            "trailing eps": self.load_number_value(ticker_info, "trailingEps"),
            "forward eps": self.load_number_value(ticker_info, "forwardEps"),
            "trailing pe": self.load_number_value(ticker_info, "trailingPE"),
            "forward pe": self.load_number_value(ticker_info, "forwardPE"),
            
            "dividend yield [%]": (dividend_yield * 100.0) if dividend_yield is not None else None,
            "dividend rate": self.load_number_value(ticker_info, "dividendRate"),
            
            "beta": self.load_number_value(ticker_info, "beta"),
        }
        return row

def fetch_from_yfinance(tickers: list[str]) -> list[dict]:
    """
    Main entry point to fetch data for a list of tickers using the TickerScraper class.
    """
    if not tickers:
        return []
    
    print(f"DEBUG: Starting YFinance threaded fetch for {len(tickers)} tickers...")
    scraper = TickerScraper(tickers, num_threads=10)
    parsed = [d for d in scraper.parsed_data if isinstance(d, dict) and d.get("ticker")]
    parsed_map = {d["ticker"]: d for d in parsed}
    print(f"DEBUG: YFinance fetch complete. Got {len(parsed)} results.")

    # IMPORTANT:
    # Always return an entry for every requested ticker, even if Yahoo failed.
    # This makes it obvious (in merged output) that fallback was attempted.
    out: list[dict] = []
    for t in tickers:
        if t in parsed_map:
            out.append(parsed_map[t])
        else:
            out.append(
                {
                    "ticker": t,
                    "name": None,
                    "industry": None,
                    "country": None,
                    "currency": None,
                    "current price": None,
                    "market cap": None,
                    "enterprise value": None,
                    "total debt": None,
                    "price to book": None,
                    "peg ratio": None,
                    "book value per share": None,
                    "trailing eps": None,
                    "forward eps": None,
                    "trailing pe": None,
                    "forward pe": None,
                    "dividend yield [%]": None,
                    "dividend rate": None,
                    "beta": None,
                }
            )

    return out
