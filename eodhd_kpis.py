import requests
import pandas as pd
import numpy as np
import os
import time
from yfinance_service import fetch_from_yfinance

# Default to environment variable, or None (enforce providing key)
DEFAULT_API_KEY = os.environ.get("EODHD_API_KEY")

# Google Sheet Configuration
# We use the 'export' format to get a clean CSV
# IMPORTANT:
# - If you include `gid=...`, you pin the import to one specific tab.
# - If you OMIT `gid`, Google returns the CSV for the FIRST (left-most) tab,
#   which lets you reorder tabs without changing code.
SHEET_ID = "1oDb5xE0INQX78i5zmYa_pMahM1186RSU6pVBr0ea9t4"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def fetch_tickers_from_sheet():
    try:
        print(f"Fetching tickers from Google Sheet...")
        # Read the CSV from the URL
        df = pd.read_csv(SHEET_URL)
        
        # We want the FIRST column, whatever its name is
        first_col_name = df.columns[0]
        print(f"Reading tickers from first column: '{first_col_name}'")
        
        raw_tickers = (
            df.iloc[:, 0]  # Select first column by position
            .dropna()      # Remove empty rows
            .astype(str)   # Ensure they are strings
            .str.strip()   # Remove whitespace
            .tolist()
        )
        
        # Filter out empty strings if any remain
        raw_tickers = [t for t in raw_tickers if t]
        
        print(f"Successfully loaded {len(raw_tickers)} tickers: {raw_tickers[:5]}...")
        return raw_tickers
    except Exception as e:
        print(f"Error fetching tickers from sheet: {e}. Using defaults.")
        return ["AAPL.US", "TSLA.US", "AMZN.US"]

# Load tickers dynamically
# We removed the global TICKERS variable to force fetching fresh tickers every time

FUNDAMENTALS_URL = "https://eodhd.com/api/fundamentals/{ticker}"
QUOTE_URL = "https://eodhd.com/api/real-time/{ticker}"

def get_nested(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def fetch_latest_delayed_price(session: requests.Session, ticker: str, api_key: str) -> float | None:
    url = f"https://eodhd.com/api/real-time/{ticker}"
    params = {"api_token": api_key, "fmt": "json"}
    try:
        r = session.get(url, params=params, timeout=20)
        if r.status_code in (403, 404):
            return None
        r.raise_for_status()

        data = r.json()
        # Single-ticker response is a dict with fields like: code, timestamp, open/high/low/close...
        close = data.get("close") if isinstance(data, dict) else None
        return float(close) if close is not None else None
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return None

def fetch_fundamentals(session, ticker: str, api_key: str) -> dict:
    url = f"https://eodhd.com/api/fundamentals/{ticker}"
    try:
        r = session.get(
            url,
            params={"api_token": api_key.strip(), "fmt": "json"},
            timeout=20,
        )
        if r.status_code == 403:
            print(f"Warning: Access forbidden for {ticker} (403). Check API key or plan limits.")
            return {}
            
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return {}

def to_dividend_yield_pct(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if x <= 1.0 else x

def build_dataframe(api_key=None):
    # Use provided key, or environment variable
    # We removed the fallback to "DEMO" to enforce using the paid key
    current_api_key = api_key or os.environ.get("EODHD_API_KEY")
    
    if not current_api_key:
        print("Error: No API key provided. Please check secrets.toml or environment variables.")
        return pd.DataFrame()

    # Fetch tickers dynamically EVERY TIME the function is called
    current_tickers = fetch_tickers_from_sheet()
    
    if not current_tickers:
        print("No tickers found.")
        return pd.DataFrame()

    with requests.Session() as session:
        # 1) Build EODHD rows, fetching price via the *working* single-ticker function.
        # IMPORTANT: We never append ".US" or otherwise modify tickers.
        rows_by_ticker: dict[str, dict] = {}
        tickers_needing_yf: list[str] = []

        for t in current_tickers:
            api_ticker = t  # use exactly as provided

            f = fetch_fundamentals(session, api_ticker, current_api_key)
            price = fetch_latest_delayed_price(session, api_ticker, current_api_key)

            name = get_nested(f, ["General", "Name"]) or get_nested(f, ["General", "Code"])
            mcap = get_nested(f, ["Highlights", "MarketCapitalization"])

            row = {
                "ticker": t,
                "name": name,
                "industry": get_nested(f, ["General", "Industry"]),
                "country": get_nested(f, ["General", "CountryName"]),
                "currency": get_nested(f, ["General", "CurrencyCode"]),
                "current price": price,
                "market cap": mcap,
                "price to book": get_nested(f, ["Valuation", "PriceBookMRQ"]),
                "book value per share": get_nested(f, ["Highlights", "BookValue"]),
                "trailing eps": get_nested(f, ["Highlights", "EarningsShare"]),
                "forward eps": get_nested(f, ["Highlights", "EPSEstimateCurrentYear"]),
                "trailing pe": get_nested(f, ["Valuation", "TrailingPE"]),
                "forward pe": get_nested(f, ["Valuation", "ForwardPE"]),
                "dividend yield [%]": to_dividend_yield_pct(get_nested(f, ["Highlights", "DividendYield"])),
                "dividend rate": get_nested(f, ["Highlights", "DividendShare"]),
                "beta": get_nested(f, ["Technicals", "Beta"]),
            }

            rows_by_ticker[t] = row

            # If EODHD can't provide a price, we try yfinance for that ticker (gap filler).
            if price is None:
                tickers_needing_yf.append(t)

        # 2) Fetch fallback data (YFinance) for tickers missing EODHD price
        if tickers_needing_yf:
            print(
                f"DEBUG: {len(tickers_needing_yf)} tickers missing EODHD price. "
                f"Trying YFinance fallback for: {tickers_needing_yf}"
            )
            yf_rows = fetch_from_yfinance(tickers_needing_yf)
            yf_map = {r.get("ticker"): r for r in yf_rows if isinstance(r, dict) and r.get("ticker")}

            # Merge: fill missing fields (especially current price) from yfinance
            for t in tickers_needing_yf:
                yf_row = yf_map.get(t)
                if not yf_row:
                    continue
                base = rows_by_ticker.get(t, {"ticker": t})
                for k, v in yf_row.items():
                    if k == "ticker":
                        continue
                    if base.get(k) is None and v is not None:
                        base[k] = v
                rows_by_ticker[t] = base

        all_rows = list(rows_by_ticker.values())

    cols = [
        "ticker", "name", "industry", "country", "currency",
        "current price", "market cap", "price to book", "book value per share",
        "trailing eps", "forward eps", "trailing pe", "forward pe",
        "dividend yield [%]", "dividend rate", "beta"
    ]
    df = pd.DataFrame(all_rows)[cols]

    # --- Calculate Graham Number & Indicator ---
    # Ensure numeric types (coerce errors to NaN). This fixes the "float and str" error.
    numeric_cols = ["book value per share", "trailing eps", "current price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Graham Number = Sqrt(22.5 * EPS * BVPS)
    def calculate_graham(row):
        eps = row["trailing eps"]
        bvps = row["book value per share"]
        
        # Check for NaN
        if pd.isna(eps) or pd.isna(bvps):
            return np.nan
        # Check for negative values (cannot sqrt negative)
        if eps < 0 or bvps < 0:
            return np.nan
            
        try:
            return round((22.5 * eps * bvps) ** 0.5, 2)
        except Exception:
            return np.nan

    df["graham"] = df.apply(calculate_graham, axis=1)

    # Graham Indicator = Graham Number - Current Price
    def calculate_indicator(row):
        graham = row["graham"]
        price = row["current price"]
        
        if pd.isna(graham) or pd.isna(price):
            return np.nan
        return graham - price

    df["graham indicator"] = df.apply(calculate_indicator, axis=1)

    # --- Graham Label ---
    # Good if Indicator > 0 (Undervalued), Bad if <= 0 (Overvalued), else "can't define"
    def get_graham_label(val):
        if pd.isna(val):
            return "can't define"
        return "good" if val > 0 else "bad"

    df["what about the graham?"] = df["graham indicator"].apply(get_graham_label)

    return df

if __name__ == "__main__":
    # For local testing, ensure EODHD_API_KEY is set or passed
    df = build_dataframe()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))
