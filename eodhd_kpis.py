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
# gid=997898919 is the specific tab ID you provided
SHEET_ID = "1oDb5xE0INQX78i5zmYa_pMahM1186RSU6pVBr0ea9t4"
SHEET_GID = "997898919" 
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"

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

def fetch_latest_delayed_prices(session: requests.Session, main: str, others: list[str], api_key: str) -> dict[str, float]:
    # This uses the real-time endpoint but allows fetching multiple tickers in one go via 's' parameter
    url = f"https://eodhd.com/api/real-time/{main}"
    params = {"api_token": api_key, "fmt": "json"}
    if others:
        params["s"] = ",".join(others)
        
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):  # single result
        # Handle case where API returns single dict
        code = data.get("code")
        price = data.get("close")
        if code and price is not None:
            return {code: float(price)}
        return {}
        
    # multiple results (list of dicts)
    out = {}
    for row in data:
        if row.get("code") and row.get("close") is not None:
            out[row["code"]] = float(row["close"])
    return out

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
        # Fetch prices in bulk using the new function
        prices = {}
        if current_tickers:
            try:
                # Prepare bulk request: first ticker is main, rest are 's'
                # Ensure .US suffix for API call
                api_main = current_tickers[0] if current_tickers[0].endswith(".US") else f"{current_tickers[0]}.US"
                api_others = [t if t.endswith(".US") else f"{t}.US" for t in current_tickers[1:]]
                
                # Split api_others into chunks of ~15 to avoid URL length limits/API limits if list is huge
                # EODHD real-time often supports fewer bulk tickers than delayed. Let's do chunking safe side.
                
                all_api_tickers = [api_main] + api_others
                
                # Chunking logic
                chunk_size = 15 # conservative for real-time
                for i in range(0, len(all_api_tickers), chunk_size):
                    chunk = all_api_tickers[i:i+chunk_size]
                    main_t = chunk[0]
                    other_t = chunk[1:]
                    
                    chunk_prices = fetch_latest_delayed_prices(session, main_t, other_t, current_api_key)
                    prices.update(chunk_prices)
                    
                print(f"DEBUG: Fetched {len(prices)} prices from EODHD.")
            except Exception as e:
                print(f"Error fetching prices: {e}")
                # Don't fail completely, just have empty prices
                prices = {}

        # 1. Build EODHD Rows
        eodhd_rows = []
        missing_tickers = [] # Track which tickers failed completely (no price or no fundamentals)

        for t in current_tickers:
            # Ensure ticker has .US for lookup
            api_ticker = t if t.endswith(".US") else f"{t}.US"
            
            # Fetch fundamentals
            f = fetch_fundamentals(session, api_ticker, current_api_key)
            
            # Get Price from our bulk fetched dictionary
            price = prices.get(api_ticker)
            
            # Check if we got valid data. If we have NO price and NO name/code from fundamentals, 
            # it implies the ticker might be unsupported by EODHD or wrong exchange.
            name = get_nested(f, ["General", "Name"]) or get_nested(f, ["General", "Code"])
            
            # Condition to use fallback: 
            # If (Price is None) OR (Name is None/Empty) -> Try YFinance
            if price is None and not name:
                missing_tickers.append(t)
                continue 

            # Try to get data from fundamentals if price is missing (fallback)
            # Market Cap fallback
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
            eodhd_rows.append(row)

        # 2. Fetch Fallback Data (YFinance)
        yf_rows = []
        if missing_tickers:
            print(f"DEBUG: {len(missing_tickers)} tickers missing from EODHD. Trying fallback: {missing_tickers}")
            yf_rows = fetch_from_yfinance(missing_tickers)
        
        # 3. Combine Results
        all_rows = eodhd_rows + yf_rows

    cols = [
        "ticker", "name", "industry", "country", "currency",
        "current price", "market cap", "price to book", "book value per share",
        "trailing eps", "forward eps", "trailing pe", "forward pe",
        "dividend yield [%]", "dividend rate", "beta"
    ]
    df = pd.DataFrame(all_rows)[cols]

    # --- Calculate Graham Number & Indicator ---
    # Ensure numeric types (coerce errors to NaN)
    for col in ["book value per share", "trailing eps", "current price"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Graham Number = Sqrt(22.5 * EPS * BVPS)
    # Logic: If EPS or BVPS is negative or NaN, Graham Number is undefined (NaN)
    def calculate_graham(row):
        eps = row["trailing eps"]
        bvps = row["book value per share"]
        
        if pd.isna(eps) or pd.isna(bvps) or eps < 0 or bvps < 0:
            return np.nan
        return round((22.5 * eps * bvps) ** 0.5, 2)

    df["graham"] = df.apply(calculate_graham, axis=1)

    # Graham Indicator = Graham Number - Last Price
    # (Positive means undervalued, Negative means overvalued relative to Graham Number)
    df["graham indicator"] = df["graham"] - df["current price"]

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
