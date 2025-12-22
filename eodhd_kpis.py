import requests
import pandas as pd
import numpy as np
import os

# Default to environment variable, fallback to "DEMO"
DEFAULT_API_KEY = os.environ.get("EODHD_API_KEY", "DEMO")

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
TICKERS = fetch_tickers_from_sheet()

FUNDAMENTALS_URL = "https://eodhd.com/api/fundamentals/{ticker}"
QUOTE_URL = "https://eodhd.com/api/us-quote-delayed"

def get_nested(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def fetch_us_quote_delayed(session: requests.Session, tickers: list[str], api_key: str) -> dict:
    # This function is deprecated in favor of the real-time check requested by user
    return {}

def get_last_price(session, ticker: str, api_key: str) -> float:
    # Use real-time endpoint as requested
    base_url = "https://eodhd.com/api/real-time"
    url = f"{base_url}/{ticker}"
    try:
        r = session.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Real-time API returns a dict for single ticker
        return float(data["close"])
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
    # Use provided key, or environment variable, or default to "DEMO"
    current_api_key = api_key or os.environ.get("EODHD_API_KEY") or "DEMO"
    
    # Safety check for the revoked key
    if current_api_key == "69498255c6e8c5.83944142":
        print("Warning: Detected invalid/revoked demo key. Reverting to 'DEMO'.")
        current_api_key = "DEMO"

    # Fetch tickers dynamically EVERY TIME the function is called
    current_tickers = fetch_tickers_from_sheet()
    
    if not current_tickers:
        print("No tickers found.")
        return pd.DataFrame()

    with requests.Session() as session:
        # We will fetch prices individually inside the loop now
        quotes = {} 

        rows = []
        for t in current_tickers:
            # Ensure ticker has .US for the API call if missing (common issue)
            api_ticker = t if t.endswith(".US") else f"{t}.US"
            
            # Fetch fundamentals
            f = fetch_fundamentals(session, t, current_api_key)
            
            # Fetch Price using the new method
            price = get_last_price(session, api_ticker, current_api_key)
            
            # Try to get data from fundamentals if price is missing (fallback)
            # Market Cap fallback
            mcap = get_nested(f, ["Highlights", "MarketCapitalization"])
            
            row = {
                "ticker": t,
                "name": get_nested(f, ["General", "Name"]),
                "industry": get_nested(f, ["General", "Industry"]),
                "country": get_nested(f, ["General", "CountryName"]),
                "currency": get_nested(f, ["General", "CurrencyCode"]),
                "last price": price,
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
            rows.append(row)

    cols = [
        "ticker", "name", "industry", "country", "currency",
        "last price", "market cap", "price to book", "book value per share",
        "trailing eps", "forward eps", "trailing pe", "forward pe",
        "dividend yield [%]", "dividend rate", "beta"
    ]
    df = pd.DataFrame(rows)[cols]

    # --- Calculate Graham Number & Indicator ---
    # Ensure numeric types (coerce errors to NaN)
    for col in ["book value per share", "trailing eps", "last price"]:
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
    df["graham indicator"] = df["graham"] - df["last price"]

    # --- Graham Label ---
    # Good if Indicator > 0 (Undervalued), Bad if <= 0 (Overvalued), else "can't define"
    def get_graham_label(val):
        if pd.isna(val):
            return "can't define"
        return "good" if val > 0 else "bad"

    df["what about the graham?"] = df["graham indicator"].apply(get_graham_label)

    return df

if __name__ == "__main__":
    df = build_dataframe()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))
