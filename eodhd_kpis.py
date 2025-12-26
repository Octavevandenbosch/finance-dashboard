import requests
import pandas as pd
import numpy as np
import time
from datetime import date, timedelta

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
        raise RuntimeError(f"Error fetching tickers from Google Sheet: {e}") from e

# NOTE: Tickers are fetched dynamically inside `build_dataframe()` so changes in the Google Sheet
# are reflected immediately when you click "Fetch Latest Data".

def get_nested(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

REALTIME_BASE = "https://eodhd.com/api/real-time"
EOD_BASE = "https://eodhd.com/api/eod"

def _normalize_ticker(t: str) -> str:
    # Use the ticker exactly as provided by the Google Sheet (no suffixes appended).
    return str(t).strip()

def fetch_latest_eod_close(session: requests.Session, ticker: str, api_key: str) -> float | None:
    """
    Fallback when real-time is not available for a ticker/plan.
    Returns latest available close from EOD endpoint (not live, but usually good enough).
    """
    url = f"{EOD_BASE}/{ticker}"
    # Query a small recent window to avoid downloading full history and avoid relying on `limit=`.
    today = date.today()
    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": (today - timedelta(days=14)).isoformat(),
        "to": today.isoformat(),
    }
    r = session.get(url, params=params, timeout=20)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        # Pick the last item returned (latest date in range).
        close = data[-1].get("close")
        return float(close) if close is not None else None
    return None

def fetch_current_prices(session: requests.Session, tickers: list[str], api_key: str, chunk_size: int = 25) -> dict[str, float]:
    """
    Fetch current prices using EODHD real-time endpoint in chunks to avoid rate-limit issues.
    Endpoint supports: /real-time/{main}?s=OTHER1,OTHER2&api_token=...&fmt=json
    Returns dict keyed by ticker as provided (trimmed) -> float price.
    """
    tickers_norm = [_normalize_ticker(t) for t in tickers if str(t).strip()]
    prices: dict[str, float] = {}

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for group in chunks(tickers_norm, chunk_size):
        # Some EODHD endpoints require an exchange suffix (e.g. AAPL.US).
        # If we have any tickers with a dot, use one of them as the "main" ticker to reduce 404s.
        group = list(group)
        main_idx = next((i for i, t in enumerate(group) if "." in t), 0)
        main = group[main_idx]
        others = group[:main_idx] + group[main_idx + 1 :]
        params = {"api_token": api_key, "fmt": "json"}
        if others:
            params["s"] = ",".join(others)

        url = f"{REALTIME_BASE}/{main}"

        # Simple retry on 429/5xx
        for attempt in range(1, 4):
            r = session.get(url, params=params, timeout=20)
            if r.status_code in (403, 404):
                # 403: plan might not include real-time for this exchange/ticker group.
                # 404: the "main" ticker may be unknown / requires a suffix; we will fallback per-ticker to EOD.
                print(f"Price fetch unavailable (HTTP {r.status_code}) for {main}. Will fallback to EOD close where possible.")
                data = []
                break
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 0.5 * attempt
                print(f"Price fetch retry {attempt}/3 for {main} (HTTP {r.status_code}); waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        else:
            # exhausted retries
            continue

        data_list = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for item in data_list:
            if not isinstance(item, dict):
                continue
            code = item.get("code") or item.get("symbol")
            if not code:
                continue
            code = _normalize_ticker(code)
            val = item.get("close")
            if val is None:
                # some variants may include other naming
                val = item.get("last_trade_price") or item.get("lastTradePrice")
            if val is None:
                # Helpful debug if API returns an error payload
                if "message" in item:
                    print(f"Price API message for {code}: {item.get('message')}")
                continue
            try:
                prices[code] = float(val)
            except Exception:
                continue

    return prices

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
    # Require API key explicitly (Streamlit secrets).
    if not api_key:
        raise ValueError("Missing EODHD_API_KEY. Set it in Streamlit secrets.")
    current_api_key = str(api_key).strip()

    # Fetch tickers dynamically EVERY TIME the function is called
    current_tickers = fetch_tickers_from_sheet()
    
    if not current_tickers:
        raise RuntimeError("No tickers found in the Google Sheet (first column is empty).")

    with requests.Session() as session:
        # Fetch prices in bulk (chunked) to avoid rate limiting
        prices = fetch_current_prices(session, current_tickers, current_api_key, chunk_size=25)

        rows = []
        for t in current_tickers:
            api_ticker = _normalize_ticker(t)
            
            # Fetch fundamentals
            f = fetch_fundamentals(session, api_ticker, current_api_key)
            
            # Current price from bulk fetch
            price = prices.get(api_ticker)
            if price is None:
                # Fallback to latest EOD close if real-time isn't available for this ticker/plan
                price = fetch_latest_eod_close(session, api_ticker, current_api_key)
            
            # Try to get data from fundamentals if price is missing (fallback)
            # Market Cap fallback
            mcap = get_nested(f, ["Highlights", "MarketCapitalization"])
            
            row = {
                "ticker": t,
                "name": get_nested(f, ["General", "Name"]),
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
            rows.append(row)

    cols = [
        "ticker", "name", "industry", "country", "currency",
        "current price", "market cap", "price to book", "book value per share",
        "trailing eps", "forward eps", "trailing pe", "forward pe",
        "dividend yield [%]", "dividend rate", "beta"
    ]
    df = pd.DataFrame(rows)[cols]

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
    df = build_dataframe()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))
