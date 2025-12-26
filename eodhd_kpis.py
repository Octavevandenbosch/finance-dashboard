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

def _latest_balance_sheet_value(fundamentals: dict, candidate_fields: list[str]) -> float | None:
    """
    Try to extract the latest available balance sheet value for one of the candidate field names.
    EODHD fundamentals often expose it under Financials -> Balance_Sheet -> (yearly|quarterly) -> {date: {...}}.
    Returns float if found, else None.
    """
    bs_yearly = get_nested(fundamentals, ["Financials", "Balance_Sheet", "yearly"], default=None)
    bs_quarterly = get_nested(fundamentals, ["Financials", "Balance_Sheet", "quarterly"], default=None)

    def _pick_latest(bs: dict | None) -> dict | None:
        if not isinstance(bs, dict) or not bs:
            return None
        # Keys are typically date strings like "2025-09-30"
        latest_key = sorted(bs.keys())[-1]
        latest = bs.get(latest_key)
        return latest if isinstance(latest, dict) else None

    for bs in (bs_yearly, bs_quarterly):
        latest = _pick_latest(bs)
        if not latest:
            continue
        for field in candidate_fields:
            v = latest.get(field)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
    return None

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
            bvps = get_nested(f, ["Highlights", "BookValue"])
            eps = get_nested(f, ["Highlights", "EarningsShare"])
            enterprise_value = get_nested(f, ["Valuation", "EnterpriseValue"])
            peg_ratio = get_nested(f, ["Highlights", "PEGRatio"])
            total_debt = _latest_balance_sheet_value(
                f,
                candidate_fields=[
                    "totalDebt",
                    "shortLongTermDebtTotal",
                    "shortLongTermDebt",
                ],
            )

            row = {
                "ticker": t,
                "name": name,
                "industry": get_nested(f, ["General", "Industry"]),
                "country": get_nested(f, ["General", "CountryName"]),
                "currency": get_nested(f, ["General", "CurrencyCode"]),
                "current price": price,
                "price source": "EODHD" if price is not None else None,
                "yfinance attempted": False,
                "market cap": mcap,
                "enterprise value": enterprise_value,
                "total debt": total_debt,
                "price to book": get_nested(f, ["Valuation", "PriceBookMRQ"]),
                "peg ratio": peg_ratio,
                "book value per share": bvps,
                "trailing eps": eps,
                "forward eps": get_nested(f, ["Highlights", "EPSEstimateCurrentYear"]),
                "trailing pe": get_nested(f, ["Valuation", "TrailingPE"]),
                "forward pe": get_nested(f, ["Valuation", "ForwardPE"]),
                "dividend yield [%]": to_dividend_yield_pct(get_nested(f, ["Highlights", "DividendYield"])),
                "dividend rate": get_nested(f, ["Highlights", "DividendShare"]),
                "beta": get_nested(f, ["Technicals", "Beta"]),
            }

            rows_by_ticker[t] = row

            # Gap filler trigger:
            # - If EODHD can't provide a price, OR
            # - If fundamentals are effectively missing (no name and no key valuation inputs),
            # then try yfinance for that ticker.
            fundamentals_missing = (name is None) and (mcap is None) and (bvps is None) and (eps is None)
            if price is None or fundamentals_missing:
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
                base = rows_by_ticker.get(t, {"ticker": t})
                base["yfinance attempted"] = True
                price_before = base.get("current price")
                for k, v in yf_row.items():
                    if k == "ticker":
                        continue
                    if base.get(k) is None and v is not None:
                        base[k] = v
                # Record price source if YFinance filled the gap
                if price_before is None and base.get("current price") is not None:
                    base["price source"] = "YFinance"
                if base.get("price source") is None and base.get("current price") is None:
                    base["price source"] = "None"
                rows_by_ticker[t] = base

        all_rows = list(rows_by_ticker.values())

    cols = [
        "ticker", "name", "industry", "country", "currency",
        "current price", "market cap", "enterprise value", "total debt",
        "price to book", "peg ratio", "book value per share",
        "trailing eps", "forward eps", "trailing pe", "forward pe",
        "dividend yield [%]", "dividend rate", "beta",
        "price source", "yfinance attempted",
    ]
    # Build DataFrame WITHOUT enforcing column order here.
    # Column ordering is handled in `app.py` so UI/export are centralized.
    df = pd.DataFrame(all_rows)

    # Ensure required columns exist (so calculations below don't KeyError).
    # Missing ones become NaN.
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

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
