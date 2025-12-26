import yfinance as yf
import pandas as pd

def fetch_yfinance_data(ticker: str) -> dict:
    """
    Fetches data for a single ticker using yfinance and maps it to the
    EODHD dictionary structure used in the main app.
    """
    try:
        # yfinance tickers might not need the .US suffix for US stocks, 
        # but often handles them fine. If fails, we might need logic to strip it.
        # Generally, "AAPL.US" works on EODHD but Yahoo expects "AAPL".
        # European stocks like "AIR.PA" are the same on both.
        
        # Simple heuristic: if it ends in .US, try stripping it for Yahoo
        yf_ticker = ticker
        if ticker.endswith(".US"):
            yf_ticker = ticker.replace(".US", "")
            
        stock = yf.Ticker(yf_ticker)
        # fast_info is sometimes faster/more reliable for price
        info = stock.info
        
        # If we didn't get valid info (e.g. no 'symbol' or 'regularMarketPrice'), return empty
        # Some keys are essential
        if not info: 
            return {}

        # Map yfinance info keys to EODHD-style keys expected by your app
        row = {
            "ticker": ticker, # Keep original ticker name for consistency
            "name": info.get("longName") or info.get("shortName"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "current price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market cap": info.get("marketCap"),
            "price to book": info.get("priceToBook"),
            "book value per share": info.get("bookValue"),
            "trailing eps": info.get("trailingEps"),
            "forward eps": info.get("forwardEps"),
            "trailing pe": info.get("trailingPE"),
            "forward pe": info.get("forwardPE"),
            
            # Dividends
            "dividend yield [%]": (info.get("dividendYield", 0) or 0) * 100 if info.get("dividendYield") else None,
            "dividend rate": info.get("dividendRate"),
            
            "beta": info.get("beta"),
        }
        
        return row
        
    except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
        return {}

def fetch_from_yfinance(tickers: list[str]) -> list[dict]:
    """
    Fetches data for a list of tickers using yfinance.
    """
    rows = []
    for t in tickers:
        data = fetch_yfinance_data(t)
        if data:
            rows.append(data)
        else:
            # Add an empty row or partial row so it appears in the table as missing
            rows.append({"ticker": t, "name": "Not Found"})
    return rows
