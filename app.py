import streamlit as st
import pandas as pd
from io import BytesIO
from eodhd_kpis import build_dataframe, SHEET_ID

# Single source of truth for column order (UI + CSV export)
COLUMN_ORDER = [
    "ticker", "name", "industry", "country", "currency",
    "current price", "market cap", "enterprise value", "total debt",
    "price to book", "peg ratio", "book value per share",
    "trailing eps", "forward eps", "trailing pe", "forward pe",
    "dividend yield [%]", "dividend rate", "beta",
    "graham", "graham indicator", "what about the graham?",
]

# Set page configuration (tab title, layout)
st.set_page_config(
    page_title="EODHD Financial KPIs",
    page_icon="📈",
    layout="wide"
)

# Title + quick link to ticker sheet (top-right)
left, right = st.columns([4, 1], vertical_alignment="top")
with left:
    st.title("📈 EODHD Financial KPI Dashboard")
with right:
    sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
    st.markdown(
        f"<div style='text-align: right; margin-top: 0.8rem;'>"
        f"<a href='{sheet_url}' target='_blank'>Edit tickers (Google Sheet)</a>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("This dashboard fetches real-time financial data using the EODHD API.")

# Require API key from Streamlit secrets (local `.streamlit/secrets.toml` or Streamlit Cloud Secrets)
api_key = st.secrets.get("EODHD_API_KEY")
if not api_key:
    st.error(
        "Missing `EODHD_API_KEY` in Streamlit secrets.\n\n"
        "Set it locally in `.streamlit/secrets.toml` or in Streamlit Cloud → App → Settings → Secrets."
    )
    st.stop()

# Button to refresh data
if "df_ordered" not in st.session_state:
    st.session_state.df_ordered = None

if st.button("Fetch Latest Data", type="primary"):
    with st.spinner("Fetching data from API..."):
        try:
            df = build_dataframe(api_key=api_key)
            st.session_state.df_ordered = df.reindex(columns=COLUMN_ORDER)
            st.success("Data fetched successfully!")
        except Exception as e:
            st.session_state.df_ordered = None
            st.error(f"An error occurred: {e}")

# Render results (persist across reruns, including when clicking Download)
df_ordered = st.session_state.df_ordered
if df_ordered is None:
    st.info("Click the button above to load the data.")
else:
    # Metric cards (Optional: calculate average market cap for fun)
    if not df_ordered.empty and "trailing pe" in df_ordered.columns:
        avg_pe = df_ordered["trailing pe"].mean()
        if pd.notna(avg_pe):
            st.metric("Average Trailing P/E", f"{avg_pe:.2f}")

    # Display formatting (keep underlying df numeric; only format for UI)
    df_display = df_ordered.copy()
    for big_col in ["market cap", "enterprise value", "total debt"]:
        if big_col in df_display.columns:
            df_display[big_col] = df_display[big_col].apply(
                lambda v: f"{float(v):,.2f}" if pd.notna(v) else ""
            )

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "market cap": st.column_config.TextColumn(help="Formatted with thousand separators"),
            "enterprise value": st.column_config.TextColumn(help="Formatted with thousand separators"),
            "total debt": st.column_config.TextColumn(help="Formatted with thousand separators"),
            "current price": st.column_config.NumberColumn(format="$%.2f"),
            "peg ratio": st.column_config.NumberColumn(format="%.2f"),
            "graham": st.column_config.NumberColumn(format="%.2f", help="Graham Number = √(22.5 * EPS * Book Value)"),
            "graham indicator": st.column_config.NumberColumn(format="%.2f", help="Graham Number - Current Price"),
            "what about the graham?": st.column_config.TextColumn(help="Good if Graham Indicator > 0"),
            "dividend yield [%]": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    # Download button (Excel) – uses persisted df_ordered
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_ordered.to_excel(writer, index=False, sheet_name="KPIs")
    buffer.seek(0)

    st.download_button(
        label="Download data as Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="eodhd_kpis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

