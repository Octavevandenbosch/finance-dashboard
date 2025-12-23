import streamlit as st
import pandas as pd
from eodhd_kpis import build_dataframe

# Set page configuration (tab title, layout)
st.set_page_config(
    page_title="EODHD Financial KPIs",
    page_icon="📈",
    layout="wide"
)

# Title and Description
st.title("📈 EODHD Financial KPI Dashboard")
st.markdown("This dashboard fetches real-time financial data using the EODHD API.")

# Button to refresh data
if st.button("Fetch Latest Data", type="primary"):
    with st.spinner("Fetching data from API..."):
        try:
            # Try to get API key from Streamlit secrets, otherwise None (uses default in script)
            api_key = st.secrets.get("EODHD_API_KEY")
            
            # Run the data fetching function
            df = build_dataframe(api_key=api_key)
            
            # Display success message
            st.success("Data fetched successfully!")
            
            # Metric cards (Optional: calculate average market cap for fun)
            if not df.empty:
                avg_pe = df['trailing pe'].mean()
                st.metric("Average Trailing P/E", f"{avg_pe:.2f}")

            # Display the DataFrame as an interactive table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "market cap": st.column_config.NumberColumn(format="$%d"),
                    "current price": st.column_config.NumberColumn(format="$%.2f"),
                    "graham": st.column_config.NumberColumn(format="%.2f", help="Graham Number = √(22.5 * EPS * Book Value)"),
                    "graham indicator": st.column_config.NumberColumn(format="%.2f", help="Graham Number - Current Price"),
                    "what about the graham?": st.column_config.TextColumn(help="Good if Graham Indicator > 0"),
                    "dividend yield [%]": st.column_config.NumberColumn(format="%.2f%%"),
                }
            )
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='eodhd_kpis.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Click the button above to load the data.")

