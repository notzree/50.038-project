import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.google_trends_collector import build_dataset

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Song Virality Trends",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Song Virality — Google Trends Explorer")
st.caption("Powered by Google Trends | Future: combined with audio model")

# --------------------------------------------------
# Sidebar — user inputs
# --------------------------------------------------
st.sidebar.header("Search Settings")

song_id    = st.sidebar.text_input("Song ID", placeholder="e.g. 001")
search_term = st.sidebar.text_input("Search Term", placeholder="e.g. Espresso Sabrina Carpenter")
geo        = st.sidebar.text_input("Region (ISO code)", placeholder="e.g. US, SG, GB — leave blank for worldwide")
timeframe  = st.sidebar.selectbox(
    "Timeframe",
    ["today 1-m", "today 3-m", "today 12-m", "today 5-y"],
    index=1
)

run = st.sidebar.button("Fetch Trends", type="primary")

# --------------------------------------------------
# Placeholder for future dataset merge
# --------------------------------------------------
# TODO: when audio model dataset is ready, load and merge it here
# audio_df = pd.read_pickle("data/audio/audio_dataset.pkl")
# merged_df = pd.merge(trends_df, audio_df, on="song_id")

# --------------------------------------------------
# Main content
# --------------------------------------------------
if run:
    if not search_term:
        st.warning("Please enter a search term.")
    else:
        with st.spinner("Fetching trends data... this may take a moment due to rate limits"):
            try:
                result = build_dataset(
                    song_id=song_id if song_id else "N/A",
                    search_term=search_term,
                    geo=geo if geo else "",
                    timeframe=timeframe
                )

                iot = result["interest_over_time"].reset_index()
                ibr = result["interest_by_region"].reset_index()

                # rename dynamic search term column
                for df in [iot, ibr]:
                    df.rename(columns={
                        col: "interest_score"
                        for col in df.columns
                        if col not in ["date", "region", "geoName", "song_id"]
                    }, inplace=True)

                # --------------------------------------------------
                # Row 1 — Interest over time
                # --------------------------------------------------
                st.subheader("📈 Interest Over Time")
                if not iot.empty:
                    fig_iot = px.line(
                        iot,
                        x="date",
                        y="interest_score",
                        title=f"Interest Over Time — {search_term}",
                        labels={"interest_score": "Interest (0–100)", "date": "Date"}
                    )
                    st.plotly_chart(fig_iot, use_container_width=True)
                else:
                    st.info("No interest over time data returned.")

                # --------------------------------------------------
                # Row 2 — Interest by region
                # --------------------------------------------------
                st.subheader("🌍 Interest By Region")
                if not ibr.empty:
                    col1, col2 = st.columns(2)

                    # Bar chart
                    with col1:
                        top_regions = ibr.sort_values("interest_score", ascending=False).head(20)
                        fig_bar = px.bar(
                            top_regions,
                            x="interest_score",
                            y="geoName",
                            orientation="h",
                            title="Top 20 Regions",
                            labels={"interest_score": "Interest (0–100)", "geoName": "Region"}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # Map
                    with col2:
                        fig_map = px.choropleth(
                            ibr,
                            locations="geoName",
                            locationmode="country names",
                            color="interest_score",
                            title="Interest by Country",
                            color_continuous_scale="Viridis",
                            labels={"interest_score": "Interest (0–100)"}
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.info("No interest by region data returned.")

                # --------------------------------------------------
                # Raw data expander (useful for debugging)
                # --------------------------------------------------
                with st.expander("View Raw Data"):
                    st.write("**Interest Over Time**")
                    st.dataframe(iot)
                    st.write("**Interest By Region**")
                    st.dataframe(ibr)

            except Exception as e:
                st.error(f"Error fetching data: {e}")