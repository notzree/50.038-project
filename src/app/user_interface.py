import sys
import os
sys.path.append("..")

import streamlit as st 
import pandas as pd
import numpy as np
import pydeck as pdk

#Page Configuration
st.set_page_config(
    page_title="50.038 Project - Song Virality Trends",
    page_icon="🎵",
    layout="wide",
)

st.title("Song Virality Trends over Region and Time")
st.subheader("By Richard, Daniel, Maddie, and Maggie")

#Sidebar for user song input
st.sidebar.header("Search for a Song")

song_name = st.sidebar.text_input("Enter a song name to search for:", placeholder="Shape of You")
geo = st.sidebar.text_input("Region (ISO code)", placeholder="e.g. US, KR, AU")
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["today 1-month", "today 3-month", "today 12-month", "today 5-year"],
    index=1
)

run = st.sidebar.button("Run Analysis")


tab1, tab2, tab3 = st.tabs(["🗺️ Map", "📈 Time Series", "📊 Popularity Index"])

#need to connect all the dataframes to the respective tabs
with tab1:
    st.map("filtered_df")

with tab2:
    st.line_chart("time_df")

with tab3:
    st.dataframe("popularity_df")