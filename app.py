import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Chicago Crime Analysis Platform",
    page_icon="🚔",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_app_data():
    return pd.read_parquet("Data/app_data.parquet")

df = load_app_data()

# ================= DISTRICT MAP =================
district_map = {
    1: "Central", 2: "Wentworth", 3: "Grand Crossing", 4: "South Chicago",
    5: "Calumet", 6: "Gresham", 7: "Englewood", 8: "Chicago Lawn",
    9: "Deering", 10: "Ogden", 11: "Harrison", 12: "Near West",
    13: "Jefferson Park", 14: "Shakespeare", 15: "Austin", 16: "Jefferson Park",
    17: "Albany Park", 18: "Near North", 19: "Town Hall", 20: "Lincoln",
    21: "Prairie", 22: "Morgan Park", 24: "Rogers Park", 25: "Grand Central"
}

df['District_Name'] = df['District'].map(district_map)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🎛️ Controls")
    st.success(f"{len(df):,} records loaded")

    years = sorted(df['Year'].unique())
    selected_years = st.multiselect("Year", years, default=years[-3:])

    crime_types = ['All'] + sorted(df['Primary Type'].unique())
    selected_crime = st.selectbox("Crime Type", crime_types)

    district_vals = ['All'] + sorted(df['District'].dropna().unique())
    district_names = ['All'] + sorted(df['District_Name'].dropna().unique())
    selected_district = st.selectbox("District", district_names)

    df_filtered = df.copy()

    if selected_years:
        df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
    if selected_crime != 'All':
        df_filtered = df_filtered[df_filtered['Primary Type'] == selected_crime]
    if selected_district != 'All':
        df_filtered = df_filtered[df_filtered['District_Name'] == selected_district]

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Map", "Temporal", "Clustering"
])

# ================= OVERVIEW =================
with tab1:
    st.header("📊 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", len(df_filtered))
    col2.metric("Crime Types", df_filtered['Primary Type'].nunique())
    col3.metric("Districts", df_filtered['District'].nunique())

    st.subheader("Top Crimes")
    top = df_filtered['Primary Type'].value_counts().head(10)

    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation='h'
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= MAP =================
with tab2:
    st.header("🗺️ Crime Heatmap")

    df_sample = df_filtered.sample(n=min(10000, len(df_filtered)), random_state=42)

    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    heat_data = [
        [row['Latitude'], row['Longitude']]
        for _, row in df_sample.iterrows()
    ]

    HeatMap(heat_data).add_to(m)
    folium_static(m)

# ================= TEMPORAL =================
with tab3:
    st.header("⏰ Temporal Patterns")

    hourly = df_filtered.groupby('Hour').size()

    fig = px.line(x=hourly.index, y=hourly.values)
    st.plotly_chart(fig, use_container_width=True)

# ================= CLUSTERING =================
with tab4:
    st.header("🔍 Clustering")

    cluster_method = st.radio(
        "Method",
        ["K-Means", "DBSCAN"],
        horizontal=True
    )

    df_sample = df_filtered.sample(n=min(10000, len(df_filtered)), random_state=42)

    cluster_col = 'kmeans_cluster' if cluster_method == "K-Means" else 'dbscan_cluster'

    fig = px.scatter_mapbox(
        df_sample,
        lat='Latitude',
        lon='Longitude',
        color=cluster_col,
        hover_data=['Primary Type', 'District_Name'],  # 👈 ADD THIS
        zoom=10,
        height=600
    )

    fig.update_layout(mapbox_style="carto-positron")

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", df_sample[cluster_col].nunique())
    col2.metric("Avg Size", int(len(df_sample)/df_sample[cluster_col].nunique()))
    col3.metric("Largest", df_sample[cluster_col].value_counts().iloc[0])

# ================= FOOTER =================
st.markdown("---")
st.markdown("Chicago Crime Analysis Platform 🚔")