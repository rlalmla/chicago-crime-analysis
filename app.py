import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Chicago Crime Intelligence Platform",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --danger-color: #d62728;
        --success-color: #2ca02c;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        background-color: #f0f7ff;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff7f0e;
        background-color: #fff7ed;
        margin: 1rem 0;
    }
    
    .danger-box {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d62728;
        background-color: #fff0f0;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_app_data():
    df = pd.read_parquet("Data/app_data.parquet")
    
    # Check what date columns exist and convert them
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    if date_columns:
        # Use the first date column found
        date_col = date_columns[0]
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        # Create a dummy date column if none exists
        df['Date'] = pd.NaT
    
    return df

df = load_app_data()

# Generate DayName if missing
# Ensure Date exists properly
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Create DayName safely
if 'DayName' not in df.columns:
    if 'Date' in df.columns:
        df['DayName'] = df['Date'].dt.day_name()


# Debug: Show columns (remove after fixing)
# st.write("Available columns:", df.columns.tolist())

# ================= DISTRICT MAP =================
district_map = {
    1: "Central", 2: "Wentworth", 3: "Grand Crossing", 4: "South Chicago",
    5: "Calumet", 6: "Gresham", 7: "Englewood", 8: "Chicago Lawn",
    9: "Deering", 10: "Ogden", 11: "Harrison", 12: "Near West",
    13: "Jefferson Park", 14: "Shakespeare", 15: "Austin", 16: "Jefferson Park",
    17: "Albany Park", 18: "Near North", 19: "Town Hall", 20: "Lincoln",
    21: "Prairie", 22: "Morgan Park", 24: "Rogers Park", 25: "Grand Central"
}

if 'District' in df.columns:
    df['District_Name'] = df['District'].map(district_map)
else:
    df['District_Name'] = 'Unknown'

# ================= HEADER =================
st.markdown('<h1 class="main-header">🚔 Chicago Crime Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Analytics & Predictive Insights for Public Safety</p>', unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🎛️ Filter Controls")
    
    st.markdown("---")
    
    # Dataset info
    st.success(f"📊 **{len(df):,}** total records")
    
    st.markdown("---")
    
    # Year filter - only if Year column exists
    if 'Year' in df.columns:
        years = sorted(df['Year'].unique())
        selected_years = st.multiselect(
            "📅 Select Years",
            years,
            default=years[-3:] if len(years) >= 3 else years,
            help="Choose one or more years to analyze"
        )
    else:
        selected_years = []
    
    # Crime type filter
    if 'Primary Type' in df.columns:
        crime_types = ['All'] + sorted(df['Primary Type'].unique())
        selected_crime = st.selectbox(
            "🔍 Crime Type",
            crime_types,
            help="Filter by specific crime category"
        )
    else:
        selected_crime = 'All'
    
    # District filter
    if 'District_Name' in df.columns:
        district_names = ['All'] + sorted(df['District_Name'].dropna().unique())
        selected_district = st.selectbox(
            "📍 District",
            district_names,
            help="Filter by police district"
        )
    else:
        selected_district = 'All'
    
    # Arrest filter - only if Arrest column exists
    if 'Arrest' in df.columns:
        arrest_filter = st.radio(
            "⚖️ Arrest Status",
            ["All", "Arrested", "Not Arrested"],
            horizontal=True
        )
    else:
        arrest_filter = "All"
    
    st.markdown("---")
    
    st.markdown("---")
    st.caption("Data Source: City of Chicago Data Portal")

# ================= APPLY FILTERS =================
df_filtered = df.copy()

if selected_years and 'Year' in df.columns:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
if selected_crime != 'All' and 'Primary Type' in df.columns:
    df_filtered = df_filtered[df_filtered['Primary Type'] == selected_crime]
if selected_district != 'All' and 'District_Name' in df.columns:
    df_filtered = df_filtered[df_filtered['District_Name'] == selected_district]
if 'Arrest' in df.columns:
    if arrest_filter == "Arrested":
        df_filtered = df_filtered[df_filtered['Arrest'] == True]
    elif arrest_filter == "Not Arrested":
        df_filtered = df_filtered[df_filtered['Arrest'] == False]

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
    "📊 Overview", "🗺️ Geographic Analysis", "⏰ Temporal Patterns", "🔍 Clustering Analysis", "📈 Insights & Trends",
    "🧠 Dimensionality Reduction"])

# ================= TAB 1: OVERVIEW =================
with tab1:
    st.header("📊 Crime Statistics Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Incidents",
            value=f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):,}" if len(df_filtered) != len(df) else None
        )
    
    with col2:
        if 'Arrest' in df_filtered.columns and len(df_filtered) > 0:
            arrest_rate = (df_filtered['Arrest'].sum() / len(df_filtered) * 100)
            st.metric(
                label="Arrest Rate",
                value=f"{arrest_rate:.1f}%"
            )
        else:
            pass
    
    with col3:
        if 'Primary Type' in df_filtered.columns:
            st.metric(
                label="Crime Types",
                value=df_filtered['Primary Type'].nunique()
            )
        else:
            st.metric(label="Crime Types", value="N/A")
    
    with col4:
        if 'District' in df_filtered.columns:
            st.metric(
                label="Districts Affected",
                value=df_filtered['District'].nunique()
            )
        else:
            st.metric(label="Districts Affected", value="N/A")
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'Primary Type' in df_filtered.columns:
            st.subheader("🔝 Top 15 Crime Categories")
            top_crimes = df_filtered['Primary Type'].value_counts().head(15)
            
            fig = px.bar(
                x=top_crimes.values,
                y=top_crimes.index,
                orientation='h',
                labels={'x': 'Number of Incidents', 'y': 'Crime Type'},
                color=top_crimes.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                showlegend=False,
                height=500,
                xaxis_title="Number of Incidents",
                yaxis_title="",
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True, key="overview_top_crimes")
    
    with col2:
        if 'District_Name' in df_filtered.columns:
            st.subheader("📍 Top Districts")
            top_districts = df_filtered['District_Name'].value_counts().head(10)
            
            fig = px.pie(
                values=top_districts.values,
                names=top_districts.index,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Crime by Year
    if 'Year' in df_filtered.columns:
        st.subheader("📅 Crime Trends Over Years")
        yearly_crimes = df_filtered.groupby('Year').size().reset_index(name='Count')
        
        fig = px.line(
            yearly_crimes,
            x='Year',
            y='Count',
            markers=True,
            labels={'Count': 'Number of Crimes'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=3, marker=dict(size=10))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="overview_district_pie")

# ================= TAB 2: MAP =================
with tab2:
    st.header("🗺️ Geographic Crime Analysis")
    
    # Check if location data exists
    if 'Latitude' not in df_filtered.columns or 'Longitude' not in df_filtered.columns:
        st.error("⚠️ Location data (Latitude/Longitude) not available in the dataset.")
    else:
        # Map type selector
        map_type = st.radio(
            "Visualization Type",
            ["Heatmap", "Cluster Map", "Scatter Plot"],
            horizontal=True
        )
        
        # Sample for performance
        sample_size = min(15000, len(df_filtered))
        df_sample = df_filtered.sample(n=sample_size, random_state=42)
        
        # Remove rows with missing coordinates
        df_sample = df_sample.dropna(subset=['Latitude', 'Longitude'])
        
        if len(df_sample) == 0:
            st.warning("⚠️ No valid location data available after filtering.")
        else:
            if map_type == "Heatmap":
                st.info(f"📍 Showing heatmap of {len(df_sample):,} incidents")
                
                m = folium.Map(
                    location=[41.8781, -87.6298],
                    zoom_start=11,
                    tiles='CartoDB positron'
                )
                
                heat_data = [
                    [row['Latitude'], row['Longitude']]
                    for _, row in df_sample.iterrows()
                ]
                
                HeatMap(
                    heat_data,
                    min_opacity=0.2,
                    max_zoom=13,
                    radius=15,
                    blur=25,
                    gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}
                ).add_to(m)
                
                folium_static(m, width=1200, height=600)
            
            elif map_type == "Cluster Map":
                st.info(f"📍 Showing clustered view of {len(df_sample):,} incidents")
                
                m = folium.Map(
                    location=[41.8781, -87.6298],
                    zoom_start=11,
                    tiles='CartoDB positron'
                )
                
                marker_cluster = MarkerCluster().add_to(m)
                
                for idx, row in df_sample.head(1000).iterrows():  # Limit markers for performance
                    crime_type = row.get('Primary Type', 'Unknown')
                    district = row.get('District_Name', 'Unknown')
                    
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=f"<b>{crime_type}</b><br>{district}",
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(marker_cluster)
                
                folium_static(m, width=1200, height=600)
            
            else:  # Scatter Plot
                st.info(f"📍 Interactive scatter plot of {len(df_sample):,} incidents")
                
                hover_cols = ['Primary Type', 'District_Name']
                if 'Date' in df_sample.columns:
                    hover_cols.append('Date')
                if 'Arrest' in df_sample.columns:
                    hover_cols.append('Arrest')
                
                # Keep only columns that exist
                hover_cols = [col for col in hover_cols if col in df_sample.columns]
                
                fig = px.scatter_mapbox(
                    df_sample,
                    lat='Latitude',
                    lon='Longitude',
                    color='Primary Type' if 'Primary Type' in df_sample.columns else None,
                    hover_data=hover_cols,
                    zoom=10,
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig.update_layout(
                    mapbox_style="carto-positron",
                    margin={"r":0,"t":0,"l":0,"b":0}
                )
                
                st.plotly_chart(fig, use_container_width=True, key="map_scatter")
            
            # Geographic statistics
            st.markdown("---")
            st.subheader("📊 Geographic Distribution Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'District_Name' in df_filtered.columns and len(df_filtered) > 0:
                    most_affected = df_filtered['District_Name'].mode()[0] if len(df_filtered['District_Name'].mode()) > 0 else "N/A"
                    st.metric("Most Affected District", most_affected)
                else:
                    st.metric("Most Affected District", "N/A")
            
            with col2:
                unique_locs = df_filtered[['Latitude', 'Longitude']].drop_duplicates().shape[0]
                st.metric("Unique Locations", f"{unique_locs:,}")
            
            with col3:
                avg_lat = df_filtered['Latitude'].mean()
                avg_lon = df_filtered['Longitude'].mean()
                st.metric("Crime Center", f"{avg_lat:.4f}, {avg_lon:.4f}")

# ================= TAB 3: TEMPORAL =================
with tab3:
    st.header("⏰ Temporal Crime Patterns")

    # ================= COMPUTE =================
    hourly = None
    daily = None

    if 'Hour' in df_filtered.columns:
        hourly = df_filtered.groupby('Hour').size().reset_index(name='Count')

    if 'DayName' in df_filtered.columns and df_filtered['DayName'].notna().sum() > 0:
        daily = df_filtered['DayName'].value_counts().reset_index()
        daily.columns = ['Day', 'Count']

    # ================= CHART ROW =================
    col1, col2 = st.columns(2)

    with col1:
        if hourly is not None:
            st.subheader("📊 Crimes by Hour")

            fig = px.bar(
                hourly,
                x='Hour',
                y='Count',
                labels={'Hour': 'Hour of Day', 'Count': 'Number of Crimes'}
            )

            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True, key="temporal_hour")

    with col2:
        if daily is not None:
            st.subheader("📊 Crimes by Day")

            fig = px.bar(
                daily,
                x='Day',
                y='Count'
            )

            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True, key="temporal_day")

    # ================= INSIGHTS ROW =================
    st.markdown("### 🔥 Key Temporal Insights")

    col3, col4 = st.columns(2)

    with col3:
        if hourly is not None and hourly['Count'].max() > 0:
            peak_hour = hourly.loc[hourly['Count'].idxmax(), 'Hour']
            peak_hour_value = hourly['Count'].max()

            st.success(f"🔥 Peak Crime Hour: {int(peak_hour)}:00 ({peak_hour_value:,} incidents)")

    with col4:
        if daily is not None and daily['Count'].max() > 0:
            peak_day = daily.loc[daily['Count'].idxmax(), 'Day']
            peak_value = daily['Count'].max()

            st.info(f"🔥 Peak Crime Day: {peak_day} ({peak_value:,} incidents)")

    # Monthly trends
    if 'Month' in df_filtered.columns:
        st.subheader("📆 Monthly Crime Trends")
        
        monthly = df_filtered.groupby('Month').size().reset_index(name='Count')
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly['Month_Name'] = monthly['Month'].map(month_names)
        
        fig = px.line(
            monthly,
            x='Month_Name',
            y='Count',
            markers=True,
            labels={'Month_Name': 'Month', 'Count': 'Number of Crimes'}
        )
        fig.update_traces(line_color='#2ca02c', line_width=3, marker=dict(size=12))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="temporal_month")
    
    # Weekend vs Weekday
    st.markdown("---")
    if 'IsWeekend' in df_filtered.columns:
        st.subheader("📊 Weekend vs Weekday Comparison")
        
        weekend_data = df_filtered.groupby('IsWeekend').size().reset_index(name='Count')
        weekend_data['Type'] = weekend_data['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                weekend_data,
                values='Count',
                names='Type',
                hole=0.4,
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="temporal_weekend")
        
        with col2:
            weekday_count = weekend_data[weekend_data['Type'] == 'Weekday']['Count'].values[0] if len(weekend_data[weekend_data['Type'] == 'Weekday']) > 0 else 0
            weekend_count = weekend_data[weekend_data['Type'] == 'Weekend']['Count'].values[0] if len(weekend_data[weekend_data['Type'] == 'Weekend']) > 0 else 0
            
            st.metric("Weekday Crimes", f"{weekday_count:,}")
            st.metric("Weekend Crimes", f"{weekend_count:,}")
            
            if weekday_count > 0:
                ratio = weekend_count / weekday_count
                st.metric("Weekend/Weekday Ratio", f"{ratio:.2f}x")

# ================= TAB 4: CLUSTERING =================
with tab4:
    st.header("🔍 Crime Hotspot Clustering Analysis")
    
    st.markdown("""
    <div class="info-box">
    <b>ℹ️ About Clustering</b><br>
    Clustering algorithms identify crime hotspots by grouping geographically similar incidents.
    This helps law enforcement allocate resources more effectively.
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering method selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        cluster_method = st.radio(
            "Select Clustering Algorithm",
            ["K-Means", "DBSCAN"],
            horizontal=True,
            help="K-Means creates fixed number of clusters, DBSCAN identifies density-based clusters"
        )
    
    with col2:
        sample_size = st.selectbox(
            "Sample Size",
            [5000, 10000, 15000, 20000],
            index=1
        )
    
    # Sample data
    df_sample = df_filtered.sample(n=min(sample_size, len(df_filtered)), random_state=42)
    
    # Select cluster column
    cluster_col = 'kmeans_cluster' if cluster_method == "K-Means" else 'dbscan_cluster'
    
    # Check if column exists
    if cluster_col not in df_sample.columns:
        st.warning(f"⚠️ {cluster_method} clustering data not available in the dataset.")
        st.info("Available columns: " + ", ".join(df_sample.columns.tolist()))
    elif 'Latitude' not in df_sample.columns or 'Longitude' not in df_sample.columns:
        st.error("⚠️ Location data required for clustering visualization.")
    else:
        # Remove rows with missing coordinates
        df_sample = df_sample.dropna(subset=['Latitude', 'Longitude', cluster_col])
        
        if len(df_sample) == 0:
            st.warning("⚠️ No valid clustering data available after filtering.")
        else:
            # Cluster Map
            st.subheader("🗺️ Cluster Visualization")
            
            hover_cols = [cluster_col]
            if 'Primary Type' in df_sample.columns:
                hover_cols.append('Primary Type')
            if 'District_Name' in df_sample.columns:
                hover_cols.append('District_Name')
            if 'Date' in df_sample.columns:
                hover_cols.append('Date')
            
            fig = px.scatter_mapbox(
                df_sample,
                lat='Latitude',
                lon='Longitude',
                color=cluster_col,
                hover_data=hover_cols,
                zoom=10,
                height=600,
                color_continuous_scale='Viridis' if cluster_method == "K-Means" else 'Plasma'
            )
            
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True, key="cluster_map")
            
            # Cluster Statistics
            st.markdown("---")
            st.subheader("📊 Cluster Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            n_clusters = df_sample[cluster_col].nunique()
            avg_size = int(len(df_sample) / n_clusters) if n_clusters > 0 else 0
            largest_cluster = df_sample[cluster_col].value_counts().iloc[0] if len(df_sample) > 0 else 0
            smallest_cluster = df_sample[cluster_col].value_counts().iloc[-1] if len(df_sample) > 0 else 0
            
            col1.metric("Total Clusters", n_clusters)
            col2.metric("Average Cluster Size", avg_size)
            col3.metric("Largest Cluster", largest_cluster)
            col4.metric("Smallest Cluster", smallest_cluster)
            
            # Cluster distribution
            st.subheader("📊 Cluster Size Distribution")
            
            cluster_counts = df_sample[cluster_col].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig = px.bar(
                cluster_counts,
                x='Cluster',
                y='Count',
                labels={'Cluster': 'Cluster ID', 'Count': 'Number of Incidents'},
                color='Count',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="cluster_distribution")
            
            # Top crimes per cluster (if Primary Type exists)
            if 'Primary Type' in df_sample.columns:
                st.markdown("---")
                st.subheader("🔍 Top Crimes by Cluster")
                
                selected_cluster = st.selectbox(
                    "Select Cluster to Analyze",
                    sorted(df_sample[cluster_col].unique())
                )
                
                cluster_data = df_sample[df_sample[cluster_col] == selected_cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    top_crimes_cluster = cluster_data['Primary Type'].value_counts().head(10)
                    
                    fig = px.bar(
                        x=top_crimes_cluster.values,
                        y=top_crimes_cluster.index,
                        orientation='h',
                        labels={'x': 'Count', 'y': 'Crime Type'},
                        color=top_crimes_cluster.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True, key="cluster_top_crimes")
                
                with col2:
                    st.metric("Cluster Size", len(cluster_data))
                    
                    if 'Arrest' in cluster_data.columns:
                        arrest_rate = (cluster_data['Arrest'].sum() / len(cluster_data) * 100)
                        st.metric("Arrest Rate", f"{arrest_rate:.1f}%")
                    
                    st.metric("Unique Crime Types", cluster_data['Primary Type'].nunique())
                    
                    most_common = cluster_data['Primary Type'].mode()[0] if len(cluster_data) > 0 else "N/A"
                    st.metric("Most Common", most_common)

# ================= TAB 5: INSIGHTS =================
with tab5:
    st.header("📈 Key Insights & Trends")
    
    # Crime trends (if Year column exists)
    if 'Year' in df_filtered.columns:
        st.subheader("📊 Year-over-Year Trends")
        
        agg_dict = {'ID': 'count'} if 'ID' in df_filtered.columns else {df_filtered.columns[0]: 'count'}
        
        if 'Arrest' in df_filtered.columns:
            agg_dict['Arrest'] = 'sum'
        
        yearly_data = df_filtered.groupby('Year').agg(agg_dict).reset_index()
        
        if 'ID' in yearly_data.columns:
            yearly_data.columns = ['Year', 'Total_Crimes'] + (['Total_Arrests'] if 'Arrest' in df_filtered.columns else [])
        else:
            yearly_data.columns = ['Year', 'Total_Crimes'] + (['Total_Arrests'] if 'Arrest' in df_filtered.columns else [])
        
        if 'Total_Arrests' in yearly_data.columns:
            yearly_data['Arrest_Rate'] = (yearly_data['Total_Arrests'] / yearly_data['Total_Crimes'] * 100).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=yearly_data['Year'],
            y=yearly_data['Total_Crimes'],
            name='Total Crimes',
            marker_color='indianred'
        ))
        
        if 'Arrest_Rate' in yearly_data.columns:
            fig.add_trace(go.Scatter(
                x=yearly_data['Year'],
                y=yearly_data['Arrest_Rate'],
                name='Arrest Rate (%)',
                yaxis='y2',
                line=dict(color='royalblue', width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            yaxis=dict(title='Number of Crimes'),
            yaxis2=dict(title='Arrest Rate (%)', overlaying='y', side='right') if 'Arrest_Rate' in yearly_data.columns else None,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key="overview_year_trend")
    
    st.markdown("---")
    
    # Key findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Findings")
        
        most_common_crime = df_filtered['Primary Type'].mode()[0] if 'Primary Type' in df_filtered.columns and len(df_filtered) > 0 else "N/A"
        most_affected_district = df_filtered['District_Name'].mode()[0] if 'District_Name' in df_filtered.columns and len(df_filtered) > 0 else "N/A"
        
        if 'Hour' in df_filtered.columns and len(df_filtered) > 0:
            hourly = df_filtered.groupby('Hour').size().reset_index(name='Count')
            fig = px.bar(
            hourly,
            x='Hour',
            y='Count',
            labels={'Hour': 'Hour of Day', 'Count': 'Number of Crimes'},
            color='Count',
            color_continuous_scale='YlOrRd'
        )

        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
      
        arrest_rate_text = "N/A"
        if 'Arrest' in df_filtered.columns and len(df_filtered) > 0:
            arrest_rate_text = f"{(df_filtered['Arrest'].sum() / len(df_filtered) * 100):.1f}%"
        
        st.markdown(f"""
        <div class="danger-box">
        <b>🔴 Most Common Crime:</b> {most_common_crime}<br>
        <b>📍 Most Affected District:</b> {most_affected_district}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💡 Recommendations")
        
        st.markdown("### 💡 Recommendations")

        st.write("""
            - Increase patrol deployment between identified peak hours
            - Allocate more officers to consistently high-crime districts
            - Monitor high-density clusters identified by DBSCAN
            - Prioritize surveillance in recurring hotspot zones
            - Focus preventive action on frequently occurring crime types
            """)
    
    # Data quality
    st.markdown("---")
    st.markdown("### 📌 Key Takeaways")

    st.write("""
        - Crime activity peaks during specific evening hours indicating high-risk periods
        - Certain districts consistently show higher crime concentration
        - Clustering reveals crime is not uniformly distributed but forms hotspots
        - Temporal patterns suggest predictable crime behavior trends
        """)    
  
    st.markdown("### 🚔 Real-World Impact")

    st.write("""
        - Improves response time in high-risk areas
        - Reduces crime through proactive monitoring
        - Enables efficient allocation of limited police resources
        - Supports data-driven decision making in law enforcement
        """)
    st.subheader("📋 Data Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    missing_lat = df_filtered['Latitude'].isna().sum() if 'Latitude' in df_filtered.columns else 0
    missing_lon = df_filtered['Longitude'].isna().sum() if 'Longitude' in df_filtered.columns else 0
    missing_district = df_filtered['District'].isna().sum() if 'District' in df_filtered.columns else 0
    
    total_records = len(df_filtered)
    complete_pct = ((total_records - missing_lat) / total_records * 100) if total_records > 0 else 0
    
    col1.metric("Complete Records", f"{complete_pct:.1f}%")
    col2.metric("Missing Locations", f"{missing_lat:,}")
    col3.metric("Missing Districts", f"{missing_district:,}")
    col4.metric("Data Completeness", f"{complete_pct:.1f}%")

    st.markdown("---")
    st.markdown("### 🔬 Clustering Model Evaluation")

    st.write("""
        Three clustering algorithms were implemented and evaluated:

        - **K-Means**: Efficient and scalable, used for primary clustering
        - **DBSCAN**: Identifies dense crime hotspots and filters noise
        - **Hierarchical Clustering (Ward linkage)**: Used to analyze nested relationships between crime locations

        **Evaluation Results:**
        - Hierarchical clustering achieved a silhouette score of ~0.31
        - This indicates weaker cluster separation compared to other models
        - Therefore, K-Means and DBSCAN were selected for deployment
        """)

    st.metric("Hierarchical Clustering Silhouette Score", "0.317")
    st.metric("Optimal Clusters Identified", "9")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <h4>🚔 Chicago Crime Intelligence Platform</h4>
        <p>Advanced Analytics for Public Safety • Data-Driven Decision Making</p>
        <p style='font-size: 0.9rem;'>Data Source: City of Chicago Data Portal | Updated Regularly</p>
    </div>
""", unsafe_allow_html=True)
with tab6:
    st.header("📉 Dimensionality Reduction Analysis")

    st.markdown("### 🔍 Overview")

    st.write("""
        - PCA (Principal Component Analysis) was used to reduce high-dimensional crime features
        - Feature engineering resulted in 16 input features
        - Two PCA configurations were evaluated:
            • 3 components → ~82% variance (best variance model)
            • 2 components → ~75% variance (selected for application)
        - Final model uses 2 components to satisfy 2D visualization requirement
        - t-SNE was applied to visualize complex crime patterns in 2D space
    """)

    st.markdown("### ⭐ Selected Model (Used in Application)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PCA Components", "2")

    with col2:
        st.metric("Variance Retained", "75.01%")

    with col3:
        st.metric("Feature Reduction", "16 → 2")

    st.success("""
    ✔ This model is used in the application as it enables clear 2D visualization 
    while retaining most of the important variance.
    """)

    st.markdown("### 🏆 Best Variance Model (MLflow Result)")

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Components", "3")

    with col5:
        st.metric("Variance", "82.22%")

    st.info("""
    ℹ Although 3 components provide higher variance retention, 
    2 components were selected based on project requirement for 2D visualization.
    """)

    st.markdown("### 📊 PCA Insights")

    st.write("""
        - Two principal components capture the majority of meaningful variance (~75%)
        - Clear cluster separation observed in reduced space
        - Minimal overlap indicates strong feature engineering
        - Supports effectiveness of clustering algorithms (KMeans & DBSCAN)
    """)

    st.markdown("### 🧠 t-SNE Visualization")

    import os
    if os.path.exists("Data/tsne_crime_types.png"):
        st.image(
            "Data/tsne_crime_types.png",
            caption="t-SNE Visualization showing clustering patterns across different crime types"
        )
    else:
        st.warning("t-SNE visualization image not found.")
    
    st.markdown("### 📊 PCA Visualization Insights")

    st.info("📊 PCA visualizations are generated using a sample of 10,000 records for performance optimization.")

    st.image(
        "Data/pca_crime_type.png",
        caption="PCA Visualization by Crime Type",
        use_container_width=True
    )

    st.image(
        "Data/pca_district.png",
        caption="PCA Visualization by District",
        use_container_width=True
    )

    st.image(
        "Data/pca_hour.png",
        caption="PCA Visualization by Hour of Day",
        use_container_width=True
    )