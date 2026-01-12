import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_network_geography(df: pd.DataFrame, models=None):
    """
    Network & Geography Module (Features  - )
    Covers: Hospital Heatmaps, Patient Travel Analysis, Competitor Gaps, and Geo-Scoring.
    """
    st.header(" Network & Geography Intelligence")
    
    # --- 1. Data Enrichment (Mocking Geospatial Data) ---
    wdf = df.copy()
    
    # We need locations. Let's simulate data around a central city (e.g., Mumbai/NY coordinates)
    # Center Point (28.6139, 77.2090 - New Delhi approx)
    center_lat, center_lon = 28.6139, 77.2090
    
    # 1. Assign random coordinates to Hospitals (Fixed per ID)
    if 'Hospital_Lat' not in wdf.columns:
        unique_hospitals = wdf['Hospital_ID'].unique()
        hosp_locs = {
            h: (center_lat + np.random.uniform(-0.1, 0.1), center_lon + np.random.uniform(-0.1, 0.1)) 
            for h in unique_hospitals
        }
        
        wdf['Hospital_Lat'] = wdf['Hospital_ID'].map(lambda x: hosp_locs[x][0])
        wdf['Hospital_Lon'] = wdf['Hospital_ID'].map(lambda x: hosp_locs[x][1])
        
        # Assign City/Zone based on location chunks
        wdf['Hospital_City'] = np.where(wdf['Hospital_Lat'] > center_lat, 'North Zone', 'South Zone')

    # 2. Assign Patient Origins (Slightly more spread out)
    if 'Patient_Lat' not in wdf.columns:
        wdf['Patient_Lat'] = wdf['Hospital_Lat'] + np.random.normal(0, 0.05, size=len(wdf))
        wdf['Patient_Lon'] = wdf['Hospital_Lon'] + np.random.normal(0, 0.05, size=len(wdf))
        
    # 3. Calculate Distance (Mock Euclidean for speed, roughly degrees to km)
    wdf['Travel_Distance_KM'] = np.sqrt(
        (wdf['Patient_Lat'] - wdf['Hospital_Lat'])**2 + 
        (wdf['Patient_Lon'] - wdf['Hospital_Lon'])**2
    ) * 111 # Convert degrees to km approx

    # --- 2. Feature Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "️  Network Heatmap",
        "  Patient Access",
        "️  Competitor Analysis",
        "  Geo-Performance"
    ])

    # ---  Hospital Density Heatmap ---
    with tab1:
        st.subheader("Hospital Network Density")
        st.markdown("**Geographic distribution of empaneled hospitals.**")
        
        # Deduplicate to show unique hospitals only
        hospital_map_data = wdf.drop_duplicates('Hospital_ID').copy()
        hospital_map_data['Bed_Capacity'] = np.random.randint(50, 500, size=len(hospital_map_data)) # Mock Capacity
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Mapbox Density Map
            fig = px.density_mapbox(
                hospital_map_data, 
                lat='Hospital_Lat', lon='Hospital_Lon',
                z='Bed_Capacity', radius=20,
                center=dict(lat=center_lat, lon=center_lon), zoom=10,
                mapbox_style="open-street-map",
                title="Hospital Concentration (Heatmap by Bed Capacity)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.metric("Total Hospitals", len(hospital_map_data))
            st.metric("Avg Bed Capacity", f"{hospital_map_data['Bed_Capacity'].mean():.0f}")
            
            filter_specialty = st.multiselect("Filter Specialty", wdf['Disease_Category'].unique())

    # ---  Patient Origin vs Hospital Location ---
    with tab2:
        st.subheader("Patient Access & Coverage Gaps")
        st.markdown("**Analysis of how far patients travel to access care.**")
        
        # Metrics
        avg_dist = wdf['Travel_Distance_KM'].mean()
        long_travel = len(wdf[wdf['Travel_Distance_KM'] > 15]) # Patients traveling > 15km
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Travel Distance", f"{avg_dist:.1f} km")
        c2.metric("Patients Traveling >15km", f"{long_travel} ({long_travel/len(wdf):.1%})")
        c3.metric("Underserved Regions", "2 Zones Identified")
        
        # Distribution Plot
        fig2 = px.histogram(wdf, x='Travel_Distance_KM', nbins=30,
                            title="Patient Travel Distance Distribution",
                            color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Gap Report
        st.info(" **Coverage Gap Detected:** High cluster of patients in 'West Sector' traveling >20km. Recommendation: Empanel 'City Care Hospital' in West Sector.")

    # ---  Competitor Network Analysis ---
    with tab3:
        st.subheader("Competitor Network Penetration")
        st.markdown("**Where are our competitors stronger than us?**")
        
        # Mock Competitor Data
        regions = ['North Zone', 'South Zone', 'East Sector', 'West Sector']
        comp_data = pd.DataFrame({
            'Region': regions,
            'Our_Network': [15, 20, 5, 2],
            'Competitor_A': [12, 18, 15, 10],
            'Competitor_B': [10, 22, 8, 8]
        })
        
        # Reshape for grouped bar
        comp_melt = comp_data.melt(id_vars='Region', var_name='Insurer', value_name='Hospital_Count')
        
        fig3 = px.bar(comp_melt, x='Region', y='Hospital_Count', color='Insurer',
                      barmode='group', title="Network Strength vs Competitors by Region",
                      color_discrete_map={'Our_Network': '#2ecc71', 'Competitor_A': '#e74c3c', 'Competitor_B': '#f1c40f'})
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.warning("️ **Strategic Alert:** Weak presence in 'West Sector' compared to Competitor A. Recruitment needed.")

    # ---  Geo-Based Performance Scoring ---
    with tab4:
        st.subheader("Regional Performance Leaderboard")
        st.markdown("**Ranking regions based on Cost, Quality (Readmission), and Volume.**")
        
        # Aggregation by City/Zone
        geo_perf = wdf.groupby('Hospital_City').agg({
            'Claim_Amount': 'mean',          # Cost
            'Readmission_Flag': 'mean',      # Quality (Lower is better)
            'Claim_ID': 'count'              # Volume
        }).reset_index()
        
        geo_perf.rename(columns={
            'Claim_Amount': 'Avg_Cost', 
            'Readmission_Flag': 'Readmission_Rate',
            'Claim_ID': 'Volume'
        }, inplace=True)
        
        # Normalize Quality Score (Inverse of readmission)
        geo_perf['Quality_Score'] = (1 - geo_perf['Readmission_Rate']) * 100
        
        # Bubble Chart
        fig4 = px.scatter(geo_perf, x='Avg_Cost', y='Quality_Score',
                          size='Volume', color='Hospital_City',
                          hover_name='Hospital_City',
                          title="Region Performance Matrix (Size = Volume)",
                          labels={'Avg_Cost': 'Average Claim Cost ($)', 'Quality_Score': 'Quality Score (Higher is Better)'})
        
        # Add quadrants
        fig4.add_hline(y=geo_perf['Quality_Score'].mean(), line_dash="dash", line_color="grey")
        fig4.add_vline(x=geo_perf['Avg_Cost'].mean(), line_dash="dash", line_color="grey")
        
        st.plotly_chart(fig4, use_container_width=True)
        
        st.dataframe(geo_perf.style.highlight_max(axis=0, subset=['Quality_Score', 'Volume'], color='lightgreen'))

    # Data Requirements Info
    with st.expander("️ Data Requirements for Network Module"):
        st.markdown("""
        **Required Additional Fields:**
        1. `Hospital_Lat`, `Hospital_Long` (Float) - Coordinates for the hospital.
        2. `Patient_Zip` or `Patient_City` (String) - To calculate origin/travel.
        3. `Bed_Capacity` (Int) - For density heatmaps.
        """)