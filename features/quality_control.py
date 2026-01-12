import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

def render_quality_control(df: pd.DataFrame, models=None):
    """
    Quality & Cost Control Module (Features  - )
    Covers: ALOS, Readmissions, Package Compliance, and Loss Ratios (ICR)
    """
    st.header(" Quality & Cost Control")
    
    # --- 1. Data Enrichment (Mocking missing fields for Demo) ---
    wdf = df.copy()
    
    # Ensure standard fields exist
    if 'Submission_Date' in wdf.columns:
        wdf['Submission_Date'] = pd.to_datetime(wdf['Submission_Date'], errors='coerce')
    
    # Generate ALOS Data (Admission/Discharge) if missing
    if 'Admission_Date' not in wdf.columns:
        # Random ALOS between 2 and 15 days
        alos_days = np.random.randint(2, 15, size=len(wdf))
        wdf['Discharge_Date'] = wdf['Submission_Date']
        wdf['Admission_Date'] = wdf['Discharge_Date'] - pd.to_timedelta(alos_days, unit='D')
        wdf['Length_Of_Stay'] = alos_days
    else:
        # Calculate real ALOS
        wdf['Admission_Date'] = pd.to_datetime(wdf['Admission_Date'])
        wdf['Discharge_Date'] = pd.to_datetime(wdf['Discharge_Date'])
        wdf['Length_Of_Stay'] = (wdf['Discharge_Date'] - wdf['Admission_Date']).dt.days

    # Generate Billing Type (Package vs Open)
    if 'Billing_Type' not in wdf.columns:
        wdf['Billing_Type'] = np.random.choice(['Package', 'Open'], size=len(wdf), p=[0.7, 0.3])
    
    # Generate Package Rates for comparison
    if 'Package_Rate' not in wdf.columns:
        # Assume Package Rate is usually 10% less than Claim Amount
        wdf['Package_Rate'] = wdf['Claim_Amount'] * 0.90

    # Generate Premium Collected (For ICR Calculation)
    # We attribute a mock "Total Premium" to each hospital for the ICR formula
    if 'Premium_Allocation' not in wdf.columns:
        wdf['Premium_Allocation'] = wdf['Claim_Amount'] * np.random.uniform(1.1, 1.5, size=len(wdf))

    # --- 2. Feature Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "️  ALOS Tracker",
        "  Readmission Monitor",
        "  Package vs Open",
        "  Loss Ratio (ICR)"
    ])

    # ---  Hospital Stay Length Tracker (ALOS) ---
    with tab1:
        st.subheader("Average Length of Stay (ALOS) Tracker")
        st.markdown("**Compare hospital stay duration against benchmarks.**")
        
        # Benchmarks Dictionary (Mock)
        benchmarks = {
            'Cardio': 5.0, 'Ortho': 4.5, 'Onco': 8.0, 'Pedia': 3.0, 'General': 4.0
        }
        
        # Calculate ALOS per Hospital & Disease
        if 'Disease_Category' in wdf.columns:
            alos_df = wdf.groupby(['Hospital_ID', 'Disease_Category'])['Length_Of_Stay'].mean().reset_index()
            
            # Add Benchmark Column
            alos_df['Benchmark'] = alos_df['Disease_Category'].map(benchmarks).fillna(4.0)
            alos_df['Variance'] = alos_df['Length_Of_Stay'] - alos_df['Benchmark']
            alos_df['Status'] = np.where(alos_df['Variance'] > 1, 'High Deviation', 'Normal')
            
            # Filter
            disease_filter = st.selectbox("Select Disease Category", wdf['Disease_Category'].unique())
            filtered_alos = alos_df[alos_df['Disease_Category'] == disease_filter].sort_values('Length_Of_Stay')
            
            # Visualization
            fig = go.Figure()
            # Bar for Actual ALOS
            fig.add_trace(go.Bar(
                x=filtered_alos['Hospital_ID'], y=filtered_alos['Length_Of_Stay'],
                name='Actual ALOS', marker_color='#3498db'
            ))
            # Line for Benchmark
            fig.add_trace(go.Scatter(
                x=filtered_alos['Hospital_ID'], y=filtered_alos['Benchmark'],
                name='Benchmark Limit', mode='lines+markers', line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title=f"ALOS Comparison: {disease_filter}", yaxis_title="Days")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show Violations
            high_alos = filtered_alos[filtered_alos['Status'] == 'High Deviation']
            if not high_alos.empty:
                st.error(f"️ {len(high_alos)} Hospitals exceeding ALOS benchmark for {disease_filter}")
                st.dataframe(high_alos[['Hospital_ID', 'Length_Of_Stay', 'Benchmark', 'Variance']])
            else:
                st.success(" All hospitals within ALOS limits.")

    # ---  Readmission Rate Monitor ---
    with tab2:
        st.subheader("30-Day Readmission Monitor")
        st.markdown("**Track readmission rates exceeding 12% threshold.**")
        
        if 'Readmission_Flag' in wdf.columns:
            # Group by Hospital
            readmit_stats = wdf.groupby('Hospital_ID').agg({
                'Claim_ID': 'count',
                'Readmission_Flag': 'sum'
            }).reset_index()
            
            readmit_stats['Readmission_Rate'] = (readmit_stats['Readmission_Flag'] / readmit_stats['Claim_ID']) * 100
            
            # Threshold Alert
            threshold = 12.0
            high_readmit = readmit_stats[readmit_stats['Readmission_Rate'] > threshold]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Readmission Rate", f"{readmit_stats['Readmission_Rate'].mean():.1f}%")
            with col2:
                st.metric("Hospitals at Risk (>12%)", len(high_readmit))
                
            # Visualization
            fig2 = px.bar(readmit_stats.sort_values('Readmission_Rate', ascending=False),
                          x='Hospital_ID', y='Readmission_Rate',
                          title="Readmission Rate by Hospital",
                          color='Readmission_Rate',
                          color_continuous_scale=['green', 'yellow', 'red'])
            
            # Add Threshold Line
            fig2.add_shape(type="line", line=dict(dash='dash', color='red'),
                          x0=-0.5, x1=len(readmit_stats)-0.5, y0=threshold, y1=threshold)
            
            st.plotly_chart(fig2, use_container_width=True)
            
            if not high_readmit.empty:
                st.warning("️ The following hospitals have critically high readmission rates:")
                st.dataframe(high_readmit)

    # ---  Package vs Open Billing ---
    with tab3:
        st.subheader("Package vs. Open Billing Alerts")
        st.markdown("**Detects cases billed as 'Open' (Itemized) exceeding standard 'Package' rates.**")
        
        # Filter for Open Billing only
        open_billing = wdf[wdf['Billing_Type'] == 'Open'].copy()
        
        # Logic: If Claim Amount > Package Rate + 10% Buffer
        open_billing['Variance'] = open_billing['Claim_Amount'] - open_billing['Package_Rate']
        open_billing['Variance_Pct'] = (open_billing['Variance'] / open_billing['Package_Rate']) * 100
        
        # Alert logic
        violations = open_billing[open_billing['Variance_Pct'] > 10].sort_values('Variance_Pct', ascending=False)
        
        st.metric("Open Billing Violations", len(violations))
        
        if not violations.empty:
            fig3 = px.scatter(violations, x='Package_Rate', y='Claim_Amount',
                              color='Hospital_ID', size='Variance_Pct',
                              title="Billing Variance: Open Bill vs Package Rate",
                              hover_data=['Claim_ID', 'Disease_Category'])
            
            # 1:1 Line
            fig3.add_shape(type="line", line=dict(dash='dash', color='grey'),
                          x0=0, x1=violations['Package_Rate'].max(),
                          y0=0, y1=violations['Package_Rate'].max())
            
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(violations[['Claim_ID', 'Hospital_ID', 'Claim_Amount', 'Package_Rate', 'Variance_Pct']])
        else:
            st.success(" No billing violations detected.")

    # ---  Loss Ratio Calculator (ICR) ---
    with tab4:
        st.subheader("Incurred Claims Ratio (ICR)")
        st.markdown("**ICR = (Claims Paid / Premium Collected) * 100**")
        st.info("ICR > 100% means the insurer is losing money on that hospital/segment.")
        
        # Calculate ICR per Hospital
        icr_df = wdf.groupby('Hospital_ID').agg({
            'Paid_Amount': 'sum',
            'Premium_Allocation': 'sum'  # Mocked field
        }).reset_index()
        
        icr_df['ICR'] = (icr_df['Paid_Amount'] / icr_df['Premium_Allocation']) * 100
        icr_df = icr_df.sort_values('ICR', ascending=False)
        
        # Status
        icr_df['Risk_Status'] = np.where(icr_df['ICR'] > 100, 'Loss Making', 
                                np.where(icr_df['ICR'] > 85, 'Warning', 'Profitable'))
        
        # Metrics
        avg_icr = icr_df['ICR'].mean()
        loss_making = len(icr_df[icr_df['ICR'] > 100])
        
        c1, c2 = st.columns(2)
        c1.metric("Network Average ICR", f"{avg_icr:.1f}%")
        c2.metric("Loss-Making Hospitals", loss_making, delta="High Risk", delta_color="inverse")
        
        # Visualization
        fig4 = px.bar(icr_df, x='Hospital_ID', y='ICR',
                      color='Risk_Status',
                      title="Incurred Claims Ratio (ICR) by Hospital",
                      color_discrete_map={'Loss Making': 'red', 'Warning': 'orange', 'Profitable': 'green'})
        
        # Add 100% Breakeven line
        fig4.add_shape(type="line", line=dict(dash='dash', color='black'),
                      x0=-0.5, x1=len(icr_df)-0.5, y0=100, y1=100)
        
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(icr_df)

    # Show Data Info
    with st.expander("️ Data Requirements for Quality Module"):
        st.markdown("""
        **Required Fields:**
        1. `Admission_Date` & `Discharge_Date` (For ALOS)
        2. `Billing_Type` ('Package' or 'Open')
        3. `Package_Rate` (Contracted rate for the CPT)
        4. `Premium_Collected` (For Loss Ratio calculation)
        """)