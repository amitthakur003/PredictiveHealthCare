import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

def render_government_schemes(df: pd.DataFrame, models=None):
    """
    Government Scheme Management Module (Features  - )
    Covers: PMJAY/CGHS Tracking, Payment Aging, ICR, and ABDM Compliance.
    """
    st.header("️ Government Scheme Management")
    
    # --- 1. Data Enrichment (Mocking Scheme Data) ---
    wdf = df.copy()
    
    # Generate Scheme Types if missing
    if 'Scheme_Type' not in wdf.columns:
        # PMJAY = Ayushman Bharat, CGHS = Central Govt Health Scheme
        scheme_choices = ['Private Insurance', 'PMJAY (Ayushman)', 'CGHS', 'State Health Scheme', 'Cash']
        # Govt schemes usually make up 40-60% of volume in eligible hospitals
        wdf['Scheme_Type'] = np.random.choice(scheme_choices, size=len(wdf), p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
    # Generate ABDM Compliance Data (for Hospital level check)
    if 'ABDM_Status' not in wdf.columns:
        wdf['ABDM_Status'] = np.random.choice(['Compliant', 'In-Progress', 'Non-Compliant'], size=len(wdf), p=[0.5, 0.3, 0.2])

    # Ensure Payment Dates for Aging calculation
    if 'Submission_Date' in wdf.columns:
        wdf['Submission_Date'] = pd.to_datetime(wdf['Submission_Date'], errors='coerce')
        
    # Calculate Days Outstanding (Pending Claims)
    # If Paid: Use Days_To_Payment. If Pending: Today - Submission Date
    today = pd.Timestamp.now()
    wdf['Aging_Days'] = np.where(
        wdf['Status'] == 'Paid',
        wdf['Days_To_Payment'],
        (today - wdf['Submission_Date']).dt.days
    )
    wdf['Aging_Days'] = wdf['Aging_Days'].fillna(0)

    # --- 2. Feature Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "  Scheme Dashboard",
        "  Payment Aging",
        "  Loss Ratio (ICR)",
    
    ])

    # Filter for Government Data only for specific tabs
    govt_df = wdf[wdf['Scheme_Type'].isin(['PMJAY (Ayushman)', 'CGHS', 'State Health Scheme'])]

    # ---  PMJAY/CGHS Claims Tracker ---
    with tab1:
        st.subheader("Government Claims Dashboard")
        st.markdown("**Overview of volumes, approvals, and metrics for public sector schemes.**")
        
        if govt_df.empty:
            st.warning("No Government Scheme data detected.")
        else:
            # Metrics
            total_vol = len(govt_df)
            total_val = govt_df['Claim_Amount'].sum()
            approval_rate = (len(govt_df[govt_df['Status']=='Paid']) / total_vol) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Govt Claim Volume", total_vol)
            c2.metric("Total Claimed Value", f"${total_val:,.0f}")
            c3.metric("Approval Rate", f"{approval_rate:.1f}%")
            c4.metric("Avg Payment Delay", f"{govt_df['Aging_Days'].mean():.0f} Days")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume by Scheme
                scheme_vol = govt_df['Scheme_Type'].value_counts().reset_index()
                scheme_vol.columns = ['Scheme', 'Count']
                fig1 = px.pie(scheme_vol, values='Count', names='Scheme', 
                              title="Claim Volume by Scheme", hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                # Approval Rate by Scheme
                scheme_approval = govt_df.groupby('Scheme_Type').agg({
                    'Claim_ID': 'count',
                    'Denial_Flag': 'mean'
                }).reset_index()
                scheme_approval['Approval_Rate'] = (1 - scheme_approval['Denial_Flag']) * 100
                
                fig2 = px.bar(scheme_approval, x='Scheme_Type', y='Approval_Rate',
                              title="Approval Rate by Scheme (%)",
                              color='Approval_Rate', color_continuous_scale='Greens')
                st.plotly_chart(fig2, use_container_width=True)

    # ---  Government Payment Aging Report ---
    with tab2:
        st.subheader("Payment Aging & Delays")
        st.markdown("**Tracks time from submission to payment. Highlights >120 Day Delays.**")
        
        # Aging Buckets
        def get_bucket(days):
            if days <= 30: return '0-30 Days'
            elif days <= 60: return '31-60 Days'
            elif days <= 90: return '61-90 Days'
            elif days <= 120: return '91-120 Days'
            else: return '120+ Days (Critical)'
            
        govt_df['Aging_Bucket'] = govt_df['Aging_Days'].apply(get_bucket)
        
        # Order the buckets
        bucket_order = ['0-30 Days', '31-60 Days', '61-90 Days', '91-120 Days', '120+ Days (Critical)']
        aging_stats = govt_df['Aging_Bucket'].value_counts().reindex(bucket_order).fillna(0).reset_index()
        aging_stats.columns = ['Bucket', 'Count']
        
        # Visualization
        fig3 = px.bar(aging_stats, x='Bucket', y='Count',
                      title="Payment Aging Distribution (Govt Schemes)",
                      color='Bucket',
                      color_discrete_map={'120+ Days (Critical)': 'red', '0-30 Days': 'green'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Critical List
        critical_claims = govt_df[govt_df['Aging_Days'] > 120].sort_values('Aging_Days', ascending=False)
        
        st.error(f" **{len(critical_claims)} Claims** are pending beyond 120 days. Escalation Recommended.")
        if not critical_claims.empty:
            st.dataframe(critical_claims[['Claim_ID', 'Scheme_Type', 'Claim_Amount', 'Submission_Date', 'Aging_Days']])
            
            if st.button("Generate Escalation Letter"):
                st.success(" Escalation Letter generated for Ministry of Health / NHA.")

    # ---  Scheme-Wise Loss Ratio (ICR) ---
    with tab3:
        st.subheader("Scheme Profitability (ICR Analysis)")
        st.markdown("**Comparison: Private Insurance vs. Government Schemes.**")

        st.info("ICR = (Total Claims Paid) ÷ (Total Premium Collected) × 100.\nIndicates how much the insurer pays in claims relative to the premiums collected.")
        
        # Generate Mock Premiums for ICR calc (Premium is usually lower for Govt schemes)
        # ICR = Claims Paid / Premium
        
        icr_data = wdf.groupby('Scheme_Type').agg({
            'Paid_Amount': 'sum',
            'Claim_Amount': 'sum'
        }).reset_index()
        
        # Mock Premium: 
        # Private: Premium is usually ~130% of claims (Profitable)
        # Govt: Premium is usually ~105% of claims (Tight Margin)
        icr_data['Estimated_Premium'] = np.where(
            icr_data['Scheme_Type'] == 'Private Insurance',
            icr_data['Paid_Amount'] * 1.3,
            icr_data['Paid_Amount'] * 1.05
        )
        
        icr_data['ICR (%)'] = (icr_data['Paid_Amount'] / icr_data['Estimated_Premium']) * 100
        
        # Visualization
        fig4 = px.bar(icr_data, x='Scheme_Type', y='ICR (%)',
                      title="Incurred Claims Ratio (Lower is Better)",
                      color='ICR (%)',
                      color_continuous_scale='RdYlGn_r') # Red is high ICR (Bad)
        
        # Benchmark Line (100% Breakeven)
        fig4.add_shape(type="line", line=dict(dash='dash', color='black'),
                      x0=-0.5, x1=len(icr_data)-0.5, y0=100, y1=100)
        
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Note: ICR > 100% indicates an underwriting loss on the scheme.")

   

    # Data Requirements
    with st.expander("️ Data Requirements for Govt Schemes"):
        st.markdown("""
        **Required Additional Fields:**
        1. `Scheme_Type` (String) - Critical. Values: 'PMJAY', 'CGHS', 'Private', 'Cash'.
        2. `ABDM_Status` (String) - For Compliance tracking. Values: 'Compliant', 'Non-Compliant'.
        3. `ABHA_ID` (String) - Optional, to verify patient identity.
        """)