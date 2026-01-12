import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_financial_intelligence(df: pd.DataFrame, models=None):
    """
    Financial Intelligence & ROI Module
    Covers: Payment Calendar, Underpayment Alerts, Ghost Patients, Outliers
    """
    st.header(" Financial Intelligence (ROI)")
    
    # --- 1. Data Preparation & Enrichment ---
    # We create a working copy to add calculated fields without altering the original session state
    wdf = df.copy()
    
    # List of date columns to convert from the new dataset
    date_cols = ['Submission_Date', 'Predicted_Payment_Date', 'Service_Date', 'Dispute_Open_Date']
    for col in date_cols:
        if col in wdf.columns:
            wdf[col] = pd.to_datetime(wdf[col], errors='coerce')

    # Ensure numeric columns are actually numeric (handling any CSV parsing string issues)
    num_cols = ['Claim_Amount', 'Paid_Amount', 'Contracted_Rate', 'Days_To_Payment']
    for col in num_cols:
        if col in wdf.columns:
            wdf[col] = pd.to_numeric(wdf[col], errors='coerce').fillna(0)

    # --- 2. High Level KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_claimed = wdf['Claim_Amount'].sum()
    total_paid = wdf['Paid_Amount'].sum()
    
    # Use actual Contracted_Rate from CSV
    total_contracted = wdf['Contracted_Rate'].sum() if 'Contracted_Rate' in wdf.columns else 0
    
    # Revenue Leakage = What we should have been paid - What we got
    leakage = total_contracted - total_paid
    
    with col1:
        st.metric("Total Claimed", f"${total_claimed:,.0f}")
    with col2:
        st.metric("Total Paid", f"${total_paid:,.0f}")
    with col3:
        st.metric("Contracted Expected", f"${total_contracted:,.0f}")
    with col4:
        st.metric("Revenue Leakage", f"${leakage:,.0f}", 
                  delta="Potential Loss" if leakage > 0 else "Surplus", 
                  delta_color="inverse")

    # --- 3. Feature Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Payment Calendar", 
        "  Underpayment Alerts", 
        "  Ghost Patients", 
        "  Value Outliers"
    ])

    # ---  Payment Date Calendar ---
    with tab1:
        st.subheader("Cash Flow Forecast Calendar")
        st.markdown("**Visualizes expected incoming payments based on predicted settlement dates.**")
        
        if 'Predicted_Payment_Date' in wdf.columns:
            # Aggregate by Week/Month
            wdf['Payment_Week'] = wdf['Predicted_Payment_Date'].dt.to_period('W').astype(str)
            
            # Filter for valid dates only
            calendar_data = wdf[wdf['Predicted_Payment_Date'].notna()]
            cash_flow = calendar_data.groupby('Payment_Week')['Paid_Amount'].sum().reset_index()
            
            # Identify Overdue (Predicted Date < Today AND Status != Paid)
            today = pd.Timestamp.now()
            overdue = wdf[
                (wdf['Predicted_Payment_Date'] < today) & 
                (wdf['Status'] != 'Paid')
            ]
            
            # Visualization
            fig = px.bar(cash_flow, x='Payment_Week', y='Paid_Amount', 
                         title="Projected Cash Flow by Week",
                         labels={'Paid_Amount': 'Expected Revenue ($)'},
                         color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
            
            if not overdue.empty:
                st.error(f"️ **{len(overdue)} Claims Overdue** (Past predicted date but not paid)")
                
                # Show relevant columns available in new dataset
                display_cols = ['Claim_ID', 'Insurer_ID', 'Claim_Amount', 'Predicted_Payment_Date']
                if 'Days_To_Payment' in overdue.columns:
                    display_cols.append('Days_To_Payment')
                    
                st.dataframe(overdue[display_cols])
            else:
                st.success(" No overdue payments detected.")
        else:
            st.warning("Predicted_Payment_Date column missing in dataset.")

    # ---  Underpayment Alerts ---
    with tab2:
        st.subheader("Contract Variance & Underpayments")
        st.markdown("**Flags claims where Paid Amount is significantly lower than Contracted Rate.**")
        
        if 'Contracted_Rate' in wdf.columns and 'Paid_Amount' in wdf.columns:
            variance_threshold = st.slider("Variance Threshold (%)", 5, 50, 10, key="var_thresh")
            
            # Logic: (Contracted - Paid) / Contracted > Threshold
            # Handle division by zero
            wdf['Variance_Pct'] = wdf.apply(
                lambda x: ((x['Contracted_Rate'] - x['Paid_Amount']) / x['Contracted_Rate'] * 100) 
                if x['Contracted_Rate'] > 0 else 0, axis=1
            )
            
            underpaid = wdf[
                (wdf['Status'] == 'Paid') & 
                (wdf['Variance_Pct'] > variance_threshold)
            ].sort_values('Variance_Pct', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Underpaid Claims Count", len(underpaid))
            with col2:
                recoverable = underpaid['Contracted_Rate'].sum() - underpaid['Paid_Amount'].sum()
                st.metric("Total Recoverable Amount", f"${recoverable:,.0f}")
                
            if not underpaid.empty:
                fig = px.scatter(underpaid, x='Contracted_Rate', y='Paid_Amount', 
                                 color='Insurer_ID', hover_data=['Claim_ID', 'Variance_Pct'],
                                 title=f"Underpayment Analysis (Variance > {variance_threshold}%)")
                # Add a 1:1 line
                fig.add_shape(type="line", line=dict(dash='dash'),
                            x0=0, x1=underpaid['Contracted_Rate'].max(),
                            y0=0, y1=underpaid['Contracted_Rate'].max())
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(underpaid[['Claim_ID', 'Insurer_ID', 'Claim_Amount', 'Contracted_Rate', 'Paid_Amount', 'Variance_Pct']])
            else:
                st.success("No significant underpayments detected.")
        else:
            st.warning("Contracted_Rate or Paid_Amount data missing.")

    # ---   Ghost Patient Detector ---
    with tab3:
        st.subheader("Ghost Patient Detection (Fraud)")
        st.markdown("**Identifies clusters where the same Patient ID is reused suspiciously across hospitals/dates.**")
        
        # Uses Patient_ID from the CSV directly
        if 'Patient_ID' in wdf.columns and 'Hospital_ID' in wdf.columns:
            ghost_suspects = []
            
            # Filter patients with multiple claims
            multi_claim_patients = wdf.groupby('Patient_ID').filter(lambda x: len(x) > 1)
            
            for pid, group in multi_claim_patients.groupby('Patient_ID'):
                unique_hospitals = group['Hospital_ID'].nunique()
                if unique_hospitals > 1:
                    # Sort by date
                    group = group.sort_values('Submission_Date')
                    # Calculate time difference between claims
                    group['Time_Diff'] = group['Submission_Date'].diff().dt.days
                    
                    # If time diff is small (< 5 days) but hospital is different
                    suspicious = group[group['Time_Diff'] < 5]
                    if not suspicious.empty:
                        ghost_suspects.append(group)
                        
            if ghost_suspects:
                ghost_df = pd.concat(ghost_suspects)
                st.error(f" Detected {ghost_df['Patient_ID'].nunique()} suspicious Patient IDs used across different hospitals concurrently.")
                
                display_cols = ['Patient_ID', 'Claim_ID', 'Hospital_ID', 'Submission_Date']
                if 'Disease_Category' in ghost_df.columns:
                    display_cols.append('Disease_Category')
                    
                st.dataframe(ghost_df[display_cols].sort_values(['Patient_ID', 'Submission_Date']))
            else:
                st.success(" No Ghost Patient patterns detected.")
        else:
            st.warning("Patient_ID or Hospital_ID missing, cannot run Ghost Patient detection.")

    # ---  Claim Value Outliers ---
    with tab4:
        st.subheader("Statistical Outliers (Upcoding)")
        st.markdown("**Detects claims with unusually high cost (>3 Std Dev) for their CPT Code.**")
        
        if 'CPT_Code' in wdf.columns:
            # Calculate Z-Score per CPT Code
            # Only consider Paid or Pending claims to avoid skewing with denied $0
            valid_claims = wdf[wdf['Claim_Amount'] > 0].copy()
            
            # Calculate stats per CPT
            cpt_stats = valid_claims.groupby('CPT_Code')['Claim_Amount'].agg(['mean', 'std']).reset_index()
            valid_claims = valid_claims.merge(cpt_stats, on='CPT_Code')
            
            # Z-Score Formula: (X - Mean) / Std
            # Avoid division by zero (std=0)
            valid_claims['Z_Score'] = valid_claims.apply(
                lambda x: (x['Claim_Amount'] - x['mean']) / x['std'] if x['std'] > 0 else 0, axis=1
            )
            
            # Filter Outliers (> 3 Sigma)
            outliers = valid_claims[valid_claims['Z_Score'] > 3].sort_values('Z_Score', ascending=False)
            
            st.metric("Statistical Outliers Found", len(outliers))
            
            if not outliers.empty:
                # Visualization
                fig = px.scatter(valid_claims, x='CPT_Code', y='Claim_Amount', 
                                 color='Z_Score',
                                 color_continuous_scale='Bluered',
                                 hover_data=['Claim_ID', 'Hospital_ID'],
                                 title="Claim Amount Distribution by CPT (Red = Outlier)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("###  Flagged Claims details")
                st.dataframe(outliers[['Claim_ID', 'Hospital_ID', 'CPT_Code', 'Claim_Amount', 'mean', 'Z_Score']]
                             .rename(columns={'mean': 'Avg_For_CPT'}))
            else:
                st.success(" No cost outliers detected.")
        else:
            st.warning("CPT_Code missing, cannot run outlier detection.")

# Execution Block to load the specific data file
if __name__ == "__main__":
    try:
        # Load the global dataset as requested
        file_path = '../healthcare_claims_data.csv'
        df_claims = pd.read_csv(file_path)
        
        # Render the financial module with the loaded data
        render_financial_intelligence(df_claims)
        
    except FileNotFoundError:
        st.error(f"Error: Global data file not found at '{file_path}'. Please ensure the file exists.")
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")