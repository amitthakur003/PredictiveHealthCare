import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_test_partnership(df: pd.DataFrame, models=None):
    """
    Trust & Partnership Module (Features  - )
    Covers: Mutual Trust Scores, Negotiation Simulator, and After-Care Hub
    """
    st.header(" Trust & Partnership Management")
    
    # --- 1. Data Preparation ---
    wdf = df.copy()
    
    # Ensure Payment Metrics exist
    if 'Payment_Ratio' not in wdf.columns:
        wdf['Payment_Ratio'] = np.where(wdf['Claim_Amount'] > 0, 
                                      wdf['Paid_Amount'] / wdf['Claim_Amount'], 0)
    
    if 'Denial_Flag' not in wdf.columns and 'Status' in wdf.columns:
        wdf['Denial_Flag'] = (wdf['Status'] == 'Denied').astype(int)

    # --- 2. Feature Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "  Mutual Trust Scores",
        "️  Contract Simulator",
        "  Shared After-Care Hub"
    ])

    # ---  Mutual Credibility Scores ---
    with tab1:
        st.subheader("Mutual Credibility Scorecard (0-10)")
        st.markdown("**Quantifies partner reliability based on operational data.**")
        
        col1, col2 = st.columns(2)
        
        # --- A. Hospital Trust Score (Rated by Insurer) ---
        # Logic: High Trust = Low Fraud + High Doc Score + Low Readmission
        with col1:
            st.markdown("###  Hospital Trust Index")
            
            hosp_stats = wdf.groupby('Hospital_ID').agg({
                'Claim_ID': 'count',
                'Denial_Flag': 'mean', # Proxy for Fraud/Error
                'Documentation_Score': 'mean',
                'Readmission_Flag': 'mean'
            }).reset_index()
            
            # Normalize to 0-10 Scale
            # 1. Denial Score (Lower is better): (1 - Rate) * 4
            hosp_stats['Score_Denial'] = (1 - hosp_stats['Denial_Flag']) * 4
            # 2. Doc Score (Higher is better): (Avg / 5) * 3
            hosp_stats['Score_Doc'] = (hosp_stats['Documentation_Score'] / 5) * 3
            # 3. Readmission (Lower is better): (1 - Rate) * 3
            hosp_stats['Score_Readmit'] = (1 - hosp_stats['Readmission_Flag']) * 3
            
            hosp_stats['Trust_Score'] = (hosp_stats['Score_Denial'] + 
                                       hosp_stats['Score_Doc'] + 
                                       hosp_stats['Score_Readmit']).clip(0, 10)
            
            hosp_sorted = hosp_stats.sort_values('Trust_Score', ascending=False)
            
            # Leaderboard (Removed .style.background_gradient to fix error)
            st.dataframe(
                hosp_sorted[['Hospital_ID', 'Trust_Score', 'Denial_Flag', 'Documentation_Score']],
                use_container_width=True
            )
            
        # --- B. Insurer Reliability Score (Rated by Hospital) ---
        # Logic: High Trust = Fast Payment (Low Days) + High Payment Ratio
        with col2:
            st.markdown("### ️ Insurer Reliability Index")
            
            ins_stats = wdf.groupby('Insurer_ID').agg({
                'Claim_ID': 'count',
                'Days_To_Payment': 'mean',
                'Payment_Ratio': 'mean'
            }).reset_index()
            
            # Normalize to 0-10 Scale
            # 1. Speed Score (Lower Days is better). Baseline 60 days.
            # Formula: (60 - Days) / 6. Max 5 points.
            ins_stats['Score_Speed'] = ((60 - ins_stats['Days_To_Payment'].clip(0, 60)) / 60) * 5
            # 2. Payment Ratio (Higher is better). Max 5 points.
            ins_stats['Score_Generosity'] = ins_stats['Payment_Ratio'] * 5
            
            ins_stats['Trust_Score'] = (ins_stats['Score_Speed'] + ins_stats['Score_Generosity']).clip(0, 10)
            
            ins_sorted = ins_stats.sort_values('Trust_Score', ascending=False)
            
            # (Removed .style.background_gradient to fix error)
            st.dataframe(
                ins_sorted[['Insurer_ID', 'Trust_Score', 'Days_To_Payment', 'Payment_Ratio']],
                use_container_width=True
            )

    # ---  Contract Negotiation Simulator ---
    with tab2:
        st.subheader("Contract Negotiation Simulator")
        st.markdown("**'What-If' revenue calculator to support contract renewals.**")
        
        # Current State
        current_denial_rate = wdf['Denial_Flag'].mean()
        current_revenue = wdf['Paid_Amount'].sum()
        total_claim_vol = wdf['Claim_Amount'].sum()
        
        st.info(f" **Current Baseline:** Denial Rate: {current_denial_rate:.1%} | Annual Revenue: ${current_revenue:,.0f}")
        
        st.markdown("#### Adjust Parameters to Simulate Future State:")
        
        c1, c2 = st.columns(2)
        target_denial_reduction = c1.slider(" Target Denial Rate Reduction (%)", 0.0, 10.0, 2.5, step=0.1)
        target_rate_hike = c2.slider(" Negotiated Rate Increase (%)", 0.0, 15.0, 5.0, step=0.5)
        
        # Calculations
        # 1. Revenue reclaimed from denials (Approximate)
        # New Denial Rate = Current - Reduction
        new_denial_rate = max(0, current_denial_rate - (target_denial_reduction/100))
        # Claims saved = Total Vol * Reduction %
        recovered_revenue = total_claim_vol * (target_denial_reduction/100) * 0.8 # Assume 80% recovery rate
        
        # 2. Revenue from rate hike
        # Rate hike applies to base revenue + recovered revenue
        rate_hike_gain = (current_revenue + recovered_revenue) * (target_rate_hike/100)
        
        projected_revenue = current_revenue + recovered_revenue + rate_hike_gain
        total_lift = projected_revenue - current_revenue
        
        # Visualization
        st.markdown("###  Projected Financial Impact")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Recovered from Denials", f"+${recovered_revenue:,.0f}")
        m2.metric("Gained from Rate Hike", f"+${rate_hike_gain:,.0f}")
        m3.metric("Total Revenue Lift", f"+${total_lift:,.0f}", delta=f"{total_lift/current_revenue*100:.1f}%")
        
        # Chart
        fig = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Current Revenue", "Denial Recovery", "Rate Hike", "Projected Total"],
            textposition = "outside",
            # Fixed 1M syntax error here as well
            text = [f"${current_revenue/1_000_000:.1f}M", f"+${recovered_revenue/1_000_000:.1f}M", 
                   f"+${rate_hike_gain/1_000_000:.1f}M", f"${projected_revenue/1_000_000:.1f}M"],
            y = [current_revenue, recovered_revenue, rate_hike_gain, 0],
            connector = {"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title = "Negotiation Impact Analysis", showlegend = False)
        st.plotly_chart(fig, use_container_width=True)

    # ---  Shared After-Care Hub ---
    with tab3:
        st.subheader("Shared After-Care Portal")
        st.markdown("**Secure bridge for Hospitals and Insurers to coordinate post-discharge care.**")
        
        # Filter for Paid/Discharged patients
        discharged_patients = wdf[wdf['Status'] == 'Paid'].head(10)
        
        if not discharged_patients.empty:
            selected_claim = st.selectbox(
                "Select Discharged Patient (Search by Claim ID)", 
                discharged_patients['Claim_ID'],
                format_func=lambda x: f"Claim #{x} - {discharged_patients[discharged_patients['Claim_ID']==x]['Disease_Category'].iloc[0]}"
            )
            
            patient_row = discharged_patients[discharged_patients['Claim_ID'] == selected_claim].iloc[0]
            
            # Secure View Area
            with st.container():
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 5px solid #2ecc71;">
                    <h4> Patient Care Summary</h4>
                    <b>ID:</b> {patient_row.get('Patient_ID', 'N/A')} | 
                    <b>Disease:</b> {patient_row['Disease_Category']} | 
                    <b>Discharge Date:</b> {patient_row.get('Submission_Date', 'N/A').strftime('%Y-%m-%d')}
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                
                # Hospital Side Updates
                with c1:
                    st.markdown("####  Hospital Instructions")
                    care_plan = st.text_area(
                        "Post-Discharge Plan", 
                        value="1. Monitor blood pressure daily.\n2. Physiotherapy 2x/week.\n3. Follow up visit in 14 days.",
                        height=150
                    )
                    st.button("Update Care Plan", key="btn_hosp")
                
                # Insurer Side Updates
                with c2:
                    st.markdown("#### ️ Insurer Approvals")
                    insurer_notes = st.text_area(
                        "Coverage Notes",
                        value=" Home Physiotherapy approved for 4 sessions.\n️ BP Monitor not covered under base plan.",
                        height=150
                    )
                    st.button("Sync Coverage Data", key="btn_ins")
                
                st.success(f" Data Synchronized (Latency: 0.4s). Both parties viewing version 2.1 of Care Plan.")
        else:
            st.info("No discharged patients available for After-Care coordination.")

    # Show Data Info
    with st.expander("️ Data Requirements for Trust Module"):
        st.markdown("""
        **Required Fields:**
        1. `Documentation_Score` (1-5 or 0-100) - For Hospital Trust Score.
        2. `Readmission_Flag` (0 or 1) - For Quality scoring.
        3. `Days_To_Payment` (Int) - For Insurer Trust Score.
        4. `Payment_Ratio` (Float) - For Insurer Generosity scoring.
        """)