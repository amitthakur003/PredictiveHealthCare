import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_patient_experience(df: pd.DataFrame, models=None):
    """
    Patient Experience & NPS Module (Features  - )
    Covers: Review Aggregation, NPS Tracking, Complaints, and Cashless Speed.
    """
    st.header(" Patient Experience & NPS")
    
    # --- 1. Data Enrichment (Mocking Sentiment Data) ---
    wdf = df.copy()
    
    # Generate Review Data if missing
    if 'Review_Platform' not in wdf.columns:
        wdf['Review_Platform'] = np.random.choice(['Google', 'Practo', 'Twitter', 'Direct'], size=len(wdf))
        
    if 'Review_Rating' not in wdf.columns:
        # Skew towards positive (3-5 stars)
        wdf['Review_Rating'] = np.random.choice([1, 2, 3, 4, 5], size=len(wdf), p=[0.05, 0.1, 0.15, 0.4, 0.3])
        
    if 'NPS_Score' not in wdf.columns:
        # 0-10 Scale
        wdf['NPS_Score'] = np.random.randint(0, 11, size=len(wdf))
        
    if 'Complaint_Category' not in wdf.columns:
        # Only ~15% of claims have complaints
        complaints = ['Billing Error', 'Staff Behavior', 'Long Wait Time', 'Cleanliness', 'Food Quality', 'None']
        wdf['Complaint_Category'] = np.random.choice(complaints, size=len(wdf), p=[0.05, 0.03, 0.04, 0.02, 0.01, 0.85])

    if 'Cashless_Approval_Hours' not in wdf.columns:
        # Time taken for pre-auth (e.g., 2 hours to 48 hours)
        wdf['Cashless_Approval_Hours'] = np.random.lognormal(mean=1.5, sigma=0.5, size=len(wdf)) # Log-normal distrib

    # --- 2. Feature Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Review Aggregator",
        "  NPS Monitor",
        "  Complaint Dashboard",
        "  Cashless Score"
    ])

    # ---  Review Aggregator ---
    with tab1:
        st.subheader("Omnichannel Review Analyzer")
        st.markdown("**Aggregates reviews from Google, Practo, and Internal Surveys.**")
        
        # Calculate Sentiment (Simple logic: >3 Positive, =3 Neutral, <3 Negative)
        wdf['Sentiment'] = np.where(wdf['Review_Rating'] > 3, 'Positive', 
                           np.where(wdf['Review_Rating'] == 3, 'Neutral', 'Negative'))
        
        # Metrics
        avg_rating = wdf['Review_Rating'].mean()
        total_reviews = len(wdf)
        positive_pct = (len(wdf[wdf['Sentiment']=='Positive']) / total_reviews) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Rating", f"{avg_rating:.1f} / 5.0", delta=f"{positive_pct:.0f}% Positive")
        c2.metric("Total Reviews Scanned", total_reviews)
        c3.metric("Top Platform", wdf['Review_Platform'].mode()[0])
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            sent_counts = wdf['Sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Count']
            fig = px.pie(sent_counts, values='Count', names='Sentiment', 
                         title="Sentiment Distribution",
                         color='Sentiment',
                         color_discrete_map={'Positive': 'green', 'Neutral': 'gold', 'Negative': 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Platform breakdown
            plat_rating = wdf.groupby('Review_Platform')['Review_Rating'].mean().reset_index()
            fig2 = px.bar(plat_rating, x='Review_Platform', y='Review_Rating',
                          title="Avg Rating by Source", range_y=[0,5],
                          color='Review_Rating', color_continuous_scale='Blues')
            st.plotly_chart(fig2, use_container_width=True)
            

    # ---  NPS Trend Monitor ---
    with tab2:
        st.subheader("Net Promoter Score (NPS) Tracker")
        st.markdown("**NPS = % Promoters (9-10) - % Detractors (0-6)**")

        # Add a layman-friendly explanation using expander
        with st.expander("What is NPS?"):
            st.markdown("""
            - **Promoters (9-10):** Very happy patients who are likely to recommend the hospital.  
            - **Detractors (0-6):** Unhappy patients who may complain or discourage others.  
            - **Passives (7-8):** Neutral patients, not counted in NPS.  
            - **NPS Score:** Positive values mean more happy patients than unhappy ones. Negative values mean more dissatisfied patients.
            """)

        # Categorize NPS
        conditions = [
            (wdf['NPS_Score'] >= 9),
            (wdf['NPS_Score'] <= 6)
        ]
        choices = ['Promoter', 'Detractor']
        wdf['NPS_Category'] = np.select(conditions, choices, default='Passive')

        # Calculate NPS
        nps_counts = wdf['NPS_Category'].value_counts(normalize=True) * 100
        nps_score = nps_counts.get('Promoter', 0) - nps_counts.get('Detractor', 0)

        st.metric("Current NPS Score", f"{nps_score:.0f}", 
                 delta="Excellent" if nps_score > 50 else "Needs Improvement")

        if 'Submission_Date' in wdf.columns:
            wdf['Month_Year'] = wdf['Submission_Date'].dt.to_period('M').astype(str)

            # Group by month and calculate NPS per month
            trend_data = []
            for month, group in wdf.groupby('Month_Year'):
                counts = group['NPS_Category'].value_counts(normalize=True) * 100
                score = counts.get('Promoter', 0) - counts.get('Detractor', 0)
                trend_data.append({'Month': month, 'NPS': score})

            trend_df = pd.DataFrame(trend_data)

            fig3 = px.line(trend_df, x='Month', y='NPS', markers=True,
                           title="NPS Trend Over Time")
            fig3.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Target")
            st.plotly_chart(fig3, use_container_width=True)


    # ---  Complaint Volume Dashboard ---
    with tab3:
        st.subheader("Grievance & Complaint Analytics")
        st.markdown("**Real-time classification of patient complaints.**")
        
        # Filter only rows with actual complaints
        complaints_df = wdf[wdf['Complaint_Category'] != 'None'].copy()
        
        if complaints_df.empty:
            st.success(" No complaints recorded in this dataset!")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Complaints", len(complaints_df))
            
            # Most complained hospital
            worst_hosp = complaints_df['Hospital_ID'].mode()[0]
            c2.metric("Most Complaints From", worst_hosp)
            
            # Treemap
            fig4 = px.treemap(complaints_df, path=['Complaint_Category', 'Hospital_ID'],
                              title="Complaint Drivers (Click to Drill Down)")
            st.plotly_chart(fig4, use_container_width=True)
            
            # Detailed Table
            st.dataframe(complaints_df[['Claim_ID', 'Hospital_ID', 'Complaint_Category', 'NPS_Score']])

    # ---  Cashless Experience Score ---
    with tab4:
        st.subheader("Cashless Experience & TAT Scoring")
        st.markdown("**Rating hospitals based on Cashless Pre-Auth Speed.**")
        
        # Define Rating Logic
        conditions = [
            (wdf['Cashless_Approval_Hours'] <= 2),
            (wdf['Cashless_Approval_Hours'] <= 4),
            (wdf['Cashless_Approval_Hours'] <= 12),
            (wdf['Cashless_Approval_Hours'] <= 24)
        ]
        stars = [5, 4, 3, 2]
        wdf['Service_Rating'] = np.select(conditions, stars, default=1)
        
        # Aggregation by Hospital
        perf = wdf.groupby('Hospital_ID').agg({
            'Cashless_Approval_Hours': 'mean',
            'Service_Rating': 'mean',
            'Claim_ID': 'count'
        }).reset_index()
        
        perf = perf.sort_values('Service_Rating', ascending=False)
        
        # Visual
        fig5 = px.scatter(perf, x='Cashless_Approval_Hours', y='Service_Rating',
                          size='Claim_ID', color='Service_Rating',
                          hover_name='Hospital_ID',
                          color_continuous_scale='RdYlGn',
                          title="Hospital Service Quality Matrix (Speed vs Rating)",
                          labels={'Cashless_Approval_Hours': 'Avg Approval Time (Hours)'})
        
        # Add "Gold Standard" box
        fig5.add_shape(type="rect", x0=0, y0=4.5, x1=2, y1=5.2,
                      line=dict(color="gold"), fillcolor="lightyellow", opacity=0.3)
        
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Gold Box = Elite Service Partners (< 2 Hours Approval)")

    # Data Info
    with st.expander("️ Data Requirements for Patient Experience"):
        st.markdown("""
        **Required Additional Fields:**
        1. `NPS_Score` (Int 0-10) - Net Promoter Score survey result.
        2. `Review_Rating` (Int 1-5) - Star rating from Google/Practo.
        3. `Complaint_Category` (String) - E.g., 'Billing', 'Staff', 'None'.
        4. `Cashless_Approval_Hours` (Float) - Time taken for pre-auth approval.
        """)
