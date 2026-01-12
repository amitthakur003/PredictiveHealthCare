import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_tpa_management(df: pd.DataFrame, models=None):
    """
    TPA & Partner Management Module (Features  - )
    Covers: Performance Benchmarking, Cost Analysis, Doc Loss, and Response Times
    """
    st.header(" Third-Party Administration Management")
    
    # --- 1. Data Enrichment (Mocking Missing Fields) ---
    wdf = df.copy()
    
    # Generate TPA Names (In-House vs External)
    if 'TPA_Name' not in wdf.columns:
        # 30% In-House, 70% split between 3 TPAs
        tpa_choices = ['In-House', 'Medi-X TPA', 'GlobalHealth TPA', 'FastTrack TPA']
        wdf['TPA_Name'] = np.random.choice(tpa_choices, size=len(wdf), p=[0.3, 0.25, 0.25, 0.2])
    
    # Generate Query Counts (How many times TPA asked for more info)
    if 'Query_Count' not in wdf.columns:
        wdf['Query_Count'] = np.random.poisson(0.5, size=len(wdf)) # Most have 0 or 1 query
        
    # Generate Document Resubmission (Did they lose docs?)
    if 'Doc_Resubmission_Count' not in wdf.columns:
        # TPAs tend to lose more docs than In-House in this mock scenario
        wdf['Doc_Resubmission_Count'] = np.where(
            wdf['TPA_Name'] == 'In-House',
            np.random.choice([0, 1], size=len(wdf), p=[0.95, 0.05]),
            np.random.choice([0, 1, 2], size=len(wdf), p=[0.85, 0.10, 0.05])
        )

    # Generate Response Time (Hours to reply to emails)
    if 'TPA_Response_Hours' not in wdf.columns:
        wdf['TPA_Response_Hours'] = np.random.randint(2, 72, size=len(wdf))

    # --- 2. Feature Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Performance Dashboard",
        "  Cost Benefit (ROI)",
        "lose  Document Loss Tracker",
        "️  Query Response Time"
    ])

    # ---  TPA Performance Dashboard ---
    with tab1:
        st.subheader("Partner Performance Benchmarking")
        st.markdown("**Compare Settlement Times, Query Rates, and Efficiency across partners.**")
        
        # Aggregation
        tpa_stats = wdf.groupby('TPA_Name').agg({
            'Claim_ID': 'count',
            'Days_To_Payment': 'mean',
            'Query_Count': 'mean',
            'Denial_Flag': 'mean' # Error/Rejection Frequency
        }).reset_index()
        
        tpa_stats.columns = ['Partner', 'Volume', 'Avg TAT (Days)', 'Avg Queries/Claim', 'Rejection Rate']
        
        # Highlight Best/Worst
        best_tat = tpa_stats.loc[tpa_stats['Avg TAT (Days)'].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fastest Partner", best_tat['Partner'], f"{best_tat['Avg TAT (Days)']:.1f} Days")
        with col2:
            avg_network_tat = tpa_stats['Avg TAT (Days)'].mean()
            st.metric("Network Avg TAT", f"{avg_network_tat:.1f} Days")

        # Visualization: Side-by-Side Comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tpa_stats['Partner'], y=tpa_stats['Avg TAT (Days)'],
            name='Turnaround Time (Days)', marker_color='#3498db'
        ))
        fig.add_trace(go.Bar(
            x=tpa_stats['Partner'], y=tpa_stats['Avg Queries/Claim'],
            name='Queries per Claim', marker_color='#e74c3c', yaxis='y2'
        ))
        
        fig.update_layout(
            title="Efficiency: TAT vs Query Intensity",
            yaxis=dict(title="Days"),
            yaxis2=dict(title="Queries", overlaying='y', side='right'),
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tpa_stats)

    # ---  In-House vs TPA Cost Analysis ---
    with tab2:
        st.subheader("Cost Analysis & ROI Calculator")
        st.markdown("**Calculate processing costs: Direct Settlement vs. TPA Model.**")
        
        c1, c2 = st.columns(2)
        in_house_fee = c1.number_input("In-House Cost per Claim ($)", 10, 100, 25)
        tpa_fee = c2.number_input("TPA Cost per Claim ($)", 10, 100, 45)
        
        # Calculate Costs
        tpa_stats_cost = wdf.groupby('TPA_Name')['Claim_ID'].count().reset_index()
        
        # Logic: 
        # Cost = (Volume * Fee)
        # We simulate a "What-If": What if ALL claims were processed by specific entity?
        total_vol = len(wdf)
        
        cost_in_house = total_vol * in_house_fee
        cost_tpa = total_vol * tpa_fee
        savings = cost_tpa - cost_in_house
        
        st.info(f"Analysis based on total volume of {total_vol} claims.")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected In-House Cost", f"${cost_in_house:,.0f}")
        m2.metric("Projected TPA Cost", f"${cost_tpa:,.0f}")
        m3.metric("Potential Savings (In-House)", f"${savings:,.0f}", 
                 delta="Positive ROI" if savings > 0 else "Negative ROI")
        
        # Breakdown by current distribution
        wdf['Processing_Cost'] = np.where(wdf['TPA_Name'] == 'In-House', in_house_fee, tpa_fee)
        actual_spend = wdf.groupby('TPA_Name')['Processing_Cost'].sum().reset_index()
        
        fig2 = px.pie(actual_spend, values='Processing_Cost', names='TPA_Name', 
                      title="Current Spend Distribution by Partner")
        st.plotly_chart(fig2, use_container_width=True)

    # ---  Document Loss Tracker ---
    with tab3:
        st.subheader("Document Loss & Resubmission Tracker")
        st.info(f'Tracks incidents where required documents are not available at processing time.')
        st.markdown("**Flags partners with high frequency of 'Missing Document' incidents.**")
        
        # Filter claims where docs were resubmitted
        doc_issues = wdf[wdf['Doc_Resubmission_Count'] > 0]
        
        loss_rate = wdf.groupby('TPA_Name').agg({
            'Doc_Resubmission_Count': lambda x: (x > 0).mean() * 100
        }).reset_index()
        loss_rate.columns = ['Partner', 'Doc Loss Rate (%)']
        
        # Alert for high loss rate
        high_loss_partners = loss_rate[loss_rate['Doc Loss Rate (%)'] > 10]
        
        if not high_loss_partners.empty:
            for i, row in high_loss_partners.iterrows():
                st.error(f"️ **{row['Partner']}** has a high document loss rate of {row['Doc Loss Rate (%)']:.1f}%")
        
        fig3 = px.bar(loss_rate, x='Partner', y='Doc Loss Rate (%)',
                      color='Doc Loss Rate (%)', color_continuous_scale='Reds',
                      title="Percentage of Claims Requiring Document Resubmission")
        st.plotly_chart(fig3, use_container_width=True)

    # ---  TPA Query Response Time ---
    with tab4:
        st.subheader("TPA Query Response Time Monitor")
        st.markdown("**Tracks how fast TPAs respond to hospital queries (SLA: 24 Hours).**")
        
        # Benchmark line
        sla_hours = 24
        
        response_stats = wdf.groupby('TPA_Name')['TPA_Response_Hours'].mean().reset_index()
        response_stats['SLA Status'] = np.where(response_stats['TPA_Response_Hours'] > sla_hours, 'Breached', 'Met')
        
        c1, c2 = st.columns(2)
        c1.metric("Industry Standard SLA (Service Level Agreement)", f"{sla_hours} Hours")
        c2.metric("Network Avg Response", f"{response_stats['TPA_Response_Hours'].mean():.1f} Hours")
        
        fig4 = px.bar(response_stats, x='TPA_Name', y='TPA_Response_Hours',
                      color='SLA Status',
                      color_discrete_map={'Met': 'green', 'Breached': 'red'},
                      title="Average Response Time vs SLA")
        
        fig4.add_shape(type="line", line=dict(dash='dash', color='black'),
                      x0=-0.5, x1=len(response_stats)-0.5, y0=sla_hours, y1=sla_hours)
        
        st.plotly_chart(fig4, use_container_width=True)

    # Data Requirements Info
    with st.expander("️ Data Requirements for TPA Module"):
        st.markdown("""
        **Required Additional Fields:**
        1. `TPA_Name` (String) - Name of the entity processing the claim (e.g., 'MediAssist', 'In-House').
        2. `Query_Count` (Int) - Number of times additional info was requested for a claim.
        3. `Doc_Resubmission_Count` (Int) - How many times docs had to be re-sent (Proxy for document loss).
        4. `TPA_Response_Hours` (Int) - Time taken by TPA to reply to queries.
        """)