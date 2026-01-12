import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_analytics(df, models):
    """Render the Advanced Analytics page"""
    st.header("Advanced Analytics")
    
    # Feature importance
    st.subheader("Feature Importance - Paid Amount Prediction")
    
    feature_mapping = {
        'Hospital_ID_Encoded': 'Hospital', 
        'Insurer_ID_Encoded': 'Insurer', 
        'Disease_Category_Encoded': 'Disease', 
        'CPT_Code_Encoded': 'CPT Code', 
        'Claim_Amount': 'Claim Amount', 
        'Documentation_Score': 'Doc Score', 
        'Readmission_Flag': 'Readmission'
    }
    
    feature_names = [feature_mapping.get(col, col) for col in models['feature_cols']]
    importance = models['paid_model'].feature_importances_
    
    fig_imp = px.bar(
        x=importance, y=feature_names,
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        color=importance,
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Hospital performance
    st.subheader("Hospital Performance Comparison")
    
    hospital_stats = df.groupby('Hospital_ID').agg({
        'Claim_Amount': 'sum',
        'Paid_Amount': 'sum',
        'Denial_Flag': 'mean',
        'Days_To_Payment': 'mean'
    }).reset_index()
    hospital_stats['Payment_Ratio'] = hospital_stats['Paid_Amount'] / hospital_stats['Claim_Amount']
    
    fig_hospital = go.Figure()
    fig_hospital.add_trace(go.Bar(
        name='Payment Ratio',
        x=hospital_stats['Hospital_ID'],
        y=hospital_stats['Payment_Ratio'],
        yaxis='y',
        marker_color='lightblue'
    ))
    fig_hospital.add_trace(go.Scatter(
        name='Denial Rate',
        x=hospital_stats['Hospital_ID'],
        y=hospital_stats['Denial_Flag'],
        yaxis='y2',
        mode='lines+markers',
        marker=dict(size=10, color='red')
    ))
    
    fig_hospital.update_layout(
        yaxis=dict(title='Payment Ratio', tickformat='.0%'),
        yaxis2=dict(title='Denial Rate', overlaying='y', side='right', tickformat='.0%'),
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_hospital, use_container_width=True)
    
    # Monthly trends
    st.subheader("Monthly Claims Trends")
    
    if 'Month' in df.columns:
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data = df.groupby('Month').agg({
            'Claim_Amount': 'sum',
            'Paid_Amount': 'sum',
            'Denial_Flag': 'mean'
        }).reindex(month_order, fill_value=0)
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            name='Claim Amount',
            x=monthly_data.index,
            y=monthly_data['Claim_Amount'],
            marker_color='lightblue'
        ))
        fig_monthly.add_trace(go.Bar(
            name='Paid Amount',
            x=monthly_data.index,
            y=monthly_data['Paid_Amount'],
            marker_color='lightgreen'
        ))
        
        fig_monthly.update_layout(barmode='group', height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)

    # ----------------------------------------------------------------------
    # 🚀 NEW SECTION ADDED: INSURER & HOSPITAL DRILL-DOWN ANALYTICS (GRAPHS)
    # ----------------------------------------------------------------------
    st.subheader("Insurer & Hospital Drill-Down Analytics")

    colA, colB = st.columns(2)

    with colA:
        selected_insurer = st.selectbox(
            "Select Insurer",
            sorted(df["Insurer_ID"].unique())
        )

    insurer_hospitals = df[df["Insurer_ID"] == selected_insurer]["Hospital_ID"].unique()

    with colB:
        selected_hospital = st.selectbox(
            "Select Hospital",
            ["All Hospitals"] + list(insurer_hospitals)
        )

    filtered = df[df["Insurer_ID"] == selected_insurer]

    if selected_hospital != "All Hospitals":
        filtered = filtered[filtered["Hospital_ID"] == selected_hospital]

    stats = filtered.groupby("Hospital_ID").agg({
        "Claim_Amount": "sum",
        "Paid_Amount": "sum",
        "Denial_Flag": "mean",
        "Days_To_Payment": "mean"
    }).reset_index()
    stats["Payment_Ratio"] = stats["Paid_Amount"] / stats["Claim_Amount"]

    # Chart 1
    st.markdown("### Claim vs Paid Amount")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=stats["Hospital_ID"], y=stats["Claim_Amount"], name="Claim Amount"))
    fig1.add_trace(go.Bar(x=stats["Hospital_ID"], y=stats["Paid_Amount"], name="Paid Amount"))
    fig1.update_layout(barmode="group", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    
    # ----------------------------------------------------------------------

    # Data table
    st.subheader("Raw Data Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.multiselect("Status", df['Status'].unique(), default=df['Status'].unique())
    with col2:
        filter_disease = st.multiselect("Disease", df['Disease_Category'].unique(), default=df['Disease_Category'].unique())
    with col3:
        filter_insurer = st.multiselect("Insurer", df['Insurer_ID'].unique(), default=df['Insurer_ID'].unique())
    
    filtered_df = df[
        (df['Status'].isin(filter_status)) &
        (df['Disease_Category'].isin(filter_disease)) &
        (df['Insurer_ID'].isin(filter_insurer))
    ]
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    st.subheader("Export Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv,
        file_name=f"claims_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
