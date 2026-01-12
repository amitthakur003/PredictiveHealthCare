import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_dashboard(df, models):
    """Render the executive dashboard"""
    st.header("Executive Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Claims", f"{len(df):,}", 
                 delta=f"{df['Hospital_ID'].nunique()} hospitals")
    
    with col2:
        st.metric("Total Paid Amount", f"${df['Paid_Amount'].sum():,.0f}",
                 delta=f"{len(df[df['Status']=='Paid'])/len(df)*100:.1f}% paid")
    
    with col3:
        st.metric("Avg Payment Ratio", f"{df['Payment_Ratio'].mean():.1%}")
    
    with col4:
        st.metric("Denial Rate", f"{df['Denial_Flag'].mean():.1%}",
                 delta_color="inverse")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claims by Insurer")
        insurer_data = df.groupby('Insurer_ID').agg({
            'Claim_Amount': 'sum', 'Paid_Amount': 'sum'
        }).reset_index()
        fig1 = px.bar(insurer_data, x='Insurer_ID', 
                     y=['Claim_Amount', 'Paid_Amount'], barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Claims by Disease")
        disease_data = df.groupby('Disease_Category').size().reset_index(name='Count')
        fig2 = px.pie(disease_data, values='Count', names='Disease_Category', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
