import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.ai_helper import get_ai_insights

def render_predictions(df, models):
    """Render the Predictions page"""
    st.header("Make Claim Predictions")
    
    st.markdown("""
    Use the trained ML models to predict paid amounts and assess denial risk for new claims.
    """)
    
    # Get available values from the PROCESSED/TRAINED data only
    label_encoders = models['label_encoders']
    
    available_hospitals = sorted(label_encoders['Hospital_ID'].classes_) if 'Hospital_ID' in label_encoders else []
    available_insurers = sorted(label_encoders['Insurer_ID'].classes_) if 'Insurer_ID' in label_encoders else []
    available_diseases = sorted(label_encoders['Disease_Category'].classes_) if 'Disease_Category' in label_encoders else []
    available_cpt_codes = sorted(label_encoders['CPT_Code'].classes_) if 'CPT_Code' in label_encoders else []
    
    if not all([available_hospitals, available_insurers, available_diseases, available_cpt_codes]):
        st.error("Cannot run predictions. One or more required categorical columns are missing or empty in the trained data (Hospital_ID, Insurer_ID, Disease_Category, CPT_Code).")
        return
    
    # Availability Expander
    with st.expander("Available Data in Training Set", expanded=False):
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Hospitals", len(available_hospitals))
        with col_info2:
            st.metric("Insurers", len(available_insurers))
        with col_info3:
            st.metric("Diseases", len(available_diseases))
        with col_info4:
            st.metric("CPT Codes", len(available_cpt_codes))
        st.info("Only showing values that exist in the trained model. This ensures accurate predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hospital_id = st.selectbox("Hospital ID", available_hospitals)
        insurer_id = st.selectbox("Insurer ID", available_insurers)
        disease = st.selectbox("Disease Category", available_diseases)
    
    with col2:
        cpt_code = st.selectbox("CPT Code", available_cpt_codes)
        claim_amount = st.number_input("Claim Amount ($)", min_value=1000.0, max_value=100000.0, value=25000.0)
        doc_score = st.slider("Documentation Score", min_value=1, max_value=5, value=3)
        readmission = st.selectbox("Readmission Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    if st.button("Generate Predictions", type="primary"):
        required_keys = ['Hospital_ID', 'Insurer_ID', 'Disease_Category', 'CPT_Code']
        if not all(k in label_encoders for k in required_keys):
            st.error("Encoding failed for one or more required categories during training. Cannot generate prediction.")
            return

        try:
            # Prepare encoded input
            input_data = pd.DataFrame({
                'Hospital_ID_Encoded': [label_encoders['Hospital_ID'].transform([hospital_id])[0]],
                'Insurer_ID_Encoded': [label_encoders['Insurer_ID'].transform([insurer_id])[0]],
                'Disease_Category_Encoded': [label_encoders['Disease_Category'].transform([disease])[0]],
                'CPT_Code_Encoded': [label_encoders['CPT_Code'].transform([cpt_code])[0]],
                'Claim_Amount': [claim_amount],
                'Documentation_Score': [doc_score],
                'Readmission_Flag': [readmission]
            })
            
            # Align feature columns
            input_data = input_data[models['feature_cols']]

            predicted_paid = models['paid_model'].predict(input_data)[0]
            denial_prob = models['denial_model'].predict_proba(input_data)[0]
            denial_class = models['denial_model'].predict(input_data)[0]
            
            st.success("Predictions Generated successfully!")
            
            col1, col2 = st.columns(2)
            
            predicted_paid = max(0, min(predicted_paid, claim_amount))
            payment_ratio = predicted_paid / claim_amount if claim_amount > 0 else 0
            
            # Paid Amount
            with col1:
                st.metric(
                    "Predicted Paid Amount",
                    f"${predicted_paid:,.2f}",
                    delta=f"{payment_ratio:.1%} of claim"
                )
                
                fig_paid = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_paid,
                    delta={'reference': claim_amount, 'valueformat': '$,.0f'},
                    title={'text': "Predicted Payment vs Claim"},
                    gauge={
                        'axis': {'range': [0, claim_amount * 1.1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, claim_amount * 0.7], 'color': "lightcoral"},
                            {'range': [claim_amount * 0.7, claim_amount * 0.9], 'color': "lightyellow"},
                            {'range': [claim_amount * 0.9, claim_amount * 1.1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': claim_amount
                        }
                    }
                ))
                st.plotly_chart(fig_paid, use_container_width=True)
            
            # Denial Risk
            with col2:
                denial_status = "HIGH RISK" if denial_class == 1 else "LOW RISK"
                st.metric(
                    "Denial Risk",
                    denial_status,
                    delta=f"{denial_prob[1]:.1%} probability"
                )
                
                fig_risk = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=denial_prob[1] * 100,
                    title={'text': "Denial Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if denial_class == 1 else "green"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # AI Explanation
            st.subheader("AI Explanation")
            
            data_summary = f"""
            Total Claims: {len(df):,}
            Total Hospitals: {df['Hospital_ID'].nunique()}
            Total Insurers: {df['Insurer_ID'].nunique()}
            Total Claim Amount: ${df['Claim_Amount'].sum():,.0f}
            Total Paid Amount: ${df['Paid_Amount'].sum():,.0f}
            Average Payment Ratio: {df['Payment_Ratio'].mean():.2%}
            Denial Rate: {df['Denial_Flag'].mean():.1%}
            Disease Categories: {', '.join(df['Disease_Category'].unique())}
            Avg Days to Payment: {df['Days_To_Payment'].mean():.1f}
            """

            explanation_prompt = f"""
            Explain these claim predictions in plain English:
            - Hospital: {hospital_id}, Insurer: {insurer_id}
            - Disease: {disease}, CPT Code: {cpt_code}
            - Claim Amount: ${claim_amount:,.2f}
            - Documentation Score: {doc_score}/5
            - Readmission: {'Yes' if readmission else 'No'}
            - Predicted Paid: ${predicted_paid:,.2f} ({payment_ratio:.1%})
            - Denial Risk: {denial_status} ({denial_prob[1]:.1%} probability)
            
            What do these numbers mean and what should the hospital do to optimize this claim?
            """
            
            with st.spinner("Getting AI explanation..."):
                explanation = get_ai_insights(explanation_prompt, data_summary)
                st.info(explanation)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
