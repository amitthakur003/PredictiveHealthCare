import streamlit as st
import pandas as pd
import numpy as np
import os

@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """Load data from file path or uploaded file"""
    try:
        if file_path:
            if not os.path.exists(file_path):
                st.warning(f"'{file_path}' not found. Generating mock data...")
                data = {
                    'Claim_ID': range(100),
                    'Hospital_ID': ['HospA'] * 30 + ['HospB'] * 40 + ['HospC'] * 30,
                    'Insurer_ID': np.random.choice(['InsurerX', 'InsurerY', 'InsurerZ'], 100),
                    'Disease_Category': np.random.choice(['Cardio', 'Ortho', 'Onco', 'Pedia'], 100),
                    'CPT_Code': np.random.choice(['C1001', 'C1002', 'C2001'], 100),
                    'Claim_Amount': np.random.randint(10000, 50000, 100),
                    'Paid_Amount': np.random.randint(8000, 45000, 100),
                    'Status': np.random.choice(['Paid', 'Denied', 'Pending'], 100, p=[0.7, 0.2, 0.1]),
                    'Denial_Reason': np.random.choice(['Doc Missing', 'Code Error', 'NA'], 100),
                    'Submission_Date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=100, freq='D')),
                    'Days_To_Payment': np.random.randint(5, 60, 100),
                    'Readmission_Flag': np.random.randint(0, 2, 100),
                    'Documentation_Score': np.random.randint(1, 6, 100)
                }
                df = pd.DataFrame(data)
            else:
                df = pd.read_csv(file_path)
        elif uploaded_file:
            if uploaded_file.name.endswith('.ods'):
                df = pd.read_excel(uploaded_file, engine='odf')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        
        # Parse dates
        if 'Submission_Date' in df.columns:
            df['Submission_Date'] = pd.to_datetime(df['Submission_Date'])
            df['Month'] = df['Submission_Date'].dt.strftime('%b')
            df['Year'] = df['Submission_Date'].dt.year
        
        # Calculate metrics
        df['Payment_Ratio'] = np.where(df['Claim_Amount'] > 0, 
                                       df['Paid_Amount'] / df['Claim_Amount'], 0)
        df['Payment_Ratio'] = df['Payment_Ratio'].clip(upper=1.0)
        df['Denial_Flag'] = (df['Status'] != 'Paid').astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None