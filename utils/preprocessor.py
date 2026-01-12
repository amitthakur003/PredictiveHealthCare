import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess data for ML models.
    Returns:
        df_processed (pd.DataFrame): Processed dataframe
        label_encoders (dict): Dictionary of fitted label encoders
    """
    if df is None or df.empty:
        return None  # This is handled by the fix in model_trainer.py
        
    df_processed = df.copy()
    label_encoders = {}
    
    # 1. Define Categorical Columns
    categorical_cols = ['Hospital_ID', 'Insurer_ID', 'Disease_Category', 'CPT_Code']
    
    # 2. Encode Categoricals
    for col in categorical_cols:
        if col in df_processed.columns:
            # Fill missing values with 'Unknown' before encoding
            df_processed[col] = df_processed[col].fillna('Unknown').astype(str)
            
            le = LabelEncoder()
            df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            
    # 3. Create Target Flags if they don't exist
    if 'Status' in df_processed.columns and 'Denial_Flag' not in df_processed.columns:
        # Assuming 'Paid' is the positive status, everything else is denial
        df_processed['Denial_Flag'] = (df_processed['Status'] != 'Paid').astype(int)
        
    # 4. Handle Numeric Cleanups
    if 'Claim_Amount' in df_processed.columns:
         df_processed['Claim_Amount'] = pd.to_numeric(df_processed['Claim_Amount'], errors='coerce').fillna(0)
         
    if 'Paid_Amount' in df_processed.columns:
         df_processed['Paid_Amount'] = pd.to_numeric(df_processed['Paid_Amount'], errors='coerce').fillna(0)
         
    if 'Documentation_Score' in df_processed.columns:
        df_processed['Documentation_Score'] = pd.to_numeric(df_processed['Documentation_Score'], errors='coerce').fillna(3) # Default score 3

    if 'Readmission_Flag' in df_processed.columns:
        df_processed['Readmission_Flag'] = pd.to_numeric(df_processed['Readmission_Flag'], errors='coerce').fillna(0)

    return df_processed, label_encoders