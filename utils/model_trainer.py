import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from utils.preprocessor import preprocess_data
from config.settings import ML_CONFIG

def train_models(df):
    """Train ML models for predictions"""
    
    # --- FIX STARTS HERE ---
    # 1. Capture the result in a single variable first
    processing_result = preprocess_data(df)
    
    # 2. Check if the result is None (which causes your crash)
    if processing_result is None:
        st.error("Preprocessing failed. Please check your data format.")
        return None, None
        
    # 3. Safe to unpack now
    df_processed, label_encoders = processing_result
    # --- FIX ENDS HERE ---

    feature_cols = [
        'Hospital_ID_Encoded', 'Insurer_ID_Encoded', 'Disease_Category_Encoded', 
        'CPT_Code_Encoded', 'Claim_Amount', 'Documentation_Score', 'Readmission_Flag'
    ]
    
    available_feature_cols = [col for col in feature_cols if col in df_processed.columns]
    
    target_cols = ['Paid_Amount', 'Denial_Flag']
    
    # Check if targets exist
    if not all(col in df_processed.columns for col in target_cols):
        st.error(f"Missing target columns: {target_cols}")
        return None, None

    df_model = df_processed.dropna(subset=available_feature_cols + target_cols)
    
    if df_model.empty or len(df_model) < 10:
        st.error("Insufficient data for training")
        return None, None

    X = df_model[available_feature_cols]
    
    # Model 1: Paid Amount Prediction
    y_paid = df_model['Paid_Amount']
    X_train_paid, X_test_paid, y_train_paid, y_test_paid = train_test_split(
        X, y_paid, test_size=ML_CONFIG['test_size'], random_state=ML_CONFIG['random_state']
    )
    
    paid_model = RandomForestRegressor(
        n_estimators=ML_CONFIG['rf_n_estimators'], 
        random_state=ML_CONFIG['random_state'], 
        max_depth=ML_CONFIG['rf_max_depth']
    )
    paid_model.fit(X_train_paid, y_train_paid)
    paid_pred = paid_model.predict(X_test_paid)
    paid_mae = mean_absolute_error(y_test_paid, paid_pred)
    
    # Model 2: Denial Risk Classification
    y_denial = df_model['Denial_Flag']
    X_train_denial, X_test_denial, y_train_denial, y_test_denial = train_test_split(
        X, y_denial, test_size=ML_CONFIG['test_size'], 
        random_state=ML_CONFIG['random_state'], 
        stratify=y_denial if len(y_denial.unique()) > 1 else None
    )
    
    denial_model = GradientBoostingClassifier(
        n_estimators=ML_CONFIG['gb_n_estimators'], 
        random_state=ML_CONFIG['random_state'], 
        max_depth=ML_CONFIG['gb_max_depth']
    )
    denial_model.fit(X_train_denial, y_train_denial)
    denial_pred = denial_model.predict(X_test_denial)
    denial_accuracy = accuracy_score(y_test_denial, denial_pred)
    
    models = {
        'paid_model': paid_model,
        'denial_model': denial_model,
        'label_encoders': label_encoders,
        'feature_cols': available_feature_cols,
        'metrics': {
            'paid_mae': paid_mae,
            'denial_accuracy': denial_accuracy
        }
    }
    
    return models, df_processed