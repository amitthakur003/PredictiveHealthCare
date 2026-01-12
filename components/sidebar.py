import streamlit as st
from config.settings import FEATURE_CATEGORIES
from utils.data_loader import load_data
from utils.model_trainer import train_models

def render_sidebar():
    """Render the sidebar with data loading, model training, and navigation"""
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"

    with st.sidebar:
        st.header("Control Panel")
        
        # Data source selection
        st.subheader("Load Data")
        data_source = st.radio("Choose data source:", ["Upload File", "Use Local File"])
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload CSV/Excel file",
                type=['csv', 'xlsx', 'xls', 'ods']
            )
            
            if uploaded_file:
                if st.session_state.df is None:
                    with st.spinner("Loading data..."):
                        st.session_state.df = load_data(uploaded_file=uploaded_file)
                        if st.session_state.df is not None:
                            st.success(f"Loaded {len(st.session_state.df)} claims")
        else:
            if st.button("Load healthcare_claims_data"):
                with st.spinner("Loading data..."):
                    st.session_state.df = load_data(file_path='healthcare_claims_data.csv')
                    if st.session_state.df is not None:
                        st.success(f"Loaded {len(st.session_state.df)} claims")
        
        # Train models
        if st.session_state.df is not None and not st.session_state.models_trained:
            st.subheader("Train Models")
            if st.button("Train ML Models", type="primary"):
                with st.spinner("Training models..."):
                    models_result, df_processed_result = train_models(st.session_state.df)
                    
                    if models_result is not None:
                        st.session_state.models = models_result
                        st.session_state.df_processed = df_processed_result
                        st.session_state.models_trained = True
                        st.success("Models trained!")
                        st.balloons()
        
        # Navigation
        if st.session_state.models_trained:
            st.subheader(" Navigate")
            
            # Display features by category
            for category, pages in FEATURE_CATEGORIES.items():
                # Keep "Core Analytics" expanded by default, or if the current page is inside it
                is_expanded = (category == "Core Analytics") or (st.session_state.current_page in pages)
                
                with st.expander(category, expanded=is_expanded):
                    for page in pages:
                        # --- FIX: Update session state when clicked ---
                        # Use a unique key for every button
                        if st.button(page, key=f"nav_{page}", use_container_width=True):
                            st.session_state.current_page = page
                            st.rerun() # Force immediate update
        
        # System info
        st.markdown("---")
        if st.session_state.df is not None:
            st.caption(f"{len(st.session_state.df):,} claims loaded")
            if st.session_state.models_trained:
                st.caption("Models trained")
        
        # --- Return the persistent value from session state ---
        return st.session_state.current_page