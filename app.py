import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config.settings import PAGE_CONFIG, CUSTOM_CSS

# Import utilities
from utils.data_loader import load_data
from utils.model_trainer import train_models

# Import components
from components.sidebar import render_sidebar

# Import all features
from features.dashboard import render_dashboard
from features.ai_assistant import render_ai_assistant
from features.predictions import render_predictions
from features.analytics import render_analytics
from features.data_validation import render_data_validation
from features.denial_analysis import render_denial_analysis
from features.financial_intelligence import render_financial_intelligence
from features.operational_workflow import render_operational_workflow
from features.test_partnership import render_test_partnership
from features.compliance import render_compliance
from features.quality_control import render_quality_control
from features.fraud_detection import render_fraud_detection
from features.government_schemes import render_government_schemes
from features.network_geography import render_network_geography
from features.tpa_management import render_tpa_management
from features.patient_experience import render_patient_experience
from features.contract_legal import render_contract_legal

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main header
st.markdown('<div class="main-header">Healthcare Claims Analytics System</div>', unsafe_allow_html=True)

# Render sidebar and get selected page
page = render_sidebar()

# Main Content Area
if st.session_state.df is None:
    st.info("Please load your claims data file to get started.")
    st.markdown("""
    ### Welcome to Healthcare Claims Analytics System
    
    This enterprise-level system provides comprehensive healthcare claims analytics with:
    
    #### Core Features
    - **Analytics & Dashboard** - Executive insights and KPIs
    - **AI Assistant** - Intelligent insights and recommendations
    - **Predictions Engine** - ML-powered claim predictions
    - **Advanced Analytics** - Deep-dive analysis and trends
    
    #### Specialized Features
    - **Data Validation** - Automated data quality checks
    - **Denial Analysis** - Risk prediction and root cause analysis
    - **Financial Intelligence** - ROI and revenue optimization
    - **Operational Workflow** - Process efficiency monitoring
    - **Fraud Detection** - Anomaly detection and alerts
    - **Compliance & Governance** - Regulatory compliance tracking
    - **Quality & Cost Control** - Quality metrics and cost analysis
    - **Government Schemes** - Scheme tracking and reimbursement
    - **Network & Geography** - Provider network analysis
    - **TPA Management** - Third-party administrator metrics
    - **Patient Experience** - NPS and satisfaction tracking
    - **Contract & Legal** - Contract management and tracking
    
    """)

elif not st.session_state.models_trained:
    st.warning("Please train the ML models using the sidebar to enable all features.")
    
    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    st.info(f"Dataset contains {len(st.session_state.df)} claims from {st.session_state.df['Hospital_ID'].nunique()} hospitals")

else:
    # Route to appropriate feature page
    if page == "Dashboard":
        render_dashboard(st.session_state.df, st.session_state.models)
    
    elif page == "AI Assistant":
        render_ai_assistant(st.session_state.df, st.session_state.models)
    
    elif page == "Predictions":
        render_predictions(st.session_state.df, st.session_state.models)
    
    elif page == "Analytics":
        render_analytics(st.session_state.df, st.session_state.models)
    
    elif page == "Data Validation":
        render_data_validation(st.session_state.df, st.session_state.models)
    
    elif page == "Denial Analysis":
        render_denial_analysis(st.session_state.df, st.session_state.models)
    
    elif page == "Financial Intelligence":
        render_financial_intelligence(st.session_state.df, st.session_state.models)
    
    elif page == "Operational Workflow":
        render_operational_workflow(st.session_state.df, st.session_state.models)
    
    elif page == "Test & Partnership":
        render_test_partnership(st.session_state.df, st.session_state.models)
    
    elif page == "Compliance":
        render_compliance(st.session_state.df, st.session_state.models)
    
    elif page == "Quality Control":
        render_quality_control(st.session_state.df, st.session_state.models)
    
    elif page == "Fraud Detection":
        render_fraud_detection(st.session_state.df, st.session_state.models)
    
    elif page == "Government Schemes":
        render_government_schemes(st.session_state.df, st.session_state.models)
    
    elif page == "Network & Geography":
        render_network_geography(st.session_state.df, st.session_state.models)
    
    elif page == "TPA Management":
        render_tpa_management(st.session_state.df, st.session_state.models)
    
    elif page == "Patient Experience":
        render_patient_experience(st.session_state.df, st.session_state.models)
    
    elif page == "Contract & Legal":
        render_contract_legal(st.session_state.df, st.session_state.models)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Healthcare Claims Analytics System | Amit Thakur (Shelter)</p>
    <p>Built with Python, Streamlit, scikit-learn, and Google Gemini</p>
</div>
""", unsafe_allow_html=True)
