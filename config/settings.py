"""
Configuration settings for the Healthcare Claims Analytics System
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Healthcare Claims Analytics",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom CSS
CUSTOM_CSS = """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stAlert {
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    </style>
"""

# Feature categories and their pages
FEATURE_CATEGORIES = {
    "Core Analytics": [
        "Dashboard",
        "AI Assistant",
        "Predictions",
        "Analytics"
    ],
    "Data Quality": [
        "Data Validation",
        "Fraud Detection"
    ],
    "Financial": [
        "Denial Analysis",
        "Financial Intelligence",
        "Quality Control"
    ],
    "Operations": [
        "Operational Workflow",
        "Test & Partnership",
        "TPA Management"
    ],
    "Compliance": [
        "Compliance",
        "Government Schemes",
        "Contract & Legal"
    ],
    "Network": [
        "Network & Geography",
        "Patient Experience"
    ]
}

# AI Model configuration
AI_CONFIG = {
    "model_name": "gemini-2.5-flash",
    "max_response_length": 200
}

# ML Model configuration
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "rf_n_estimators": 100,
    "rf_max_depth": 10,
    "gb_n_estimators": 100,
    "gb_max_depth": 5
}
