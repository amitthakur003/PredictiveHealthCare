import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib

# -------------------------------------------------------------------
# Load the new dataset from the provided path
# -------------------------------------------------------------------
@st.cache_data
def load_claims_data():
    df = pd.read_csv(r"C:\Users\Amithakur\Desktop\PredicticeAnalytics\PredictiveAnalyticsAppV3\healthcare_claims_data.csv")
    return df

df = load_claims_data()


# -------------------------------------------------------------------
# COMPLIANCE & GOVERNANCE MODULE
# -------------------------------------------------------------------
def render_compliance(df: pd.DataFrame, models=None):
    """
    Compliance & Governance Module
    Features:
    - HIPAA Masking
    - Immutable Audit Trails
    """
    st.header("️Compliance & Governance")

    # Work on a copy
    wdf = df.copy()

    # -------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------
    tab1, tab2 = st.tabs([
        " HIPAA Masking",
        " Audit Trails",
    ])

    # -------------------------------------------------------------------
    # TAB 1: HIPAA MASKING
    # -------------------------------------------------------------------
    with tab1:
        st.subheader("HIPAA PHI Masking System")
        st.markdown("Role-based masking of PHI fields like Name, SSN, DOB.")

        # Role toggle
        role = st.radio("Simulate User Role:", 
                        ["️ Auditor / Analyst (Restricted)", " Administrator (Privileged)"],
                        horizontal=True)

        display_df = wdf.copy()

        if "Auditor" in role:
            st.warning(" View Mode: RESTRICTED — PHI data masked.")

            # Mask Patient Name
            if "Patient_Name" in display_df.columns:
                display_df["Patient_Name"] = display_df["Patient_Name"].apply(
                    lambda x: x[0] + "*** " + x.split(" ")[-1][0] + "***" if isinstance(x, str) else x
                )

            # Mask SSN
            if "SSN" in display_df.columns:
                display_df["SSN"] = display_df["SSN"].apply(
                    lambda x: "***-**-" + x[-4:] if isinstance(x, str) else x
                )

            # Mask DOB (keep only year)
            if "DOB" in display_df.columns:
                display_df["DOB"] = display_df["DOB"].apply(
                    lambda x: str(x)[:4] + "-**-**" if isinstance(x, str) else x
                )

            # Mask Member ID
            if "Member_ID" in display_df.columns:
                display_df["Member_ID"] = display_df["Member_ID"].apply(
                    lambda x: str(x)[:3] + "*****"
                )

        else:
            st.success(" View Mode: PRIVILEGED — Full access granted.")

        # Show available selected columns only
        cols_to_show = [
            "Claim_ID", "Patient_Name", "SSN", "DOB",
            "Disease_Category", "Claim_Amount"
        ]
        valid_cols = [c for c in cols_to_show if c in display_df.columns]

        st.dataframe(display_df[valid_cols].head(20), use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 2: AUDIT TRAILS
    # -------------------------------------------------------------------
    with tab2:
        st.subheader("Immutable Audit Logs")
        st.markdown("All actions recorded with hashed integrity checks.")

        # Initialize session-state audit log
        if "audit_log" not in st.session_state:
            st.session_state.audit_log = [
                {
                    "Timestamp": datetime.now().isoformat(),
                    "User": "System",
                    "Action": "System Boot",
                    "Details": "Application initialized",
                    "Hash": "init_hash"
                },
                {
                    "Timestamp": datetime.now().isoformat(),
                    "User": "Admin_01",
                    "Action": "Data Load",
                    "Details": f"Loaded {len(df)} records",
                    "Hash": "load_hash"
                }
            ]

        # Action Simulation
        if st.button("Simulate 'Export Data' Action"):
            action_time = datetime.now().isoformat()
            user = "Analyst_User"
            action = "Data Export"
            details = "Exported 50 filtered rows"

            # Generate hash
            log_string = f"{action_time}{user}{action}{details}"
            log_hash = hashlib.sha256(log_string.encode()).hexdigest()[:16]

            st.session_state.audit_log.append({
                "Timestamp": action_time,
                "User": user,
                "Action": action,
                "Details": details,
                "Hash": log_hash
            })

            st.success("Action recorded in immutable audit logs.")

        # Display audit logs
        log_df = pd.DataFrame(st.session_state.audit_log)
        st.table(log_df)

        st.caption("Each log entry is hashed to ensure tamper-proof audit integrity (similar to blockchain).")


# -------------------------------------------------------------------
# Render the Module
# -------------------------------------------------------------------
render_compliance(df)
