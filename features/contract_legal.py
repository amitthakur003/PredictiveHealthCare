import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# -----------------------------------------------------
# Helper: Parse dates safely
# -----------------------------------------------------
def _to_date(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# -----------------------------------------------------
# MAIN MODULE
# -----------------------------------------------------
def render_contract_legal(df: pd.DataFrame, models=None):

    st.header("️ Contract & Legal Management")

    # Make a working copy
    wdf = df.copy()

    # Ensure all date fields are real datetime objects
    date_cols = [
        "Contract_End_Date",
        "Dispute_Open_Date",
        "Dispute_Close_Date",
        "Submission_Date",
        "Procedure_Date",
        "Service_Date",
        "Admission_Date",
        "Discharge_Date",
        "Predicted_Payment_Date"
    ]

    wdf = _to_date(wdf, date_cols)

    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "‍️  Medicolegal Risk",
        "️  Dispute Resolution",
        "  Contract Renewals",
        "️  Penalty Tracker"
    ])


    # -----------------------------------------------------
    # 1️⃣ MEDICOLEGAL RISK
    # -----------------------------------------------------
    with tab1:
        st.subheader("Medicolegal Risk Scorecard")

        if "Litigation_Flag" not in wdf.columns:
            st.info("Litigation_Flag column not present in dataset.")
            return

        lit_df = wdf[wdf["Litigation_Flag"] == 1]

        if lit_df.empty:
            st.success(" No active litigation or malpractice cases found.")
        else:
            # Risk Model
            risk_stats = wdf.groupby('Hospital_ID').agg({
                'Litigation_Flag': 'sum',
                'Claim_Amount': 'sum'
            }).reset_index()

            risk_stats.rename(columns={
                "Litigation_Flag": "Case_Count",
                "Claim_Amount": "Total_Exposure"
            }, inplace=True)

            risk_stats["Risk_Score"] = (
                risk_stats["Case_Count"] * 15
                + (risk_stats["Total_Exposure"] / 100000)
            ).clip(0, 100)

            high_risk = risk_stats.sort_values("Risk_Score", ascending=False).head(10)

            c1, c2 = st.columns(2)
            c1.metric("Total Active Cases", risk_stats["Case_Count"].sum())
            c2.metric("Total Financial Exposure", f"${risk_stats['Total_Exposure'].sum():,.0f}")

            fig = px.scatter(
                risk_stats,
                x="Case_Count",
                y="Total_Exposure",
                color="Risk_Score",
                size="Risk_Score",
                hover_name="Hospital_ID",
                color_continuous_scale="Reds",
                title="Hospital Risk Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(high_risk)


    # -----------------------------------------------------
    # 2️⃣ DISPUTE RESOLUTION
    # -----------------------------------------------------
    with tab2:
        st.subheader("Dispute Resolution Efficiency")

        if "Is_Disputed" not in wdf.columns:
            st.warning("No dispute information available in dataset.")
            return

        disputes = wdf[wdf["Is_Disputed"] == 1].copy()

        if disputes.empty:
            st.info("No disputed claims in dataset.")
        else:
            # Compute resolution days safely
            disputes["Resolution_Days"] = (
                disputes["Dispute_Close_Date"] - disputes["Dispute_Open_Date"]
            ).dt.days

            # Remove invalid negatives or NaT
            disputes = disputes[disputes["Resolution_Days"] >= 0]

            avg_resolution = disputes["Resolution_Days"].mean()
            industry_target = 30

            c1, c2 = st.columns(2)
            c1.metric("Avg Resolution Time", f"{avg_resolution:.1f} Days")
            c2.metric("Total Disputed Claims", len(disputes))

            perf = disputes.groupby("Insurer_ID")["Resolution_Days"].mean().reset_index()
            perf["Status"] = np.where(
                perf["Resolution_Days"] > industry_target, "Breach", "Compliant"
            )

            fig2 = px.bar(
                perf,
                x="Insurer_ID",
                y="Resolution_Days",
                color="Status",
                color_discrete_map={"Breach": "red", "Compliant": "green"},
                title="Avg Resolution Time by Insurer"
            )
            st.plotly_chart(fig2, use_container_width=True)


    # -----------------------------------------------------
    # 3️⃣ CONTRACT RENEWALS
    # -----------------------------------------------------
    with tab3:
        st.subheader("Contract Renewal Manager")

        if "Contract_End_Date" not in wdf.columns:
            st.warning("Contract_End_Date field missing.")
            return

        today = pd.Timestamp.now()
        contracts = wdf[["Hospital_ID", "Contract_End_Date"]].drop_duplicates()

        contracts["Days_To_Expiry"] = (contracts["Contract_End_Date"] - today).dt.days

        expiring = contracts[
            (contracts["Days_To_Expiry"] > 0) &
            (contracts["Days_To_Expiry"] <= 90)
        ].sort_values("Days_To_Expiry")

        if expiring.empty:
            st.success("No contracts expiring within 90 days.")
        else:
            st.error(f"{len(expiring)} contract(s) expiring in next 90 days")

            for idx, row in expiring.iterrows():
                with st.expander(f"{row['Hospital_ID']} — Expires in {row['Days_To_Expiry']} Days"):
                    st.write(f"**Expiry Date:** {row['Contract_End_Date'].date()}")
                    st.warning("Action Needed: Begin renewal process")

            st.dataframe(expiring)


    # -----------------------------------------------------
    # 4️⃣ PENALTY TRACKER
    # -----------------------------------------------------
    with tab4:
        st.subheader("Contract Penalty Monitor")

        if "Penalty_Amount" not in wdf.columns:
            st.info("Penalty_Amount column missing.")
            return

        total_penalty = wdf["Penalty_Amount"].sum()
        total_paid = wdf["Paid_Amount"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total Penalties", f"${total_penalty:,.0f}")
        c2.metric("Revenue Impact", f"{(total_penalty/total_paid)*100:.2f}%")

        penalty_df = wdf[wdf["Penalty_Amount"] > 0]

        if penalty_df.empty:
            st.success("No penalties found.")
        else:
            if "Days_To_Payment" in penalty_df.columns:
                penalty_df["Cause"] = np.where(
                    penalty_df["Days_To_Payment"] > 45,
                    "Late Payment SLA",
                    "Quality Breach"
                )
            else:
                penalty_df["Cause"] = "Penalty"

            fig3 = px.pie(
                penalty_df,
                values="Penalty_Amount",
                names="Cause",
                title="Penalty Distribution"
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(
                penalty_df[
                    ["Claim_ID", "Insurer_ID", "Penalty_Amount", "Cause", "Days_To_Payment"]
                    if "Days_To_Payment" in penalty_df.columns
                    else ["Claim_ID", "Insurer_ID", "Penalty_Amount", "Cause"]
                ]
            )
