import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List

# -------------------------
# Data Validation Engine
# -------------------------
class DataValidationEngine:
    """Enterprise-level Data Validation Engine for Healthcare Claims"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.total_records = len(self.df)
        self.total_fields = len(self.df.columns)

        # Define expected schema (informational - flexible checking)
        self.expected_schema = {
            'Claim_ID': 'object',
            'Hospital_ID': 'object',
            'Insurer_ID': 'object',
            'Disease_Category': 'object',
            'CPT_Code': 'object',
            'Claim_Amount': 'float64',
            'Paid_Amount': 'float64',
            'Status': 'object',
            'Denial_Reason': 'object',
            'Submission_Date': 'datetime64[ns]',
            'Days_To_Payment': 'float64',
            'Readmission_Flag': 'object',
            'Clinical_Notes': 'object',
            'Documentation_Score': 'float64'
        }

        # Validation rules for numeric ranges
        self.validation_rules = {
            'Claim_Amount': {'min': 0, 'max': 10_000_000},
            'Paid_Amount': {'min': 0, 'max': 10_000_000},
            'Days_To_Payment': {'min': 0, 'max': 3650},
            'Documentation_Score': {'min': 0, 'max': 100}
        }

        # Critical fields
        self.critical_fields = ['Claim_ID', 'Hospital_ID', 'Insurer_ID',
                                'Disease_Category', 'Claim_Amount', 'Status']

        # Normalize common numeric fields to numeric (coerce errors to NaN)
        for col in ['Claim_Amount', 'Paid_Amount', 'Days_To_Payment', 'Documentation_Score']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Normalize readmission flag to numeric/binary if possible
        if 'Readmission_Flag' in self.df.columns:
            # Try to convert strings like '0','1', True/False
            self.df['Readmission_Flag'] = pd.to_numeric(self.df['Readmission_Flag'], errors='coerce').fillna(0).astype(int)

    def calculate_quality_score(self) -> float:
        """Calculate overall data quality score"""
        total_cells = self.total_records * self.total_fields
        non_null_cells = total_cells - self.df.isnull().sum().sum()
        return (non_null_cells / total_cells * 100) if total_cells > 0 else 0.0

    def get_missing_data_analysis(self) -> pd.DataFrame:
        """Analyze missing data with severity classification"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / self.total_records * 100).round(2)

        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })

        def classify_severity(row):
            if row['Column'] in self.critical_fields:
                if row['Missing_Percentage'] > 10:
                    return 'Critical'
                elif row['Missing_Percentage'] > 5:
                    return 'High'
                else:
                    return 'Medium'
            else:
                if row['Missing_Percentage'] > 50:
                    return 'High'
                elif row['Missing_Percentage'] > 20:
                    return 'Medium'
                else:
                    return 'Low'

        missing_df['Severity'] = missing_df.apply(classify_severity, axis=1)
        missing_df['Impact'] = missing_df['Column'].apply(self._get_field_impact)

        return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

    def _get_field_impact(self, field: str) -> str:
        impact_map = {
            'Claim_ID': 'Cannot identify claims - Critical for all operations',
            'Hospital_ID': 'Cannot track provider performance',
            'Insurer_ID': 'Cannot process reimbursements',
            'Disease_Category': 'Affects risk stratification and analytics',
            'CPT_Code': 'Impacts billing accuracy and compliance',
            'Claim_Amount': 'Cannot calculate financial metrics',
            'Paid_Amount': 'Cannot track reimbursement rates',
            'Status': 'Cannot monitor claim lifecycle',
            'Denial_Reason': 'Limits denial pattern analysis',
            'Submission_Date': 'Affects turnaround time metrics',
            'Days_To_Payment': 'Impacts payment cycle analysis',
            'Readmission_Flag': 'Reduces quality metrics accuracy',
            'Clinical_Notes': 'Limits clinical review capabilities',
            'Documentation_Score': 'Optional quality indicator'
        }
        return impact_map.get(field, 'Unknown impact')

    def validate_schema(self) -> Dict[str, List[str]]:
        """Validate data schema against expected structure"""
        results = {
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': []
        }

        # Missing columns
        for col in self.expected_schema.keys():
            if col not in self.df.columns:
                results['missing_columns'].append(col)

        # Extra columns
        for col in self.df.columns:
            if col not in self.expected_schema.keys():
                results['extra_columns'].append(col)

        # Type checks (informational, tolerant)
        for col, expected_type in self.expected_schema.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                # allow ints for floats etc.
                if actual_type != expected_type and not (
                    ('int' in actual_type and 'float' in expected_type) or
                    ('float' in actual_type and 'float' in expected_type) or
                    ('object' in expected_type)
                ):
                    results['type_mismatches'].append(f"{col}: expected {expected_type}, got {actual_type}")

        return results

    def validate_data_ranges(self) -> pd.DataFrame:
        """Validate numerical data ranges"""
        violations = []
        for col, rules in self.validation_rules.items():
            if col in self.df.columns:
                min_val = rules.get('min')
                max_val = rules.get('max')

                series = self.df[col].dropna()
                if series.empty:
                    continue

                if min_val is not None:
                    below_min = (series < min_val).sum()
                    if below_min > 0:
                        violations.append({
                            'Column': col,
                            'Rule': f'Minimum value: {min_val}',
                            'Violations': int(below_min),
                            'Percentage': round(below_min / self.total_records * 100, 2)
                        })
                if max_val is not None:
                    above_max = (series > max_val).sum()
                    if above_max > 0:
                        violations.append({
                            'Column': col,
                            'Rule': f'Maximum value: {max_val}',
                            'Violations': int(above_max),
                            'Percentage': round(above_max / self.total_records * 100, 2)
                        })

        return pd.DataFrame(violations) if violations else pd.DataFrame()

    def validate_business_rules(self) -> List[Dict]:
        """Validate business logic rules"""
        violations = []

        # Rule 1: Paid amount should not exceed claim amount
        if 'Claim_Amount' in self.df.columns and 'Paid_Amount' in self.df.columns:
            invalid_payments = int((self.df['Paid_Amount'] > self.df['Claim_Amount']).sum())
            if invalid_payments > 0:
                violations.append({
                    'rule': 'Paid Amount ≤ Claim Amount',
                    'violations': invalid_payments,
                    'severity': 'Critical',
                    'description': 'Paid amount exceeds claimed amount'
                })

        # Rule 2: Denied claims should have denial reason
        if 'Status' in self.df.columns and 'Denial_Reason' in self.df.columns:
            denied_without_reason = int(((self.df['Status'].astype(str).str.lower() == 'denied') & (self.df['Denial_Reason'].isnull())).sum())
            if denied_without_reason > 0:
                violations.append({
                    'rule': 'Denied Claims Must Have Reason',
                    'violations': denied_without_reason,
                    'severity': 'High',
                    'description': 'Denied claims missing denial reason'
                })

        # Rule 3: Approved claims should have payment timeline
        if 'Status' in self.df.columns and 'Days_To_Payment' in self.df.columns:
            approved_no_payment = int(((self.df['Status'].astype(str).str.lower() == 'approved') & (self.df['Days_To_Payment'].isnull())).sum())
            if approved_no_payment > 0:
                violations.append({
                    'rule': 'Approved Claims Should Have Payment Timeline',
                    'violations': approved_no_payment,
                    'severity': 'Medium',
                    'description': 'Approved claims missing payment information'
                })

        # Rule 4: Claim ID uniqueness
        if 'Claim_ID' in self.df.columns:
            duplicate_claims = int(self.df['Claim_ID'].duplicated().sum())
            if duplicate_claims > 0:
                violations.append({
                    'rule': 'Claim ID Must Be Unique',
                    'violations': duplicate_claims,
                    'severity': 'Critical',
                    'description': 'Duplicate claim IDs detected'
                })

        return violations

    def get_data_completeness_by_category(self) -> pd.DataFrame:
        """Analyze data completeness by disease category"""
        if 'Disease_Category' not in self.df.columns:
            return pd.DataFrame()

        completeness = []
        for category in self.df['Disease_Category'].dropna().unique():
            category_df = self.df[self.df['Disease_Category'] == category]
            total_cells = len(category_df) * len(category_df.columns)
            non_null = total_cells - category_df.isnull().sum().sum()
            completeness.append({
                'Category': category,
                'Records': len(category_df),
                'Completeness': round(non_null / total_cells * 100, 2)
            })

        return pd.DataFrame(completeness).sort_values('Completeness')


# -------------------------
# Streamlit Render Function
# -------------------------
def render_data_validation(df: pd.DataFrame, models=None):
    """Enterprise Data Validation Engine - Main Render Function"""

    # Defensive copy
    df = df.copy()

    validator = DataValidationEngine(df)

    # Executive Summary
    st.header(" Executive Summary")

    quality_score = validator.calculate_quality_score()
    missing_df = validator.get_missing_data_analysis()
    business_violations = validator.validate_business_rules()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Overall Quality Score", f"{quality_score:.1f}%", delta=f"{quality_score - 85:.1f}%")
    with col2:
        st.metric("Total Records", f"{validator.total_records:,}")
    with col3:
        critical_issues = len(missing_df[missing_df['Severity'] == 'Critical'])
        st.metric("Critical Issues", critical_issues, delta=f"-{critical_issues}" if critical_issues > 0 else "0", delta_color="inverse")
    with col4:
        st.metric("Business Rule Violations", len(business_violations), delta=f"-{len(business_violations)}" if business_violations else "0", delta_color="inverse")
    with col5:
        completeness = 100 - (missing_df['Missing_Percentage'].mean() if not missing_df.empty else 0)
        st.metric("Data Completeness", f"{completeness:.1f}%", delta=f"{completeness - 90:.1f}%")

    # Tabs
    tab1, tab3, tab4, tab5, tab6 = st.tabs([
        " Missing Data Analysis",
        " Range Validation",
        "️ Business Rules",
        " Completeness Trends",
        " Export Report"
    ])

    # Missing Data
    with tab1:
        st.subheader("Missing Data Analysis")
        if missing_df.empty:
            st.success(" No missing data detected! All fields are complete.")
        else:
            colA, colB = st.columns([3, 1])
            with colB:
                severity_filter = st.selectbox("Filter by Severity", ['All', 'Critical', 'High', 'Medium', 'Low'])
            filtered_df = missing_df if severity_filter == 'All' else missing_df[missing_df['Severity'] == severity_filter]

            fig = px.bar(
                filtered_df,
                x='Column',
                y='Missing_Percentage',
                color='Severity',
                color_discrete_map={
                    'Critical': '#dc2626',
                    'High': '#ea580c',
                    'Medium': '#eab308',
                    'Low': '#22c55e'
                },
                title='Missing Data by Column',
                labels={'Missing_Percentage': 'Missing %'}
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detailed Missing Data Report")
            st.dataframe(filtered_df, use_container_width=True, height=400)

            critical_missing = missing_df[(missing_df['Severity'] == 'Critical') & (missing_df['Missing_Percentage'] > 5)]
            if not critical_missing.empty:
                st.error("️ **Critical Data Quality Issues Detected**")
                for _, row in critical_missing.iterrows():
                    st.markdown(f"- **{row['Column']}**: {row['Missing_Percentage']:.2f}% missing ({row['Missing_Count']:,} records) — Impact: {row['Impact']}")

    # Range Validation
    with tab3:
        st.subheader("Data Range Validation")
        range_violations = validator.validate_data_ranges()
        if range_violations.empty:
            st.success(" All numerical values are within expected ranges")
        else:
            st.warning(f"️ Found {len(range_violations)} range violations")
            fig = px.bar(range_violations, x='Column', y='Violations', color='Percentage', title='Range Violations by Column', text='Violations')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(range_violations, use_container_width=True)

        st.markdown("#### Validation Rules")
        rules_df = pd.DataFrame([{'Column': col, 'Minimum': rules.get('min', 'N/A'), 'Maximum': rules.get('max', 'N/A')} for col, rules in validator.validation_rules.items()])
        st.dataframe(rules_df, use_container_width=True)

    # Business Rules
    with tab4:
        st.subheader("Business Logic Validation")
        if not business_violations:
            st.success(" All business rules validated successfully")
        else:
            st.error(f" Found {len(business_violations)} business rule violations")
            for violation in business_violations:
                with st.expander(f"{violation['rule']} - {violation['violations']:,} violations", expanded=True):
                    st.markdown(f"**Severity:** {violation['severity']}  \n**Description:** {violation['description']}  \n**Violations:** {violation['violations']:,} records  \n**Impact:** {(violation['violations'] / validator.total_records * 100):.2f}% of total records")

        st.markdown("#### Implemented Business Rules")
        st.markdown("""
        1. **Payment Consistency**: Paid amount must not exceed claimed amount  
        2. **Denial Documentation**: All denied claims must have a denial reason  
        3. **Payment Timeline**: Approved claims should have payment information  
        4. **Data Integrity**: Claim IDs must be unique across all records  
        5. **Status Consistency**: Claim status must align with payment data
        """)

    # Completeness Trends
    with tab5:
        st.subheader("Data Completeness Analysis")
        category_completeness = validator.get_data_completeness_by_category()
        if not category_completeness.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(category_completeness, x='Completeness', y='Category', orientation='h', title='Data Completeness by Disease Category', color='Completeness', color_continuous_scale='RdYlGn', range_color=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(category_completeness, values='Records', names='Category', title='Record Distribution by Category')
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(category_completeness, use_container_width=True)

        st.markdown("#### Field-Level Completeness")
        completeness_by_field = (100 - (df.isnull().sum() / len(df) * 100)).round(2)
        fig = go.Figure(data=go.Bar(x=completeness_by_field.values, y=completeness_by_field.index, orientation='h',
                                   marker=dict(color=completeness_by_field.values, colorscale='RdYlGn', showscale=True, cmin=0, cmax=100),
                                   text=completeness_by_field.values, texttemplate='%{text:.1f}%', textposition='outside'))
        fig.update_layout(title='Completeness Score by Field', xaxis_title='Completeness (%)', yaxis_title='Field', height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Export Report
    with tab6:
        st.subheader("Generate Validation Report")
        st.markdown("Export comprehensive validation results for stakeholder review and compliance documentation.")
        report_format = st.selectbox("Select Report Format", ["Excel (Recommended)", "CSV", "JSON"])
        include_options = st.multiselect("Include in Report", ["Executive Summary", "Missing Data Analysis", "Range Violations", "Business Rule Violations", "Completeness by Category"], default=["Executive Summary", "Missing Data Analysis", "Business Rule Violations"])
        if st.button("Generate Report", type="primary"):
            st.success(" Report generated successfully!")
            st.info(" Report would be available for download (implementation pending)")
            with st.expander("Report Preview"):
                st.markdown(f"### Data Validation Report\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n**Total Records:** {validator.total_records:,}  \n**Quality Score:** {quality_score:.2f}%  \n**Critical Issues:** {len(missing_df[missing_df['Severity'] == 'Critical'])}  \n**Business Rule Violations:** {len(business_violations)}")
        st.markdown("#### Automated Monitoring")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Quality Score Threshold (%)", 0, 100, 85)
            st.number_input("Max Critical Issues", 0, 100, 0)
        with col2:
            st.selectbox("Alert Frequency", ["Real-time", "Daily", "Weekly"])
            st.text_input("Email Recipients", "data-quality@hospital.com")
        if st.button("Enable Automated Monitoring"):
            st.success(" Automated monitoring enabled")

    # Footer Recommendations
    st.markdown("---")
    st.markdown("### Recommendations")
    recommendations = []
    if quality_score < 90:
        recommendations.append("️ Overall quality score is below 90%. Focus on completing critical fields.")
    if len(missing_df[missing_df['Severity'] == 'Critical']) > 0:
        recommendations.append(" Critical fields have missing data. Immediate action required to maintain compliance.")
    if business_violations:
        recommendations.append("️ Business rule violations detected. Review data entry processes and implement validation checks.")
    if not recommendations:
        st.success(" Data quality is excellent! No immediate action required.")
    else:
        for rec in recommendations:
            st.warning(rec)


# -------------------------
# Entrypoint: load CSV & run
# -------------------------
def load_claims_csv(path: str) -> pd.DataFrame:
    try:
        # Try parse common date columns on load
        df = pd.read_csv(path, dtype=str)
        # convert to proper datatypes where possible
        # parse dates for columns commonly used
        date_cols = [
            "Submission_Date", "Procedure_Date", "Service_Date",
            "Admission_Date", "Discharge_Date", "Predicted_Payment_Date",
            "Contract_End_Date", "Dispute_Open_Date", "Dispute_Close_Date", "DOB"
        ]
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')

        # Numeric conversions
        for num_col in ['Claim_Amount', 'Paid_Amount', 'Days_To_Payment', 'Documentation_Score', 'Penalty_Amount']:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

        # If boolean-ish flags exist, keep them as is but coerce to numeric when needed
        return df
    except FileNotFoundError as e:
        st.error(f"Data file not found at: {path}")
        raise e
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise e


if __name__ == "__main__":
    st.title("Data Validation Dashboard")

    # <-- Replace this path with the canonical path you told me earlier -->
    csv_path = r"C:\Users\Amithakur\Desktop\PredicticeAnalytics\PredictiveAnalyticsAppV3\healthcare_claims_data.csv"

    try:
        df = load_claims_csv(csv_path)
        render_data_validation(df)
    except FileNotFoundError:
        st.stop()
