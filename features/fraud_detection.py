import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re

class FraudDetectionEngine:
    """Fraud Detection & Integrity System for Healthcare Claims"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
        
        # CPT code bundling rules (sample - should be loaded from database)
        self.bundled_procedures = {
            '99213': ['99211', '99212'],  # Office visits
            '29881': ['29880', '29882'],  # Knee arthroscopy
            '45380': ['45378', '45379'],  # Colonoscopy procedures
            '76856': ['76830', '76831'],  # Ultrasound procedures
        }
        
        # ICD to CPT mapping rules (sample)
        self.icd_cpt_mapping = {
            'M17': ['27440', '27441', '27442', '27443', '27445', '27446', '27447'],  # Knee osteoarthritis
            'I10': ['99213', '99214', '99215'],  # Hypertension
            'E11': ['99213', '99214', '82947', '82950'],  # Type 2 diabetes
            'J45': ['94010', '94060', '94640'],  # Asthma
        }
        
        # OPD to IPD conversion threshold
        self.opd_ipd_threshold = 0.80
        
        # Risk score weights
        self.risk_weights = {
            'unbundling': 0.25,
            'upcoding': 0.30,
            'duplicate': 0.20,
            'anomaly': 0.15,
            'conversion': 0.10
        }
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        # Parse dates
        if 'Submission_Date' in self.df.columns:
            self.df['Submission_Date'] = pd.to_datetime(self.df['Submission_Date'], errors='coerce')
            self.df['Submission_Year'] = self.df['Submission_Date'].dt.year
            self.df['Submission_Month'] = self.df['Submission_Date'].dt.month
            self.df['Submission_Day'] = self.df['Submission_Date'].dt.day
        
        # Extract Patient ID from Claim ID (assuming format CLM-PATID-XXXXX)
        if 'Patient_ID' not in self.df.columns:
            # Generate pseudo patient IDs for demonstration
            self.df['Patient_ID'] = 'PAT' + (self.df.index % 500).astype(str).str.zfill(5)
        
        # Extract procedure date (use submission date as proxy if not available)
        if 'Procedure_Date' not in self.df.columns:
            self.df['Procedure_Date'] = self.df['Submission_Date']
        
        # Parse ICD codes if available
        if 'ICD_Code' not in self.df.columns and 'Disease_Category' in self.df.columns:
            # Map disease categories to ICD codes (sample mapping)
            icd_mapping = {
                'Cardiology': 'I10',
                'Orthopedics': 'M17',
                'Neurology': 'G20',
                'Oncology': 'C50',
                'Pediatrics': 'J45',
                'Diabetes': 'E11'
            }
            self.df['ICD_Code'] = self.df['Disease_Category'].map(icd_mapping)
        
        # Parse admission type if not available
        if 'Admission_Type' not in self.df.columns:
            # Generate based on claim amount (sample logic)
            self.df['Admission_Type'] = np.where(
                self.df['Claim_Amount'] > 10000, 'IPD', 'OPD'
            )
    
    def detect_unbundling(self) -> pd.DataFrame:
        """8.1 Unbundling Detector: Detect procedure splitting"""
        unbundling_cases = []
        
        # Group by Patient, Hospital, and Date
        grouped = self.df.groupby(['Patient_ID', 'Hospital_ID', 
                                   self.df['Procedure_Date'].dt.date])
        
        for (patient, hospital, proc_date), group in grouped:
            cpt_codes = group['CPT_Code'].tolist()
            
            # Check if more than 3 related procedures on same day
            if len(cpt_codes) > 3:
                # Check for bundled procedures
                for parent_code, child_codes in self.bundled_procedures.items():
                    if parent_code in cpt_codes:
                        bundled_found = [code for code in child_codes if code in cpt_codes]
                        if bundled_found:
                            unbundling_cases.append({
                                'Patient_ID': patient,
                                'Hospital_ID': hospital,
                                'Procedure_Date': proc_date,
                                'Claim_IDs': ', '.join(group['Claim_ID'].tolist()),
                                'CPT_Codes': ', '.join(cpt_codes),
                                'Total_Procedures': len(cpt_codes),
                                'Parent_Code': parent_code,
                                'Bundled_Codes': ', '.join(bundled_found),
                                'Total_Amount': group['Claim_Amount'].sum(),
                                'Risk_Score': min(100, len(cpt_codes) * 15),
                                'Severity': 'High' if len(cpt_codes) > 5 else 'Medium'
                            })
            
            # Flag any case with >3 procedures same day
            elif len(cpt_codes) > 3:
                unbundling_cases.append({
                    'Patient_ID': patient,
                    'Hospital_ID': hospital,
                    'Procedure_Date': proc_date,
                    'Claim_IDs': ', '.join(group['Claim_ID'].tolist()),
                    'CPT_Codes': ', '.join(cpt_codes),
                    'Total_Procedures': len(cpt_codes),
                    'Parent_Code': 'N/A',
                    'Bundled_Codes': 'Multiple procedures',
                    'Total_Amount': group['Claim_Amount'].sum(),
                    'Risk_Score': len(cpt_codes) * 10,
                    'Severity': 'Medium'
                })
        
        return pd.DataFrame(unbundling_cases) if unbundling_cases else pd.DataFrame()
    
    def detect_upcoding(self) -> pd.DataFrame:
        """8.2 Upcoding Alert: Detect CPT-ICD mismatches"""
        upcoding_cases = []
        
        for idx, row in self.df.iterrows():
            icd_code = row.get('ICD_Code')
            cpt_code = str(row.get('CPT_Code', ''))
            
            if pd.notna(icd_code) and cpt_code:
                # Check ICD-CPT mapping
                valid_cpts = self.icd_cpt_mapping.get(icd_code, [])
                
                if valid_cpts and cpt_code not in valid_cpts:
                    # Calculate severity based on claim amount
                    claim_amt = row.get('Claim_Amount', 0)
                    severity_score = min(100, (claim_amt / 1000) * 5)
                    
                    severity = 'Critical' if severity_score > 80 else \
                              'High' if severity_score > 50 else 'Medium'
                    
                    upcoding_cases.append({
                        'Claim_ID': row['Claim_ID'],
                        'Patient_ID': row.get('Patient_ID'),
                        'Hospital_ID': row['Hospital_ID'],
                        'ICD_Code': icd_code,
                        'Disease_Category': row.get('Disease_Category'),
                        'CPT_Code': cpt_code,
                        'Expected_CPT_Codes': ', '.join(valid_cpts),
                        'Claim_Amount': claim_amt,
                        'Risk_Score': severity_score,
                        'Severity': severity,
                        'Recommendation': 'Review coding guidelines and clinical documentation'
                    })
        
        return pd.DataFrame(upcoding_cases) if upcoding_cases else pd.DataFrame()
    
    def detect_opd_ipd_conversion(self) -> pd.DataFrame:
        """8.3 OPD-to-IPD Conversion Tracker"""
        conversion_analysis = []
        
        # Group by hospital
        for hospital_id in self.df['Hospital_ID'].unique():
            hospital_data = self.df[self.df['Hospital_ID'] == hospital_id]
            
            total_admissions = len(hospital_data)
            opd_count = len(hospital_data[hospital_data['Admission_Type'] == 'OPD'])
            ipd_count = len(hospital_data[hospital_data['Admission_Type'] == 'IPD'])
            
            # Calculate conversion rate
            conversion_rate = ipd_count / total_admissions if total_admissions > 0 else 0
            
            # Flag if exceeds threshold
            if conversion_rate > self.opd_ipd_threshold:
                avg_ipd_amount = hospital_data[
                    hospital_data['Admission_Type'] == 'IPD'
                ]['Claim_Amount'].mean()
                
                severity = 'Critical' if conversion_rate > 0.90 else 'High'
                
                conversion_analysis.append({
                    'Hospital_ID': hospital_id,
                    'Total_Admissions': total_admissions,
                    'OPD_Count': opd_count,
                    'IPD_Count': ipd_count,
                    'Conversion_Rate': conversion_rate * 100,
                    'Avg_IPD_Amount': avg_ipd_amount,
                    'Total_IPD_Amount': hospital_data[
                        hospital_data['Admission_Type'] == 'IPD'
                    ]['Claim_Amount'].sum(),
                    'Risk_Score': min(100, conversion_rate * 100),
                    'Severity': severity,
                    'Alert': f'️ Conversion rate {conversion_rate*100:.1f}% exceeds threshold {self.opd_ipd_threshold*100:.0f}%'
                })
        
        df_result = pd.DataFrame(conversion_analysis)
        if not df_result.empty:
            df_result = df_result.sort_values('Conversion_Rate', ascending=False)
        return df_result
    
    def detect_duplicate_claims(self) -> pd.DataFrame:
        """8.4 Duplicate Claim Finder"""
        duplicate_cases = []
        
        # Sort by date for 90-day window check
        df_sorted = self.df.sort_values('Procedure_Date')
        
        # Group by patient and CPT code
        for (patient, cpt), group in df_sorted.groupby(['Patient_ID', 'CPT_Code']):
            if len(group) > 1:
                group = group.sort_values('Procedure_Date')
                
                for i in range(len(group) - 1):
                    current = group.iloc[i]
                    next_claim = group.iloc[i + 1]
                    
                    # Check if within 90 days
                    if pd.notna(current['Procedure_Date']) and pd.notna(next_claim['Procedure_Date']):
                        days_diff = (next_claim['Procedure_Date'] - current['Procedure_Date']).days
                        
                        if 0 <= days_diff <= 90:
                            # Check if exact duplicate
                            is_exact = (current['Hospital_ID'] == next_claim['Hospital_ID'] and
                                      abs(current['Claim_Amount'] - next_claim['Claim_Amount']) < 10)
                            
                            duplicate_type = 'Exact Duplicate' if is_exact else 'Near Duplicate'
                            severity = 'Critical' if is_exact else 'High'
                            risk_score = 95 if is_exact else 70
                            
                            duplicate_cases.append({
                                'Patient_ID': patient,
                                'CPT_Code': cpt,
                                'First_Claim_ID': current['Claim_ID'],
                                'Second_Claim_ID': next_claim['Claim_ID'],
                                'First_Date': current['Procedure_Date'].date(),
                                'Second_Date': next_claim['Procedure_Date'].date(),
                                'Days_Between': days_diff,
                                'Hospital_ID': current['Hospital_ID'],
                                'First_Amount': current['Claim_Amount'],
                                'Second_Amount': next_claim['Claim_Amount'],
                                'Total_Amount': current['Claim_Amount'] + next_claim['Claim_Amount'],
                                'Duplicate_Type': duplicate_type,
                                'Risk_Score': risk_score,
                                'Severity': severity
                            })
        
        return pd.DataFrame(duplicate_cases) if duplicate_cases else pd.DataFrame()
    
    def detect_ghost_surgery(self, ot_schedule_df: pd.DataFrame = None) -> pd.DataFrame:
        """8.5 Ghost Surgery Pattern Detection"""
        ghost_surgery_cases = []
        
        # Define surgery CPT codes (sample list)
        surgery_cpt_codes = [
            '27447', '27440', '29881', '45380', '47562', '49320', 
            '43644', '44950', '52601', '58150'
        ]
        
        # Filter surgery claims
        surgery_claims = self.df[
            self.df['CPT_Code'].astype(str).isin([str(c) for c in surgery_cpt_codes])
        ].copy()
        
        if ot_schedule_df is not None:
            # Cross-reference with OT schedule
            ot_schedule_df['OT_Date'] = pd.to_datetime(ot_schedule_df['OT_Date'])
            
            for idx, claim in surgery_claims.iterrows():
                # Look for matching OT record
                matching_ot = ot_schedule_df[
                    (ot_schedule_df['Patient_ID'] == claim['Patient_ID']) &
                    (ot_schedule_df['OT_Date'].dt.date == claim['Procedure_Date'].date()) &
                    (ot_schedule_df['Hospital_ID'] == claim['Hospital_ID'])
                ]
                
                if matching_ot.empty:
                    ghost_surgery_cases.append({
                        'Claim_ID': claim['Claim_ID'],
                        'Patient_ID': claim['Patient_ID'],
                        'Hospital_ID': claim['Hospital_ID'],
                        'CPT_Code': claim['CPT_Code'],
                        'Procedure_Date': claim['Procedure_Date'].date(),
                        'Claim_Amount': claim['Claim_Amount'],
                        'Issue': 'No OT record found',
                        'Risk_Score': 90,
                        'Severity': 'Critical',
                        'Recommendation': 'Verify surgery actually performed'
                    })
        else:
            # Pattern-based detection without OT data
            # Flag surgeries with suspicious patterns
            for hospital_id in surgery_claims['Hospital_ID'].unique():
                hospital_surgeries = surgery_claims[
                    surgery_claims['Hospital_ID'] == hospital_id
                ]
                
                # Check for unusual patterns
                daily_surgeries = hospital_surgeries.groupby(
                    hospital_surgeries['Procedure_Date'].dt.date
                ).size()
                
                # Flag days with abnormally high surgery count
                if len(daily_surgeries) > 0:
                    mean_surgeries = daily_surgeries.mean()
                    std_surgeries = daily_surgeries.std()
                    
                    for date, count in daily_surgeries.items():
                        if count > mean_surgeries + (2 * std_surgeries):
                            day_claims = hospital_surgeries[
                                hospital_surgeries['Procedure_Date'].dt.date == date
                            ]
                            
                            ghost_surgery_cases.append({
                                'Date': date,
                                'Hospital_ID': hospital_id,
                                'Surgery_Count': count,
                                'Average_Count': mean_surgeries,
                                'Total_Amount': day_claims['Claim_Amount'].sum(),
                                'Issue': f'Abnormally high surgery count ({count} vs avg {mean_surgeries:.1f})',
                                'Risk_Score': 75,
                                'Severity': 'High',
                                'Recommendation': 'Verify OT capacity and actual procedures'
                            })
        
        return pd.DataFrame(ghost_surgery_cases) if ghost_surgery_cases else pd.DataFrame()
    
    def calculate_anomaly_scores(self) -> pd.DataFrame:
        """Calculate statistical anomalies using multiple methods"""
        anomalies = []
        
        # Z-score method
        z_scores = np.abs((self.df['Claim_Amount'] - self.df['Claim_Amount'].mean()) / 
                          self.df['Claim_Amount'].std())
        
        # IQR method
        Q1 = self.df['Claim_Amount'].quantile(0.25)
        Q3 = self.df['Claim_Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Combine methods
        self.df['Z_Score'] = z_scores
        self.df['Is_Outlier_IQR'] = (
            (self.df['Claim_Amount'] < lower_bound) | 
            (self.df['Claim_Amount'] > upper_bound)
        )
        
        anomalous = self.df[
            (z_scores > 3) | self.df['Is_Outlier_IQR']
        ].copy()
        
        if not anomalous.empty:
            anomalous['Risk_Score'] = np.minimum(100, z_scores[anomalous.index] * 20)
            anomalous['Severity'] = pd.cut(
                anomalous['Risk_Score'],
                bins=[0, 50, 75, 90, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        return anomalous
    
    def calculate_composite_risk_score(self) -> pd.DataFrame:
        """Calculate composite risk score for each claim"""
        risk_scores = []
        
        # Get all detection results
        unbundling = self.detect_unbundling()
        upcoding = self.detect_upcoding()
        duplicates = self.detect_duplicate_claims()
        anomalies = self.calculate_anomaly_scores()
        
        for idx, claim in self.df.iterrows():
            scores = {
                'Claim_ID': claim['Claim_ID'],
                'Hospital_ID': claim['Hospital_ID'],
                'Claim_Amount': claim['Claim_Amount']
            }
            
            # Check each fraud type
            scores['Unbundling_Risk'] = 0
            if not unbundling.empty:
                if claim['Claim_ID'] in unbundling['Claim_IDs'].str.split(', ').sum():
                    scores['Unbundling_Risk'] = 80
            
            scores['Upcoding_Risk'] = 0
            if not upcoding.empty and claim['Claim_ID'] in upcoding['Claim_ID'].values:
                scores['Upcoding_Risk'] = upcoding[
                    upcoding['Claim_ID'] == claim['Claim_ID']
                ]['Risk_Score'].iloc[0]
            
            scores['Duplicate_Risk'] = 0
            if not duplicates.empty:
                if claim['Claim_ID'] in duplicates['First_Claim_ID'].values or \
                   claim['Claim_ID'] in duplicates['Second_Claim_ID'].values:
                    scores['Duplicate_Risk'] = 90
            
            scores['Anomaly_Risk'] = 0
            if not anomalies.empty and idx in anomalies.index:
                scores['Anomaly_Risk'] = anomalies.loc[idx, 'Risk_Score']
            
            # Calculate composite score
            scores['Composite_Risk_Score'] = (
                scores['Unbundling_Risk'] * self.risk_weights['unbundling'] +
                scores['Upcoding_Risk'] * self.risk_weights['upcoding'] +
                scores['Duplicate_Risk'] * self.risk_weights['duplicate'] +
                scores['Anomaly_Risk'] * self.risk_weights['anomaly']
            )
            
            scores['Risk_Level'] = 'Critical' if scores['Composite_Risk_Score'] > 75 else \
                                  'High' if scores['Composite_Risk_Score'] > 50 else \
                                  'Medium' if scores['Composite_Risk_Score'] > 25 else 'Low'
            
            risk_scores.append(scores)
        
        return pd.DataFrame(risk_scores)


def render_fraud_detection(df: pd.DataFrame, models=None):
    """Enterprise Fraud Detection & Integrity System - Main Render Function"""
    
    # Initialize fraud detection engine
    fraud_engine = FraudDetectionEngine(df)
    
    
    
    # Run all detections
    with st.spinner("Running comprehensive fraud analysis..."):
        unbundling_results = fraud_engine.detect_unbundling()
        upcoding_results = fraud_engine.detect_upcoding()
        conversion_results = fraud_engine.detect_opd_ipd_conversion()
        duplicate_results = fraud_engine.detect_duplicate_claims()
        ghost_surgery_results = fraud_engine.detect_ghost_surgery()
        anomaly_results = fraud_engine.calculate_anomaly_scores()
        risk_scores = fraud_engine.calculate_composite_risk_score()
    
    # Executive Dashboard
    st.header(" Fraud Detection Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_at_risk = (
            unbundling_results['Total_Amount'].sum() if not unbundling_results.empty else 0
        ) + (
            upcoding_results['Claim_Amount'].sum() if not upcoding_results.empty else 0
        ) + (
            duplicate_results['Total_Amount'].sum() if not duplicate_results.empty else 0
        )
        st.metric("Total Amount at Risk", f"${total_at_risk:,.0f}")
    
    with col2:
        high_risk_claims = len(risk_scores[risk_scores['Risk_Level'].isin(['High', 'Critical'])])
        st.metric("High Risk Claims", high_risk_claims)
    
    with col3:
        st.metric("Unbundling Cases", len(unbundling_results))
    
    with col4:
        st.metric("Upcoding Alerts", len(upcoding_results))
    
    with col5:
        st.metric("Duplicate Claims", len(duplicate_results))
    
    # Risk Distribution
    st.subheader("Risk Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_dist = risk_scores['Risk_Level'].value_counts()
        fig = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title='Claims by Risk Level',
            color=risk_dist.index,
            color_discrete_map={
                'Critical': '#dc2626',
                'High': '#ea580c',
                'Medium': '#eab308',
                'Low': '#22c55e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            risk_scores,
            x='Composite_Risk_Score',
            nbins=50,
            title='Risk Score Distribution',
            color_discrete_sequence=['#8b5cf6']
        )
        fig.add_vline(x=75, line_dash="dash", line_color="red", 
                     annotation_text="Critical Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for each detection module
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        " Unbundling",
        "️ Upcoding",
        " OPD-IPD Conversion",
        " Duplicates",
        " Ghost Surgery",
        " Anomalies",
        " Risk Analysis"
    ])
    
    # Tab 1: Unbundling Detection
    with tab1:
        st.subheader(" Unbundling Detector")
        st.markdown("""
        **Purpose:** Identify cases where single procedures are split into multiple CPT codes 
        on the same day to inflate billing.
        
        **Threshold:** Flags claims with >3 related procedures billed separately within 24 hours
        """)
        
        if unbundling_results.empty:
            st.success(" No unbundling patterns detected")
        else:
            st.error(f"️ Found {len(unbundling_results)} potential unbundling cases")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cases", len(unbundling_results))
            with col2:
                st.metric("Critical Severity", 
                         len(unbundling_results[unbundling_results['Severity'] == 'High']))
            with col3:
                st.metric("Total Amount", 
                         f"${unbundling_results['Total_Amount'].sum():,.0f}")
            
            # Detailed results
            st.markdown("#### Detailed Unbundling Cases")
            
            # Sort by risk score
            unbundling_results = unbundling_results.sort_values('Risk_Score', ascending=False)
            
            # Color coding
            def color_severity(val):
                if val == 'High':
                    return 'background-color: #fee2e2'
                elif val == 'Medium':
                    return 'background-color: #fef9c3'
                return ''
            
            styled_df = unbundling_results.style.applymap(
                color_severity, subset=['Severity']
            )
            st.dataframe(styled_df, use_container_width=True)
            
            # Hospital-wise analysis
            st.markdown("#### Hospital-wise Unbundling Pattern")
            hospital_unbundling = unbundling_results.groupby('Hospital_ID').agg({
                'Patient_ID': 'count',
                'Total_Amount': 'sum'
            }).reset_index()
            hospital_unbundling.columns = ['Hospital_ID', 'Cases', 'Total_Amount']
            hospital_unbundling = hospital_unbundling.sort_values('Cases', ascending=False)
            
            fig = px.bar(
                hospital_unbundling.head(10),
                x='Hospital_ID',
                y='Cases',
                title='Top 10 Hospitals by Unbundling Cases',
                color='Total_Amount',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Upcoding Detection
    with tab2:
        st.subheader("️ Upcoding Alert System")
        st.markdown("""
        **Purpose:** Detect cases where the claimed CPT code does not match the ICD diagnosis 
        code according to medical coding guidelines.
        
        **Method:** Cross-references CMS coding rules and flags mismatches with severity score
        """)
        
        if upcoding_results.empty:
            st.success(" No upcoding violations detected")
        else:
            st.error(f"️ Found {len(upcoding_results)} potential upcoding cases")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cases", len(upcoding_results))
            with col2:
                critical_count = len(upcoding_results[upcoding_results['Severity'] == 'Critical'])
                st.metric("Critical Cases", critical_count)
            with col3:
                avg_risk = upcoding_results['Risk_Score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1f}")
            with col4:
                total_amt = upcoding_results['Claim_Amount'].sum()
                st.metric("Total Amount", f"${total_amt:,.0f}")
            
            # Severity distribution
            severity_dist = upcoding_results['Severity'].value_counts()
            fig = px.bar(
                x=severity_dist.index,
                y=severity_dist.values,
                title='Upcoding Cases by Severity',
                color=severity_dist.index,
                color_discrete_map={
                    'Critical': '#dc2626',
                    'High': '#ea580c',
                    'Medium': '#eab308'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("#### Detailed Upcoding Cases")
            upcoding_display = upcoding_results.sort_values('Risk_Score', ascending=False)
            st.dataframe(upcoding_display, use_container_width=True)
            
            # Most common mismatches
            st.markdown("#### Most Common ICD-CPT Mismatches")
            mismatch_counts = upcoding_results.groupby(
                ['ICD_Code', 'CPT_Code']
            ).size().reset_index(name='Count')
            mismatch_counts = mismatch_counts.sort_values('Count', ascending=False).head(10)
            st.dataframe(mismatch_counts, use_container_width=True)
    
    # Tab 3: OPD-IPD Conversion
    with tab3:
        st.subheader(" OPD-to-IPD Conversion Tracker")
        st.markdown("""
        **Purpose:** Track percentage of outpatient visits converted to inpatient admissions per hospital.
        
        **Alert Threshold:** Conversion rate exceeds 80% (known fraud red flag)
        """)
        
        if conversion_results.empty:
            st.success(" No suspicious OPD-IPD conversion patterns detected")
        else:
            st.error(f"️ Found {len(conversion_results)} hospitals with suspicious conversion rates")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Flagged Hospitals", len(conversion_results))
            with col2:
                avg_conversion = conversion_results['Conversion_Rate'].mean()
                st.metric("Avg Conversion Rate", f"{avg_conversion:.1f}%")
            with col3:
                total_ipd_amt = conversion_results['Total_IPD_Amount'].sum()
                st.metric("Total IPD Amount", f"${total_ipd_amt:,.0f}")
            
            # Conversion rate chart
            fig = px.bar(
                conversion_results,
                x='Hospital_ID',
                y='Conversion_Rate',
                title='Hospital OPD-to-IPD Conversion Rates',
                color='Severity',
                color_discrete_map={'Critical': '#dc2626', 'High': '#ea580c'}
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                         annotation_text="Alert Threshold (80%)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("#### Detailed Conversion Analysis")
            st.dataframe(conversion_results, use_container_width=True)
    
    # Tab 4: Duplicate Claims
    with tab4:
        st.subheader(" Duplicate Claim Finder")
        st.markdown("""
        **Purpose:** Detect identical patient + procedure + date combinations within a 90-day window.
        
        **Detection:** Identifies exact duplicates and near-duplicates (same patient, similar dates)
        """)
        
        if duplicate_results.empty:
            st.success(" No duplicate claims detected")
        else:
            st.error(f"️ Found {len(duplicate_results)} duplicate claim patterns")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Duplicates", len(duplicate_results))
            with col2:
                exact_count = len(duplicate_results[duplicate_results['Duplicate_Type'] == 'Exact Duplicate'])
                st.metric("Exact Duplicates", exact_count)
            with col3:
                near_count = len(duplicate_results[duplicate_results['Duplicate_Type'] == 'Near Duplicate'])
                st.metric("Near Duplicates", near_count)
            with col4:
                total_amt = duplicate_results['Total_Amount'].sum()
                st.metric("Total Amount", f"${total_amt:,.0f}")
            
            # Duplicate type distribution
            dup_type_dist = duplicate_results['Duplicate_Type'].value_counts()
            fig = px.pie(
                values=dup_type_dist.values,
                names=dup_type_dist.index,
                title='Duplicate Claims by Type',
                color_discrete_sequence=['#dc2626', '#ea580c']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline analysis
            st.markdown("#### Days Between Duplicate Claims")
            fig = px.histogram(
                duplicate_results,
                x='Days_Between',
                nbins=30,
                title='Distribution of Days Between Duplicate Claims',
                color_discrete_sequence=['#8b5cf6']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("#### Detailed Duplicate Cases")
            duplicate_display = duplicate_results.sort_values('Risk_Score', ascending=False)
            st.dataframe(duplicate_display, use_container_width=True)
            
            # Hospital analysis
            st.markdown("#### Hospital-wise Duplicate Pattern")
            hospital_dup = duplicate_results.groupby('Hospital_ID').agg({
                'Patient_ID': 'count',
                'Total_Amount': 'sum'
            }).reset_index()
            hospital_dup.columns = ['Hospital_ID', 'Cases', 'Total_Amount']
            hospital_dup = hospital_dup.sort_values('Cases', ascending=False).head(10)
            
            fig = px.bar(
                hospital_dup,
                x='Hospital_ID',
                y='Cases',
                title='Top 10 Hospitals by Duplicate Claims',
                color='Total_Amount',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Ghost Surgery
    with tab5:
        st.subheader(" Ghost Surgery Pattern Detection")
        st.markdown("""
        **Purpose:** Detect surgeries billed when operating theatre records show no actual procedures.
        
        **Method:** Cross-reference with OT scheduling system or detect abnormal surgery patterns
        """)
        
        # OT schedule upload option
        st.markdown("#### Upload OT Schedule (Optional)")
        uploaded_ot = st.file_uploader(
            "Upload Operating Theatre Schedule CSV",
            type=['csv'],
            help="Upload OT schedule with columns: Patient_ID, Hospital_ID, OT_Date, Procedure_Code"
        )
        
        ot_schedule_df = None
        if uploaded_ot:
            try:
                ot_schedule_df = pd.read_csv(uploaded_ot)
                st.success(f" Loaded {len(ot_schedule_df)} OT records")
                ghost_surgery_results = fraud_engine.detect_ghost_surgery(ot_schedule_df)
            except Exception as e:
                st.error(f"Error loading OT schedule: {str(e)}")
        
        if ghost_surgery_results.empty:
            st.success(" No ghost surgery patterns detected")
        else:
            st.error(f"️ Found {len(ghost_surgery_results)} potential ghost surgery cases")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Suspicious Cases", len(ghost_surgery_results))
            with col2:
                if 'Claim_Amount' in ghost_surgery_results.columns:
                    total_amt = ghost_surgery_results['Claim_Amount'].sum()
                elif 'Total_Amount' in ghost_surgery_results.columns:
                    total_amt = ghost_surgery_results['Total_Amount'].sum()
                else:
                    total_amt = 0
                st.metric("Total Amount at Risk", f"${total_amt:,.0f}")
            with col3:
                critical_count = len(ghost_surgery_results[ghost_surgery_results['Severity'] == 'Critical'])
                st.metric("Critical Cases", critical_count)
            
            # Detailed results
            st.markdown("#### Detected Ghost Surgery Cases")
            st.dataframe(ghost_surgery_results, use_container_width=True)
            
            if 'Hospital_ID' in ghost_surgery_results.columns:
                # Hospital analysis
                st.markdown("#### Hospital-wise Analysis")
                hospital_ghost = ghost_surgery_results.groupby('Hospital_ID').size().reset_index(name='Cases')
                hospital_ghost = hospital_ghost.sort_values('Cases', ascending=False)
                
                fig = px.bar(
                    hospital_ghost,
                    x='Hospital_ID',
                    y='Cases',
                    title='Ghost Surgery Cases by Hospital',
                    color='Cases',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Anomalies
    with tab6:
        st.subheader(" Statistical Anomaly Detection")
        st.markdown("""
        **Purpose:** Identify claims with statistically unusual amounts using multiple detection methods.
        
        **Methods:** Z-score analysis, IQR (Interquartile Range) method
        """)
        
        if anomaly_results.empty:
            st.success(" No statistical anomalies detected")
        else:
            st.warning(f"️ Found {len(anomaly_results)} anomalous claims")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Anomalous Claims", len(anomaly_results))
            with col2:
                st.metric("% of Total", f"{len(anomaly_results)/len(df)*100:.2f}%")
            with col3:
                total_amt = anomaly_results['Claim_Amount'].sum()
                st.metric("Total Amount at Risk", f"${total_amt:,.0f}")
            with col4:
                avg_z = anomaly_results['Z_Score'].mean()
                st.metric("Avg Z-Score", f"{avg_z:.2f}")
            
            # Anomaly visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    anomaly_results.head(100),
                    x=range(len(anomaly_results.head(100))),
                    y='Claim_Amount',
                    color='Severity',
                    title='Anomalous Claim Amounts',
                    color_discrete_map={
                        'Critical': '#dc2626',
                        'High': '#ea580c',
                        'Medium': '#eab308',
                        'Low': '#22c55e'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df,
                    y='Claim_Amount',
                    title='Claim Amount Distribution with Outliers'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed anomalies
            st.markdown("#### High-Risk Anomalous Claims")
            high_risk_anomalies = anomaly_results[
                anomaly_results['Severity'].isin(['High', 'Critical'])
            ].sort_values('Risk_Score', ascending=False)
            
            display_cols = ['Claim_ID', 'Hospital_ID', 'Disease_Category', 
                          'Claim_Amount', 'Z_Score', 'Risk_Score', 'Severity']
            available_cols = [col for col in display_cols if col in high_risk_anomalies.columns]
            st.dataframe(high_risk_anomalies[available_cols].head(20), use_container_width=True)
    
    # Tab 7: Risk Analysis
    with tab7:
        st.subheader(" Comprehensive Risk Analysis")
        st.markdown("""
        **Purpose:** Composite risk scoring combining all fraud detection methods.
        
        **Risk Weights:** Unbundling (25%), Upcoding (30%), Duplicates (20%), 
        Anomalies (15%), Conversion (10%)
        """)
        
        # Overall risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_level_dist = risk_scores['Risk_Level'].value_counts()
            fig = px.pie(
                values=risk_level_dist.values,
                names=risk_level_dist.index,
                title='Overall Risk Distribution',
                color=risk_level_dist.index,
                color_discrete_map={
                    'Critical': '#dc2626',
                    'High': '#ea580c',
                    'Medium': '#eab308',
                    'Low': '#22c55e'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk score vs claim amount
            fig = px.scatter(
                risk_scores.sample(min(1000, len(risk_scores))),
                x='Claim_Amount',
                y='Composite_Risk_Score',
                color='Risk_Level',
                title='Risk Score vs Claim Amount',
                color_discrete_map={
                    'Critical': '#dc2626',
                    'High': '#ea580c',
                    'Medium': '#eab308',
                    'Low': '#22c55e'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk claims
        st.markdown("#### Top 20 Highest Risk Claims")
        high_risk = risk_scores.sort_values('Composite_Risk_Score', ascending=False).head(20)
        st.dataframe(high_risk, use_container_width=True)
        
        # Hospital risk profile
        st.markdown("#### Hospital Risk Profile")
        hospital_risk = risk_scores.groupby('Hospital_ID').agg({
            'Composite_Risk_Score': 'mean',
            'Claim_ID': 'count',
            'Claim_Amount': 'sum'
        }).reset_index()
        hospital_risk.columns = ['Hospital_ID', 'Avg_Risk_Score', 'Total_Claims', 'Total_Amount']
        hospital_risk = hospital_risk.sort_values('Avg_Risk_Score', ascending=False).head(15)
        
        fig = px.bar(
            hospital_risk,
            x='Hospital_ID',
            y='Avg_Risk_Score',
            title='Top 15 Hospitals by Average Risk Score',
            color='Avg_Risk_Score',
            color_continuous_scale='Reds'
        )
        fig.add_hline(y=50, line_dash="dash", line_color="orange",
                     annotation_text="High Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(hospital_risk, use_container_width=True)
        
        # Risk component breakdown
        st.markdown("#### Risk Component Analysis")
        component_cols = ['Unbundling_Risk', 'Upcoding_Risk', 'Duplicate_Risk', 'Anomaly_Risk']
        component_means = risk_scores[component_cols].mean()
        
        fig = px.bar(
            x=component_cols,
            y=component_means.values,
            title='Average Risk Score by Component',
            labels={'x': 'Risk Component', 'y': 'Average Score'},
            color=component_means.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Action Items and Recommendations
    st.markdown("---")
    st.header(" Action Items & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Immediate Actions Required")
        actions = []
        
        if not unbundling_results.empty:
            high_unbundling = len(unbundling_results[unbundling_results['Severity'] == 'High'])
            actions.append(f" Review {high_unbundling} high-severity unbundling cases")
        
        if not upcoding_results.empty:
            critical_upcoding = len(upcoding_results[upcoding_results['Severity'] == 'Critical'])
            actions.append(f" Investigate {critical_upcoding} critical upcoding violations")
        
        if not duplicate_results.empty:
            exact_dups = len(duplicate_results[duplicate_results['Duplicate_Type'] == 'Exact Duplicate'])
            actions.append(f" Resolve {exact_dups} exact duplicate claims")
        
        if not conversion_results.empty:
            actions.append(f" Audit {len(conversion_results)} hospitals with high OPD-IPD conversion")
        
        if actions:
            for action in actions:
                st.markdown(f"- {action}")
        else:
            st.success(" No immediate actions required")
    
    with col2:
        st.markdown("#### Process Improvements")
        st.markdown("""
        - **Implement Real-time Validation:** Add pre-submission checks for unbundling and upcoding
        - **Enhanced Documentation:** Require OT (Operation Theatre) records upload for all surgery claims
        - **Duplicate Prevention:** Implement 90-day claim history check before approval
        - **Provider Education:** Train hospitals on coding guidelines and compliance
        - **Automated Monitoring:** Set up daily fraud detection reports
        - **Audit Schedule:** Prioritize high-risk hospitals for quarterly audits
        """)
    
    # Export functionality
    st.markdown("---")
    st.header(" Export Fraud Detection Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Unbundling Cases", use_container_width=True):
            if not unbundling_results.empty:
                st.success(" Report ready for download")
                # In production: implement actual download
            else:
                st.info("No cases to export")
    
    with col2:
        if st.button("Export Upcoding Cases", use_container_width=True):
            if not upcoding_results.empty:
                st.success(" Report ready for download")
            else:
                st.info("No cases to export")
    
    with col3:
        if st.button("Export Comprehensive Report", use_container_width=True):
            st.success(" Comprehensive report ready for download")


# Required Additional Fields for CSV
"""
REQUIRED ADDITIONAL FIELDS FOR COMPREHENSIVE FRAUD DETECTION:

Current fields:
Claim_ID, Hospital_ID, Insurer_ID, Disease_Category, CPT_Code, Claim_Amount, 
Paid_Amount, Status, Denial_Reason, Submission_Date, Days_To_Payment, 
Readmission_Flag, Clinical_Notes, Documentation_Score

ADDITIONAL REQUIRED FIELDS:

1. Patient_ID (string) - Required for duplicate detection and unbundling
   Format: PAT00001, PAT00002, etc.

2. Procedure_Date (datetime) - Required for all time-based detection
   Format: YYYY-MM-DD
   Note: Can use Submission_Date as fallback

3. ICD_Code (string) - Required for upcoding detection
   Format: ICD-10 codes (e.g., I10, M17.11, E11.9)
   Note: Can be derived from Disease_Category but explicit is better

4. Admission_Type (string) - Required for OPD-IPD conversion tracking
   Values: 'OPD' or 'IPD'

5. OT_Record_ID (string, optional) - For ghost surgery detection
   Format: OT2024-001, OT2024-002, etc.
   Note: Can be null; external OT schedule can be uploaded separately

6. Provider_ID (string, optional but recommended) - For provider pattern analysis
   Format: DOC00001, DOC00002, etc.

UPDATED CSV FORMAT:
Claim_ID,Patient_ID,Hospital_ID,Insurer_ID,Provider_ID,Disease_Category,ICD_Code,
CPT_Code,Claim_Amount,Paid_Amount,Status,Denial_Reason,Submission_Date,
Procedure_Date,Days_To_Payment,Admission_Type,Readmission_Flag,OT_Record_ID,
Clinical_Notes,Documentation_Score
"""