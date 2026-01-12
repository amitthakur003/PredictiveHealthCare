import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Set
import warnings

warnings.filterwarnings('ignore')

class DenialRiskAnalysisEngine:
    """Enterprise-level Denial Risk Analysis & Prediction System"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Initialize dictionaries first
        self._init_rules()
        self._prepare_data()
        
    def _init_rules(self):
        """Initialize all rule dictionaries"""
        # Denial reason categories
        self.denial_categories = {
            'Coding Errors': ['Invalid CPT', 'Invalid ICD', 'Unbundling', 'Upcoding', 
                             'Modifier Error', 'Code Mismatch'],
            'Authorization Issues': ['Prior Auth Missing', 'Auth Expired', 'Not Authorized',
                                    'Auth Limit Exceeded', 'Out of Network'],
            'Missing Documents': ['Missing Clinical Notes', 'Missing Reports', 
                                 'Incomplete Documentation', 'Missing Signature'],
            'Medical Necessity': ['Not Medically Necessary', 'Experimental', 
                                 'Alternative Treatment Available'],
            'Eligibility Issues': ['Patient Not Covered', 'Coverage Expired', 
                                  'Wrong Insurance', 'Coordination of Benefits'],
            'Billing Errors': ['Duplicate Claim', 'Timely Filing', 'Incorrect Amount',
                              'Wrong Billing Code']
        }
        
        # Prior authorization required procedures (sample)
        self.prior_auth_required = {
            'High Risk': ['27447', '27440', '43644', '47562', '49320'],  # Major surgeries
            'Medium Risk': ['29881', '45380', '52601', '58150'],  # Moderate procedures
            'Low Risk': ['99213', '99214', '99215']  # Complex office visits
        }
        
        # Payer-specific rules (sample)
        self.payer_rules = {
            'INS001': {
                'name': 'Blue Cross',
                'max_claim_amount': 50000,
                'prior_auth_threshold': 10000,
                'documentation_required': ['Clinical Notes', 'Lab Reports'],
                'timely_filing_days': 90,
                'excluded_cpts': ['99499'],
                'bundling_rules': True
            },
            'INS002': {
                'name': 'United Health',
                'max_claim_amount': 75000,
                'prior_auth_threshold': 15000,
                'documentation_required': ['Clinical Notes', 'Imaging Reports'],
                'timely_filing_days': 120,
                'excluded_cpts': [],
                'bundling_rules': True
            },
            'INS003': {
                'name': 'Aetna',
                'max_claim_amount': 60000,
                'prior_auth_threshold': 12000,
                'documentation_required': ['Clinical Notes'],
                'timely_filing_days': 60,
                'excluded_cpts': ['99091'],
                'bundling_rules': False
            }
        }
        
        # ICD-CPT compliance mapping
        self.coding_compliance_rules = {
            'I10': ['99213', '99214', '99215', '93000'],  # Hypertension
            'M17': ['27440', '27441', '27442', '27447', '29881'],  # Knee disorders
            'E11': ['99213', '99214', '82947', '82950', '83036'],  # Diabetes
            'J45': ['94010', '94060', '94640', '99213'],  # Asthma
            'G20': ['99213', '99214', '99215', '95886']  # Parkinson's
        }
    
    def _prepare_data(self):
        """Prepare data for analysis using the loaded CSV data"""
        
        # List of date columns to parse from the global file
        date_cols = ['Submission_Date', 'Procedure_Date', 'Dispute_Open_Date', 
                     'Dispute_Close_Date', 'Admission_Date', 'Discharge_Date']
        
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Categorize denial reasons
        if 'Denial_Reason' in self.df.columns:
            self.df['Denial_Category'] = self.df['Denial_Reason'].apply(self._categorize_denial)
        
        # Ensure numerical types for calculations
        if 'Claim_Amount' in self.df.columns:
            self.df['Claim_Amount'] = pd.to_numeric(self.df['Claim_Amount'], errors='coerce').fillna(0)
            
        # Ensure Prior_Auth_Status exists (using CSV data, filling only if completely missing)
        if 'Prior_Auth_Status' not in self.df.columns:
             self.df['Prior_Auth_Status'] = 'Not Required'
        else:
             self.df['Prior_Auth_Status'] = self.df['Prior_Auth_Status'].fillna('Not Required')

        # Add resubmission flag default if missing (CSV usually has it)
        if 'Resubmission_Count' not in self.df.columns:
            self.df['Resubmission_Count'] = 0
    
    def _categorize_denial(self, reason: str) -> str:
        """Categorize denial reason into main categories"""
        if pd.isna(reason):
            return 'Unknown'
        
        reason_lower = str(reason).lower()
        for category, keywords in self.denial_categories.items():
            for keyword in keywords:
                if keyword.lower() in reason_lower:
                    return category
        return 'Other'
    
    def analyze_denial_trends(self) -> Dict:
        """5.1 Denial Trend Identification"""
        if 'Status' not in self.df.columns:
             return {'total_denials': 0, 'denial_rate': 0}

        denied_claims = self.df[self.df['Status'] == 'Denied'].copy()
        
        if denied_claims.empty:
            return {
                'total_denials': 0,
                'denial_rate': 0,
                'by_category': pd.DataFrame(),
                'by_payer': pd.DataFrame(),
                'by_provider': pd.DataFrame(),
                'by_procedure': pd.DataFrame(),
                'trends_over_time': pd.DataFrame()
            }
        
        # Denial by category
        by_category = denied_claims['Denial_Category'].value_counts().reset_index()
        by_category.columns = ['Category', 'Count']
        by_category['Percentage'] = (by_category['Count'] / len(denied_claims) * 100).round(2)
        by_category['Total_Amount'] = denied_claims.groupby('Denial_Category')['Claim_Amount'].sum().values
        
        # Denial by payer
        by_payer = denied_claims.groupby('Insurer_ID').agg({
            'Claim_ID': 'count',
            'Claim_Amount': 'sum',
            'Denial_Reason': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        by_payer.columns = ['Insurer_ID', 'Denial_Count', 'Total_Amount', 'Top_Reason']
        
        # Calculate denial rate safely
        total_claims_per_insurer = self.df.groupby('Insurer_ID').size()
        by_payer['Denial_Rate'] = by_payer.apply(
            lambda x: (x['Denial_Count'] / total_claims_per_insurer.get(x['Insurer_ID'], 1) * 100), axis=1
        ).round(2)
        
        # Denial by provider
        by_provider = denied_claims.groupby('Hospital_ID').agg({
            'Claim_ID': 'count',
            'Claim_Amount': 'sum',
            'Denial_Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        by_provider.columns = ['Hospital_ID', 'Denial_Count', 'Total_Amount', 'Top_Category']
        
        # Denial by procedure
        by_procedure = denied_claims.groupby('CPT_Code').agg({
            'Claim_ID': 'count',
            'Claim_Amount': 'sum',
            'Denial_Reason': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        by_procedure.columns = ['CPT_Code', 'Denial_Count', 'Total_Amount', 'Top_Reason']
        by_procedure = by_procedure.sort_values('Denial_Count', ascending=False)
        
        # Trends over time
        if 'Submission_Date' in denied_claims.columns:
            denied_claims['Month'] = denied_claims['Submission_Date'].dt.to_period('M').astype(str)
            trends = denied_claims.groupby(['Month', 'Denial_Category']).size().reset_index(name='Count')
        else:
            trends = pd.DataFrame()
        
        return {
            'total_denials': len(denied_claims),
            'denial_rate': len(denied_claims) / len(self.df) * 100,
            'by_category': by_category,
            'by_payer': by_payer,
            'by_provider': by_provider,
            'by_procedure': by_procedure,
            'trends_over_time': trends
        }
    
    def calculate_denial_probability(self, claim_data: Dict = None) -> pd.DataFrame:
        """5.2 Claim Rejection Probability Scoring"""
        # If specific claim provided, score it
        if claim_data:
            return self._score_single_claim(claim_data)
        
        # Score all claims
        denial_scores = []
        
        # Calculate historical denial rates
        historical_denial = self.df[self.df['Status'] == 'Denied'] if 'Status' in self.df.columns else pd.DataFrame()
        
        if historical_denial.empty:
            payer_denial_rate = {}
            procedure_denial_rate = {}
            provider_denial_rate = {}
        else:
            payer_denial_rate = (historical_denial.groupby('Insurer_ID').size() / self.df.groupby('Insurer_ID').size()).to_dict()
            procedure_denial_rate = (historical_denial.groupby('CPT_Code').size() / self.df.groupby('CPT_Code').size()).to_dict()
            provider_denial_rate = (historical_denial.groupby('Hospital_ID').size() / self.df.groupby('Hospital_ID').size()).to_dict()
        
        for idx, claim in self.df.iterrows():
            score_components = {
                'Claim_ID': claim['Claim_ID'],
                'Hospital_ID': claim['Hospital_ID'],
                'Insurer_ID': claim['Insurer_ID'],
                'CPT_Code': claim.get('CPT_Code'),
                'Claim_Amount': claim['Claim_Amount']
            }
            
            # Component scores
            payer_risk = payer_denial_rate.get(claim['Insurer_ID'], 0.2) * 100
            procedure_risk = procedure_denial_rate.get(claim.get('CPT_Code'), 0.2) * 100
            provider_risk = provider_denial_rate.get(claim['Hospital_ID'], 0.2) * 100
            
            # Amount-based risk
            amount_risk = min(100, (claim['Claim_Amount'] / 50000) * 50)
            
            # Documentation risk
            doc_score = claim.get('Documentation_Score', 50)
            doc_risk = max(0, 100 - doc_score)
            
            # Prior auth risk
            auth_status = claim.get('Prior_Auth_Status', 'Not Required')
            auth_risk = {
                'Missing': 90,
                'Expired': 80,
                'Pending': 50,
                'Approved': 10,
                'Not Required': 0
            }.get(auth_status, 30)
            
            # Composite probability
            denial_probability = (
                payer_risk * 0.25 +
                procedure_risk * 0.20 +
                provider_risk * 0.15 +
                amount_risk * 0.15 +
                doc_risk * 0.15 +
                auth_risk * 0.10
            )
            
            score_components.update({
                'Payer_Risk': round(payer_risk, 2),
                'Procedure_Risk': round(procedure_risk, 2),
                'Provider_Risk': round(provider_risk, 2),
                'Amount_Risk': round(amount_risk, 2),
                'Documentation_Risk': round(doc_risk, 2),
                'Authorization_Risk': round(auth_risk, 2),
                'Denial_Probability': round(denial_probability, 2),
                'Risk_Level': 'Critical' if denial_probability > 75 else
                             'High' if denial_probability > 50 else
                             'Medium' if denial_probability > 25 else 'Low'
            })
            
            denial_scores.append(score_components)
        
        return pd.DataFrame(denial_scores)
    
    def _score_single_claim(self, claim_data: Dict) -> pd.DataFrame:
        """Score a single claim dynamically"""
        # Simplified implementation for single claim - normally would use trained model
        return pd.DataFrame([claim_data])
    
    def check_prior_authorization(self) -> pd.DataFrame:
        """5.3 Prior Authorization Check"""
        auth_issues = []
        
        for idx, claim in self.df.iterrows():
            cpt_code = str(claim.get('CPT_Code', ''))
            claim_amount = claim.get('Claim_Amount', 0)
            auth_status = claim.get('Prior_Auth_Status', 'Missing')
            
            # Check if prior auth required
            risk_level = None
            for level, codes in self.prior_auth_required.items():
                if cpt_code in codes:
                    risk_level = level
                    break
            
            # Check payer-specific thresholds
            payer_id = claim.get('Insurer_ID')
            payer_threshold = 10000  # Default
            if payer_id in self.payer_rules:
                payer_threshold = self.payer_rules[payer_id]['prior_auth_threshold']
            
            # Flag issues
            requires_auth = risk_level is not None or claim_amount > payer_threshold
            
            if requires_auth:
                issue_severity = None
                issue_type = None
                
                if auth_status == 'Missing':
                    issue_severity = 'Critical'
                    issue_type = 'Prior Authorization Missing'
                elif auth_status == 'Expired':
                    issue_severity = 'High'
                    issue_type = 'Prior Authorization Expired'
                elif auth_status == 'Pending':
                    issue_severity = 'Medium'
                    issue_type = 'Prior Authorization Pending'
                
                if issue_severity:
                    auth_issues.append({
                        'Claim_ID': claim['Claim_ID'],
                        'Hospital_ID': claim['Hospital_ID'],
                        'Insurer_ID': payer_id,
                        'CPT_Code': cpt_code,
                        'Procedure_Risk_Level': risk_level or 'Amount-Based',
                        'Claim_Amount': claim_amount,
                        'Auth_Status': auth_status,
                        'Issue_Type': issue_type,
                        'Severity': issue_severity,
                        'Required_Action': 'Obtain prior authorization before submission',
                        'Payer_Threshold': payer_threshold
                    })
        
        return pd.DataFrame(auth_issues) if auth_issues else pd.DataFrame()
    
    def verify_coding_compliance(self) -> pd.DataFrame:
        """5.4 Coding Compliance Verification"""
        compliance_issues = []
        
        for idx, claim in self.df.iterrows():
            icd_code = claim.get('ICD_Code')
            cpt_code = str(claim.get('CPT_Code', ''))
            
            if pd.notna(icd_code) and cpt_code:
                # Check ICD-CPT compliance
                valid_cpts = self.coding_compliance_rules.get(icd_code, [])
                
                if valid_cpts and cpt_code not in valid_cpts:
                    # Calculate severity
                    claim_amount = claim.get('Claim_Amount', 0)
                    severity_score = min(100, (claim_amount / 1000) * 5)
                    
                    severity = 'Critical' if severity_score > 75 else \
                              'High' if severity_score > 50 else \
                              'Medium' if severity_score > 25 else 'Low'
                    
                    compliance_issues.append({
                        'Claim_ID': claim['Claim_ID'],
                        'Hospital_ID': claim['Hospital_ID'],
                        'ICD_Code': icd_code,
                        'Disease_Category': claim.get('Disease_Category'),
                        'CPT_Code': cpt_code,
                        'Valid_CPT_Codes': ', '.join(valid_cpts),
                        'Compliance_Gap': 'CPT-ICD Mismatch',
                        'Claim_Amount': claim_amount,
                        'Severity': severity,
                        'Severity_Score': round(severity_score, 2),
                        'Recommendation': f'Use one of: {", ".join(valid_cpts[:3])}',
                        'Impact': 'High denial probability due to coding mismatch'
                    })
        
        return pd.DataFrame(compliance_issues) if compliance_issues else pd.DataFrame()
    
    def validate_payer_rules(self) -> pd.DataFrame:
        """5.5 Payer-Specific Rule Engine"""
        payer_violations = []
        
        for idx, claim in self.df.iterrows():
            payer_id = claim.get('Insurer_ID')
            
            if payer_id in self.payer_rules:
                rules = self.payer_rules[payer_id]
                violations = []
                
                # Check max claim amount
                if claim['Claim_Amount'] > rules['max_claim_amount']:
                    violations.append({
                        'rule': 'Maximum Claim Amount',
                        'violation': f"Amount ${claim['Claim_Amount']:,.0f} exceeds limit ${rules['max_claim_amount']:,.0f}",
                        'severity': 'High'
                    })
                
                # Check excluded CPTs
                cpt_code = str(claim.get('CPT_Code', ''))
                if cpt_code in rules['excluded_cpts']:
                    violations.append({
                        'rule': 'Excluded Procedure',
                        'violation': f"CPT {cpt_code} not covered by {rules['name']}",
                        'severity': 'Critical'
                    })
                
                # Check timely filing
                if 'Submission_Date' in claim and pd.notna(claim['Submission_Date']):
                    procedure_date = claim.get('Procedure_Date', claim['Submission_Date'])
                    if pd.notna(procedure_date):
                        days_elapsed = (claim['Submission_Date'] - procedure_date).days
                        if days_elapsed > rules['timely_filing_days']:
                            violations.append({
                                'rule': 'Timely Filing',
                                'violation': f"Submitted {days_elapsed} days after procedure (limit: {rules['timely_filing_days']})",
                                'severity': 'Critical'
                            })
                
                # Check prior auth
                if claim['Claim_Amount'] > rules['prior_auth_threshold']:
                    auth_status = claim.get('Prior_Auth_Status', 'Missing')
                    if auth_status not in ['Approved', 'Not Required']:
                        violations.append({
                            'rule': 'Prior Authorization Required',
                            'violation': f"Auth {auth_status} for amount exceeding ${rules['prior_auth_threshold']:,.0f}",
                            'severity': 'High'
                        })
                
                # Add all violations for this claim
                for violation in violations:
                    payer_violations.append({
                        'Claim_ID': claim['Claim_ID'],
                        'Payer_ID': payer_id,
                        'Payer_Name': rules['name'],
                        'Hospital_ID': claim['Hospital_ID'],
                        'Rule_Category': violation['rule'],
                        'Violation_Details': violation['violation'],
                        'Severity': violation['severity'],
                        'Claim_Amount': claim['Claim_Amount'],
                        'Recommendation': self._get_correction_recommendation(violation['rule'])
                    })
        
        return pd.DataFrame(payer_violations) if payer_violations else pd.DataFrame()
    
    def _get_correction_recommendation(self, rule_category: str) -> str:
        """Get correction recommendations"""
        recommendations = {
            'Maximum Claim Amount': 'Split claim into multiple submissions or review pricing',
            'Excluded Procedure': 'Verify coverage or use alternative CPT code',
            'Timely Filing': 'File appeal with explanation for delay',
            'Prior Authorization Required': 'Obtain retroactive authorization or resubmit with auth'
        }
        return recommendations.get(rule_category, 'Review and correct claim details')
    
    def assess_resubmission_readiness(self) -> pd.DataFrame:
        """5.6 Resubmission Readiness Check"""
        denied_claims = self.df[self.df['Status'] == 'Denied'].copy()
        
        if denied_claims.empty:
            return pd.DataFrame()
        
        resubmission_assessment = []
        
        for idx, claim in denied_claims.iterrows():
            readiness_checks = {
                'Claim_ID': claim['Claim_ID'],
                'Hospital_ID': claim['Hospital_ID'],
                'Insurer_ID': claim['Insurer_ID'],
                'Original_Denial_Reason': claim.get('Denial_Reason'),
                'Denial_Category': claim.get('Denial_Category'),
                'Claim_Amount': claim['Claim_Amount']
            }
            
            # Check documentation completeness
            doc_score = claim.get('Documentation_Score', 0)
            doc_complete = doc_score >= 80
            readiness_checks['Documentation_Complete'] = 'Yes' if doc_complete else 'No'
            readiness_checks['Documentation_Score'] = doc_score
            
            # Check coding corrections
            icd_code = claim.get('ICD_Code')
            cpt_code = str(claim.get('CPT_Code', ''))
            coding_valid = False
            if pd.notna(icd_code) and cpt_code:
                valid_cpts = self.coding_compliance_rules.get(icd_code, [])
                coding_valid = not valid_cpts or cpt_code in valid_cpts
            readiness_checks['Coding_Corrected'] = 'Yes' if coding_valid else 'No'
            
            # Check prior auth
            auth_status = claim.get('Prior_Auth_Status', 'Missing')
            auth_obtained = auth_status == 'Approved'
            readiness_checks['Authorization_Obtained'] = 'Yes' if auth_obtained else 'No'
            readiness_checks['Auth_Status'] = auth_status
            
            # Check if previously resubmitted
            resubmission_count = claim.get('Resubmission_Count', 0)
            readiness_checks['Resubmission_Count'] = resubmission_count
            readiness_checks['Max_Resubmissions_Reached'] = 'Yes' if resubmission_count >= 2 else 'No'
            
            # Calculate readiness score
            readiness_score = 0
            if doc_complete: readiness_score += 40
            if coding_valid: readiness_score += 30
            if auth_obtained: readiness_score += 30
            
            readiness_checks['Readiness_Score'] = readiness_score
            readiness_checks['Resubmission_Status'] = (
                'Ready' if readiness_score >= 90 else
                'Needs Minor Corrections' if readiness_score >= 70 else
                'Needs Major Corrections' if readiness_score >= 50 else
                'Not Ready'
            )
            
            # Missing items
            missing_items = []
            if not doc_complete: missing_items.append('Complete documentation')
            if not coding_valid: missing_items.append('Correct CPT-ICD pairing')
            if not auth_obtained and claim['Claim_Amount'] > 10000: missing_items.append('Prior authorization')
            
            readiness_checks['Missing_Items'] = '; '.join(missing_items) if missing_items else 'None'
            readiness_checks['Action_Required'] = (
                'Approved for resubmission' if readiness_score >= 90 else
                f"Complete: {'; '.join(missing_items)}"
            )
            
            resubmission_assessment.append(readiness_checks)
        
        return pd.DataFrame(resubmission_assessment)

    def generate_strategic_recommendations(self, auth_issues: pd.DataFrame, coding_issues: pd.DataFrame, 
                                         payer_violations: pd.DataFrame, denial_trends: Dict) -> pd.DataFrame:
        """5.7 AI Strategic Recommendation Engine"""
        recommendations = []
        
        # 1. Authorization Recommendations
        if not auth_issues.empty:
            # Frequency analysis
            top_missing_auth_proc = auth_issues['CPT_Code'].mode()
            if not top_missing_auth_proc.empty:
                code = top_missing_auth_proc[0]
                affected_count = len(auth_issues[auth_issues['CPT_Code'] == code])
                money_at_risk = auth_issues[auth_issues['CPT_Code'] == code]['Claim_Amount'].sum()
                
                recommendations.append({
                    "Priority": "Critical",
                    "Category": "Authorization",
                    "Issue Detected": f"Frequent auth failures for CPT {code}",
                    "Recommended Action": "Implement mandatory pre-submission auth check",
                    "Estimated Impact": f"Recover ${money_at_risk:,.0f} / {affected_count} claims"
                })
            
            # Global auth Check
            if len(auth_issues) > 5:
                recommendations.append({
                    "Priority": "High",
                    "Category": "Workflow",
                    "Issue Detected": f"{len(auth_issues)} Total Authorization Failures",
                    "Recommended Action": "Integrate real-time Payer Portal Verification API",
                    "Estimated Impact": "Reduce manual auth errors by ~40%"
                })

        # 2. Coding Compliance Recommendations
        if not coding_issues.empty:
            top_mismatch = coding_issues['ICD_Code'].mode()
            if not top_mismatch.empty:
                icd = top_mismatch[0]
                recommendations.append({
                    "Priority": "Medium",
                    "Category": "Training",
                    "Issue Detected": f"Recurring mismatch for ICD-10 {icd}",
                    "Recommended Action": "Conduct refresher training for Coding Team",
                    "Estimated Impact": "Improve first-pass acceptance rate"
                })

        # 3. Payer Rule Recommendations
        if not payer_violations.empty:
            # Check for timely filing issues
            timely_filing_errs = payer_violations[payer_violations['Rule_Category'] == 'Timely Filing']
            if not timely_filing_errs.empty:
                recommendations.append({
                    "Priority": "Critical",
                    "Category": "Operations",
                    "Issue Detected": "Timely Filing Deadline Missed",
                    "Recommended Action": "Audit submission workflow & set alerts T-10 days before deadline",
                    "Estimated Impact": "Prevent 100% loss on expired claims"
                })
            
            # Check for specific payer friction
            top_payer = payer_violations['Payer_Name'].mode()
            if not top_payer.empty:
                payer = top_payer[0]
                recommendations.append({
                    "Priority": "High",
                    "Category": "Contracting",
                    "Issue Detected": f"High violation rate with {payer}",
                    "Recommended Action": "Review payer contract terms or update Rules Engine",
                    "Estimated Impact": "Align submissions with payer policies"
                })

        # 4. Global Denial Rate
        denial_rate = denial_trends.get('denial_rate', 0)
        if denial_rate > 15:
             recommendations.append({
                 "Priority": "Critical",
                 "Category": "Strategic",
                 "Issue Detected": f"High Denial Rate ({denial_rate:.1f}%)",
                 "Recommended Action": "Initiate comprehensive Revenue Cycle Audit",
                 "Estimated Impact": "Reduce denials to industry standard (<5%)"
             })

        if not recommendations:
            return pd.DataFrame([{
                "Priority": "Low", 
                "Category": "General", 
                "Issue Detected": "No critical issues found", 
                "Recommended Action": "Maintain current high-performance workflows", 
                "Estimated Impact": "N/A"
            }])
            
        return pd.DataFrame(recommendations)

def render_denial_analysis(df: pd.DataFrame, models=None):
    """Enterprise Denial Risk Analysis System - Main Render Function"""
    
    # Initialize engine
    denial_engine = DenialRiskAnalysisEngine(df)
    
    # Run all analyses
    with st.spinner("Analyzing denial patterns and risk factors..."):
        denial_trends = denial_engine.analyze_denial_trends()
        denial_probabilities = denial_engine.calculate_denial_probability()
        auth_issues = denial_engine.check_prior_authorization()
        coding_issues = denial_engine.verify_coding_compliance()
        payer_violations = denial_engine.validate_payer_rules()
        resubmission_readiness = denial_engine.assess_resubmission_readiness()
        # NEW: Generate Recommendations
        recommendations_df = denial_engine.generate_strategic_recommendations(
            auth_issues, coding_issues, payer_violations, denial_trends
        )
    
    # Executive Dashboard
    st.header(" Denial Risk Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Denials",
            denial_trends['total_denials'],
            delta=f"{denial_trends['denial_rate']:.1f}% rate"
        )
    
    with col2:
        high_risk_claims = len(denial_probabilities[
            denial_probabilities['Risk_Level'].isin(['High', 'Critical'])
        ])
        st.metric("High Risk Claims", high_risk_claims)
    
    with col3:
        st.metric("Auth Issues", len(auth_issues))
    
    with col4:
        st.metric("Coding Issues", len(coding_issues))
    
    with col5:
        ready_for_resubmission = len(resubmission_readiness[
            resubmission_readiness['Resubmission_Status'] == 'Ready'
        ]) if not resubmission_readiness.empty else 0
        st.metric("Ready for Resubmission", ready_for_resubmission)
    
    # Tabs for different modules
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        " Denial Trends",
        " Probability Scoring",
        " Prior Authorization",
        " Coding Compliance",
        " Payer Rules",
        " Resubmission",
        " Recommendations"
    ])
    
    # Tab 1: Denial Trend Identification
    with tab1:
        st.subheader(" Denial Trend Identification")
        st.markdown("""
        **Purpose:** Identify recurring reasons for claim denials across payers, procedures, and providers.
        """)
        
        if denial_trends['total_denials'] == 0:
            st.success(" No denials in the current dataset")
        else:
            # Denial by category
            st.markdown("#### Denials by Category")
            col1, col2 = st.columns(2)
            
            by_category = denial_trends['by_category']
            
            with col1:
                fig = px.pie(
                    by_category,
                    values='Count',
                    names='Category',
                    title='Denial Distribution by Category',
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    by_category,
                    x='Count',
                    y='Category',
                    orientation='h',
                    title='Denial Count by Category',
                    color='Total_Amount',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(by_category, use_container_width=True)
            
            # Denial by payer
            st.markdown("#### Denials by Payer")
            by_payer = denial_trends['by_payer'].sort_values('Denial_Rate', ascending=False)
            
            fig = px.bar(
                by_payer.head(10),
                x='Insurer_ID',
                y='Denial_Rate',
                title='Top 10 Payers by Denial Rate',
                color='Denial_Count',
                color_continuous_scale='Reds',
                text='Denial_Rate'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(by_payer, use_container_width=True)
            
            # Denial by provider
            st.markdown("#### Denials by Provider")
            by_provider = denial_trends['by_provider'].sort_values('Denial_Count', ascending=False).head(15)
            
            fig = px.bar(
                by_provider,
                x='Hospital_ID',
                y='Denial_Count',
                title='Top 15 Providers by Denial Count',
                color='Total_Amount',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Denial by procedure
            st.markdown("#### Denials by Procedure")
            by_procedure = denial_trends['by_procedure'].head(10)
            st.dataframe(by_procedure, use_container_width=True)
            
            # Trends over time
            if not denial_trends['trends_over_time'].empty:
                st.markdown("#### Denial Trends Over Time")
                trends = denial_trends['trends_over_time']
                fig = px.line(
                    trends,
                    x='Month',
                    y='Count',
                    color='Denial_Category',
                    title='Denial Trends by Category Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Claim Rejection Probability Scoring
    with tab2:
        st.subheader(" Claim Rejection Probability Scoring")
        st.markdown("""
        **Purpose:** Assign probability scores indicating likelihood of denial based on historical patterns.
        
        **Dynamic Scoring:** Scores update based on claim attributes
        """)
        
        # Risk level distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_dist = denial_probabilities['Risk_Level'].value_counts()
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
                denial_probabilities,
                x='Denial_Probability',
                nbins=50,
                title='Denial Probability Distribution',
                color_discrete_sequence=['#dc2626']
            )
            fig.add_vline(x=75, line_dash="dash", line_color="red",
                         annotation_text="Critical Threshold")
            fig.add_vline(x=50, line_dash="dash", line_color="orange",
                         annotation_text="High Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk claims
        st.markdown("#### High-Risk Claims Requiring Immediate Attention")
        high_risk = denial_probabilities[
            denial_probabilities['Risk_Level'].isin(['High', 'Critical'])
        ].sort_values('Denial_Probability', ascending=False)
        
        st.dataframe(high_risk.head(20), use_container_width=True)
        
        # Risk component analysis
        st.markdown("#### Risk Score Component Breakdown")
        component_cols = ['Payer_Risk', 'Procedure_Risk', 'Provider_Risk', 
                         'Amount_Risk', 'Documentation_Risk', 'Authorization_Risk']
        component_means = denial_probabilities[component_cols].mean()
        
        fig = px.bar(
            x=component_cols,
            y=component_means.values,
            title='Average Risk Contribution by Component',
            labels={'x': 'Risk Component', 'y': 'Average Score'},
            color=component_means.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Prior Authorization Check
    with tab3:
        st.subheader(" Prior Authorization Check")
        st.markdown("""
        **Purpose:** Validate whether claims requiring prior authorization have appropriate documents.
        
        **Alert:** Flags cases missing mandatory authorization for high-risk procedures
        """)
        
        if auth_issues.empty:
            st.success(" All claims with prior authorization requirements are compliant")
        else:
            st.error(f"️ Found {len(auth_issues)} claims with authorization issues")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                critical_count = len(auth_issues[auth_issues['Severity'] == 'Critical'])
                st.metric("Critical Issues", critical_count)
            
            with col2:
                total_amount = auth_issues['Claim_Amount'].sum()
                st.metric("Amount at Risk", f"${total_amount:,.0f}")
            
            with col3:
                unique_providers = auth_issues['Hospital_ID'].nunique()
                st.metric("Affected Providers", unique_providers)
            
            # Issues by severity
            severity_dist = auth_issues['Severity'].value_counts()
            fig = px.pie(
                values=severity_dist.values,
                names=severity_dist.index,
                title='Authorization Issues by Severity',
                color_discrete_sequence=['#dc2626', '#ea580c', '#eab308']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed issues
            st.markdown("#### Detailed Authorization Issues")
            auth_display = auth_issues.sort_values('Severity', 
                                                   key=lambda x: x.map({'Critical': 0, 'High': 1, 'Medium': 2}))
            st.dataframe(auth_display, use_container_width=True)

    # Tab 4: Coding Compliance
    with tab4:
        st.subheader(" Coding Compliance Verification")
        if coding_issues.empty:
            st.success(" All CPT-ICD pairings are compliant")
        else:
            st.error(f"️ Found {len(coding_issues)} coding compliance issues")
            st.dataframe(coding_issues, use_container_width=True)

    # Tab 5: Payer Rules
    with tab5:
        st.subheader(" Payer-Specific Rule Engine")
        if payer_violations.empty:
            st.success(" All claims comply with payer rules")
        else:
            st.error(f"️ Found {len(payer_violations)} rule violations")
            st.dataframe(payer_violations, use_container_width=True)
            
            # Reference Rules
            st.markdown("#### Payer-Specific Rules Reference")
            rules_data = []
            for payer_id, rules in denial_engine.payer_rules.items():
                rules_data.append({
                    'Payer_ID': payer_id,
                    'Payer_Name': rules['name'],
                    'Max_Claim_Amount': f"${rules['max_claim_amount']:,}",
                    'Excluded_CPTs': ', '.join(rules['excluded_cpts'])
                })
            st.dataframe(pd.DataFrame(rules_data), use_container_width=True)

    # Tab 6: Resubmission Readiness
    with tab6:
        st.subheader(" Resubmission Readiness Check")
        if resubmission_readiness.empty:
            st.info("️ No denied claims to assess")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ready_count = len(resubmission_readiness[resubmission_readiness['Resubmission_Status'] == 'Ready'])
                st.metric("Ready for Resubmission", ready_count)
            with col2:
                potential_rec = resubmission_readiness[resubmission_readiness['Resubmission_Status'] == 'Ready']['Claim_Amount'].sum()
                st.metric("Potential Recovery", f"${potential_rec:,.0f}")
            
            st.dataframe(resubmission_readiness, use_container_width=True)

    # Tab 7: Strategic Recommendations (NEWLY UPDATED)
    with tab7:
        st.subheader(" AI Strategic Recommendations")
        st.markdown("""
        **Purpose:** Dynamic, prioritized actionable insights derived from the analysis of Authorization, Coding, and Payer Rule violations.
        """)
        
        # Display the dynamic recommendations
        if not recommendations_df.empty:
            # Sort by Priority (Critical first)
            priority_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
            recommendations_df['sort_key'] = recommendations_df['Priority'].map(priority_map)
            recommendations_df = recommendations_df.sort_values('sort_key').drop('sort_key', axis=1)
            
            st.data_editor(
                recommendations_df,
                column_config={
                    "Priority": st.column_config.TextColumn(
                        "Priority",
                        help="Urgency of the action",
                        validate="^(Critical|High|Medium|Low)$"
                    ),
                    "Estimated Impact": st.column_config.TextColumn(
                        "Estimated Financial Impact",
                        help="Projected recovery or savings"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No strategic recommendations needed. Systems operating optimally.")
            
    # Show Data Requirements Info
    with st.expander("️ Data Format Requirements"):
        st.code("""
        REQUIRED ADDITIONAL FIELDS:
        1. ICD_Code (e.g., I10, E11.9)
        2. Prior_Auth_Status (Approved/Pending/Missing)
        3. Resubmission_Count (0, 1, 2)
        """)

# Execution Block to load the specific data file
if __name__ == "__main__":
    try:
        # Load the global dataset as requested
        # Using a relative path as specified: ../healthcare_claims_data.csv
        df_claims = pd.read_csv('../healthcare_claims_data.csv')
        
        # Render the analysis with the loaded data
        render_denial_analysis(df_claims)
        
    except FileNotFoundError:
        st.error("Error: Global data file not found at '../healthcare_claims_data.csv'. Please ensure the file exists.")
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")