import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# --- Generate 2000-row mock data ---
num_rows = 2000

data = {
    "Patient_ID": [f"P{1000+i}" for i in range(num_rows)],
    "Patient_Name": [fake.name() for _ in range(num_rows)],
    "Member_ID": [f"M{2000+i}" for i in range(num_rows)],
    "SSN": [fake.ssn() for _ in range(num_rows)],
    "DOB": [fake.date_of_birth(minimum_age=0, maximum_age=90) for _ in range(num_rows)],
    "Patient_Age": [fake.random_int(0, 90) for _ in range(num_rows)],
    "Patient_Gender": [fake.random_element(["Male","Female"]) for _ in range(num_rows)],
    "Patient_Zip": [fake.zipcode() for _ in range(num_rows)],
    "Hospital_ID": [f"H{100+i%50}" for i in range(num_rows)],
    "Hospital_Lat": [fake.latitude() for _ in range(num_rows)],
    "Hospital_Lon": [fake.longitude() for _ in range(num_rows)],
    "Patient_Lat": [fake.latitude() for _ in range(num_rows)],
    "Patient_Lon": [fake.longitude() for _ in range(num_rows)],
    "Hospital_City": [fake.city() for _ in range(num_rows)],
    "Bed_Capacity": [fake.random_int(50, 500) for _ in range(num_rows)],
    "Provider_ID": [f"PR{3000+i}" for i in range(num_rows)],
    "Claim_ID": [f"C{5000+i}" for i in range(num_rows)],
    "Insurer_ID": [f"I{600+i%10}" for i in range(num_rows)],
    "Disease_Category": [fake.random_element(["Cardio","Ortho","Onco","Pedia","General"]) for _ in range(num_rows)],
    "ICD_Code": [fake.bothify(text="??###") for _ in range(num_rows)],
    "Prior_Auth_Status": [fake.random_element(["Approved","Pending","Denied"]) for _ in range(num_rows)],
    "Resubmission_Count": [fake.random_int(0,3) for _ in range(num_rows)],
    "CPT_Code": [fake.bothify(text="CPT###") for _ in range(num_rows)],
    "Claim_Amount": [fake.random_int(1000, 50000) for _ in range(num_rows)],
    "Paid_Amount": [fake.random_int(500, 45000) for _ in range(num_rows)],
    "Status": [fake.random_element(["Paid","Denied","Pending"]) for _ in range(num_rows)],
    "Denial_Reason": [fake.sentence(nb_words=5) for _ in range(num_rows)],
    "denial_categories": [fake.random_element(["Coding","Documentation","Eligibility","Other"]) for _ in range(num_rows)],
    "Contracted_Rate": [fake.random_int(1000, 40000) for _ in range(num_rows)],
    "Scheme_Type": [fake.random_element(["Cashless","Reimbursement"]) for _ in range(num_rows)],
    "ABDM_Status": [fake.random_element(["Linked","Not Linked"]) for _ in range(num_rows)],
    "ABHA_ID": [f"ABHA{fake.random_number(digits=8)}" for _ in range(num_rows)],
    "Is_Disputed": [fake.random_element([0,1]) for _ in range(num_rows)],
    "Dispute_Open_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
    "Dispute_Close_Date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(num_rows)],
    "Litigation_Flag": [fake.random_element([0,1]) for _ in range(num_rows)],
    "Penalty_Amount": [fake.random_int(0,10000) for _ in range(num_rows)],
    "Procedure_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
    "Service_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
    "Submission_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
    "Predicted_Payment_Date": [fake.date_between(start_date='today', end_date='+90d') for _ in range(num_rows)],
    "Admission_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
    "Discharge_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
    "Length_Of_Stay": [fake.random_int(1,15) for _ in range(num_rows)],
    "Admission_Type": [fake.random_element(["Emergency","Planned"]) for _ in range(num_rows)],
    "Readmission_Flag": [fake.random_element([0,1]) for _ in range(num_rows)],
    "OT_Log_Found": [fake.random_element([0,1]) for _ in range(num_rows)],
    "OT_Record_ID": [f"OT{7000+i}" for i in range(num_rows)],
    "Clinical_Notes": [fake.sentence(nb_words=10) for _ in range(num_rows)],
    "Documentation_Score": [fake.random_int(1,5) for _ in range(num_rows)],
    "Billing_Type": [fake.random_element(["Package","Open"]) for _ in range(num_rows)],
    "Package_Rate": [fake.random_int(1000,40000) for _ in range(num_rows)],
    "Premium_Collected": [fake.random_int(5000,100000) for _ in range(num_rows)],
    "Premium_Allocation": [fake.random_int(5000,100000) for _ in range(num_rows)],
    "Payment_Ratio": [np.random.uniform(0.5,1.0) for _ in range(num_rows)],
    "Review_Platform": [fake.random_element(["Google","JustDial","Facebook"]) for _ in range(num_rows)],
    "Review_Rating": [fake.random_int(1,5) for _ in range(num_rows)],
    "NPS_Score": [fake.random_int(0,10) for _ in range(num_rows)],
    "Complaint_Category": [fake.random_element(["Billing","Service","Quality","Other"]) for _ in range(num_rows)],
    "Cashless_Approval_Hours": [fake.random_int(1,48) for _ in range(num_rows)],
    "TPA_Name": [fake.random_element(["In-House","Medi-X TPA","GlobalHealth TPA","FastTrack TPA"]) for _ in range(num_rows)],
    "Query_Count": [fake.random_int(0,5) for _ in range(num_rows)],
    "Doc_Resubmission_Count": [fake.random_int(0,2) for _ in range(num_rows)],
    "TPA_Response_Hours": [fake.random_int(2,72) for _ in range(num_rows)],
    "Days_To_Payment": [fake.random_int(1,60) for _ in range(num_rows)],
    "Contract_End_Date": [fake.date_between(start_date='today', end_date='+2y') for _ in range(num_rows)]
}

df = pd.DataFrame(data)

# --- Streamlit UI ---
st.title("Healthcare Claims Data (2000 Records)")
st.dataframe(df)  # Interactive scrollable table
