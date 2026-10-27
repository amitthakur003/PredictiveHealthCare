---

#  Healthcare Claims Analytics Platform

A comprehensive, full-stack analytics application designed to revolutionize healthcare claims processing. This platform leverages **Generative AI**, **Supervised Machine Learning**, and **Unsupervised Learning** to predict claim denials, detect fraud, and optimize financial workflows.

---

##  Key Features & Capabilities

### 🧠 AI & Machine Learning Core

| Feature Category | AI Technique Used | Tech Stack | Complexity |
| --- | --- | --- | --- |
| **Denial Prediction** | Supervised Classification | Scikit-learn, XGBoost | Medium |
| **Smart Fixes/Explanation** | Generative AI (LLM) | Google Gemini API | High |
| **Fraud Detection** | Unsupervised Learning | PyOD (Outlier Detection) | High |
| **Data Scrubbing** | Rule-Based Logic | Pandas (Advanced) | Low |

###  Functional Modules

* **Executive Dashboard:** Real-time KPI monitoring (Net Collection Rate, Denial Rate, Days in AR).
* **AI Assistant:** Chat-based interface to query data and get explanations for denied claims using Gemini.
* **Financial Intelligence:** ROI analysis, revenue leakage detection, and payer performance tracking.
* **Operational Workflow:** Tools to streamline TPA management, patient experience, and compliance governance.

---

##  Technical Architecture

The project is structured as a modular **Streamlit** application with a clear separation of concerns between data processing, ML training, and UI rendering.

```text
healthcare_claims_analytics/
├── app.py                          # Main application entry point
├── config/                         # Configuration & Environment settings
├── utils/                          # Utilities (Data Loader, Preprocessor, ML Trainer)
├── features/                       # Feature Modules
│   ├── dashboard.py                # Executive Dashboard
│   ├── ai_assistant.py             # GenAI Integration
│   ├── predictions.py              # ML Inference Engine
│   ├── fraud_detection.py          # Anomaly Detection
│   ├── denial_analysis.py          # Root Cause Analysis
│   └── ... (See full list below)
└── components/                     #  UI Components (Sidebar, Widgets)

```

---

##  Installation & Setup

### Prerequisites

* Python 3.9+
* A Google Cloud API Key (for Gemini AI)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/healthcare-claims-analytics.git
cd healthcare_claims_analytics

```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment

Create a `.env` file in the root directory and add your API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here

```

### 5. Run the Application

```bash
streamlit run app.py

```

*Access the app at `http://localhost:8501*`

---

##   Data Format Requirements

To ensure the ML models and analytics function correctly, your input CSV/Excel file should contain the following columns:

* `Claim_ID`, `Hospital_ID`, `Insurer_ID`
* `Disease_Category`, `CPT_Code`
* `Claim_Amount`, `Paid_Amount`
* `Status` (Paid/Denied/Pending), `Denial_Reason`
* `Submission_Date`, `Days_To_Payment`
* `Readmission_Flag` (0/1), `Documentation_Score` (1-5)

---

##  Roadmap & Feature Status

**Core Analytics**

* [x] Executive Dashboard
* [x] AI Assistant (Gemini Integration)
* [x] Predictive Modeling Engine
* [x] Advanced Analytics

**Data Integrity & Finance**

* [x] Automated Data Validation
* [x] Fraud/Anomaly Detection
* [x] Denial Risk Analysis
* [x] Financial Intelligence
* [ ] Quality Control Modules

**Operations & Compliance**

* [x] Compliance Governance
* [ ] Operational Workflow
* [ ] Network & Geography Analysis
* [ ] Patient Experience (NPS)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/NewFeature`.
3. Add your feature logic in the `features/` directory.
4. Register your feature in `config/settings.py`.
5. Submit a Pull Request.

---

**Built with ❤️ form Amit Thakur => using Streamlit and Google Gemini.**

---
