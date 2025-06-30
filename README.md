# Credit Risk model

## Project Overview

This project aims to build a smart credit scoring engine for Bati Bank’s Buy-Now-Pay-Later (BNPL) service by transforming raw eCommerce behavioral data into predictive insights. Using customer Recency, Frequency, and Monetary (RFM) patterns, we engineer a proxy for credit default risk in the absence of traditional credit history. The model outputs a risk probability score aligned with Basel II regulatory standards, enabling Bati Bank to make data-driven lending decisions, promote financial inclusion, and expand responsible credit access in Ethiopia’s digital commerce ecosystem.

---

## Credit Scoring Business Understanding

The development of a credit scoring model in a regulated financial context requires careful consideration of risk measurement, data limitations, and model interpretability to ensure alignment with business objectives and regulatory requirements.

### Basel II Accord and Risk Measurement
The Basel II Accord mandates rigorous, transparent credit risk assessment to ensure institutions hold sufficient capital against potential defaults. This creates a strong need for interpretable and well-documented models that clearly explain risk factors and support regulatory audits and internal governance.

### Necessity and Risks of Proxy Variables for Default
When actual default labels are unavailable, proxy variables—like late payments or behavioral patterns—are essential to estimate credit risk. However, these proxies can introduce business risks such as misclassification, leading to either overly conservative lending or underestimated risk, both of which can harm profitability and compliance.

### Trade-offs Between Simple and Complex Models
Simple models like Logistic Regression with Weight of Evidence (WoE) offer regulatory-friendly transparency and ease of explanation, while complex models like Gradient Boosting improve predictive power but reduce interpretability. In regulated settings, the preference often leans toward interpretable models, though high-performing models can be used in champion-challenger setups to strike a balance between performance and compliance.

---

## Project Structure
---
```
credit-risk-model/
├── .github/
│ └── workflows/ # GitHub Actions workflows
├── data/
│ ├── raw/ # Raw data (should never be modified)
│ └── processed/ # Processed/cleaned data (gitignored)
├── notebooks/
  │ └── 1.0_eda.ipynb # performs eda and statistics analysis on data
  │ └── 04_RFM_Target_Variable_Engineering.ipynb # rfm calculation
│ └── README.md # Documentation for notebooks
├── scripts/
│ └── README.md # Documentation for scripts
├── src/
│ └── utils/ # Utility functions
│  │ └── data_loader.py # loads csv files
│ └── config.py # contains constants
│ └── features.py # features selection
│ └── preprocessing.py # data preprocessing
│ └── pipeline.py # pipeline for feature engineering and preprocessing
│ └── run_pipeline.py # runs the pipeline and saves the output
│ └── README.md # Documentation for source code
├── tests/
│ └── README.md # Testing documentation
├── .gitattributes
├── .gitignore
├── README.md # Main project documentation
└── requirements.txt # Python dependencies
```
---
## 🛠 Tools & Technologies

- Python  
- Pandas, NumPy  
- NLTK (text preprocessing)  
- Scikit-learn (TF-IDF, LDA)  
- Matplotlib, Seaborn (visualization)  
- Jupyter Notebook  
- Git  

---

## Key Task Completed 

### ✅ Task 1 - Understanding Credit Risk 

Understood credit risk

### ✅ Task 2 - Exploratory Data Analysis (EDA) 

Conducted exploratory data analysis on 95,662 transactions. Checked data types and confirmed no missing values. Analyzed numerical and categorical features, identified outliers and skewed distributions, and explored correlations—especially between `Amount` and `Value`. Findings will guide feature engineering and modeling.

### ✅ Task 3 - Feature Engineering

Built a reproducible data pipeline using sklearn.pipeline.Pipeline. Created aggregate and time-based features, encoded categorical variables, handled missing values, and scaled numerical data. All transformations were implemented in modular scripts under the src/ directory.

### ✅ Task 4 - Proxy Target Variable Engineering
Engineered a proxy target variable is_high_risk using RFM-based customer segmentation. Calculated Recency, Frequency, and Monetary metrics per customer, scaled features, and applied K-Means clustering to identify disengaged customers. Labeled the least engaged cluster as high-risk (1) and others as low-risk (0). Integrated the binary target into the processed dataset for downstream modeling.

---

## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/she-code/credit-risk-model
cd credit-risk-model
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
3. **Install dependencies:**

```bash
pip install -r requirements.txt

```
