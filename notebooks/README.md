# Notebooks

This folder contains Jupyter notebooks illustrating the process of data ingestion, cleaning, and initial exploration of Telegram e-commerce data.

### 1.0_eda.ipynb

Checked data shape, types, and confirmed no missing values.

Reviewed summary statistics for numerical features.

Plotted distributions of Amount and Value → found right skewness and outliers.

Explored categorical features → noted imbalances and high cardinality.

Computed correlation matrix → found positive correlation between Amount and Value.

### 04_RFM_Target_Variable_Engineering.ipynb

Calculated RFM metrics per CustomerId using transaction history.

Scaled RFM features and applied K-Means clustering (k=3, random_state set).

Identified least engaged cluster as high-risk based on low frequency and monetary value.

Created binary proxy target column is_high_risk (1 = high-risk, 0 = otherwise).

Merged target variable into main processed dataset for model training.