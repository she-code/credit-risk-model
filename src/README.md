# Src

### features.py
Built FeatureAggregator to calculate per-group transaction stats (counts, sums, averages)

Created DateTimeFeatures to extract time components (hour/day/month) and weekend flags

Designed sklearn-compatible transformers with proper fit/transform methods

Ensured data safety with copy operations and null handling

Generated 10+ engineered features while maintaining pipeline compatibility

### preprocessing.py
Converted numeric/categorical columns and enforced proper data types.

Implemented IQR-based outlier capping for numerical features.

Built preprocessing pipeline:

Numerical: Median imputation → Standard scaling

Categorical: Mode imputation → One-hot encoding

Handled edge cases:

Invalid numeric conversions → coerced to NaN

Unseen categories → ignored during transform

Missing values → auto-filled per feature type

### pipeline.py
Constructed end-to-end data processing pipeline with:

Feature Engineering:
✓ Type conversion → Numerical/Categorical
✓ Datetime feature extraction
✓ Customer/product transaction aggregations
✓ Outlier capping

Preprocessing:
✓ Standardized 15+ numerical features
✓ One-hot encoded categorical variables
✓ Handled 100% of missing values

### run_pipeline.py
Executed end-to-end data processing:

Loaded raw transaction data from specified path

Separated features and target variable

Applied complete preprocessing pipeline:
✓ Feature engineering
✓ Outlier handling
✓ Feature scaling/encoding

Saved processed data with target column

Exported pipeline artifact for inference

Verified successful execution with output paths:

### train_model.py

Loaded processed dataset with engineered features and proxy target is_high_risk.

Split data into train/test sets using stratified sampling to preserve class distribution.

Scaled numerical features using StandardScaler.

Trained and evaluated three models: Logistic Regression, Random Forest, and Gradient Boosting.

Applied GridSearchCV for hyperparameter tuning using ROC-AUC as the scoring metric.

Logged model parameters, metrics (Accuracy, Precision, Recall, F1, ROC-AUC), and artifacts using MLflow.

Registered the best-performing model in the MLflow Model Registry for future deployment.