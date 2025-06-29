# Configuration constants
RAW_DATA_PATH = '../data/raw/data.csv'
PROCESSED_DATA_PATH = '../data/processed/data_processed.csv'
MODEL_ARTIFACT_PATH = '../models/pipeline.pkl'

# Feature engineering parameters
AGGREGATION_LEVELS = ['CustomerId', 'ProductCategory']
DATE_COLUMN = 'TransactionStartTime'
CATEGORICAL_COLS = ['ProductCategory', 'CountryCode', 'CurrencyCode']
NUMERICAL_COLS = ['Amount', 'Value']
TARGET_COL = 'FraudResult'