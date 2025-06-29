from sklearn.pipeline import Pipeline
from src.features import FeatureAggregator, DateTimeFeatures
from src.preprocessing import (
    TypeConverter, 
    OutlierHandler, 
    build_preprocessor
)
from src.config import (
    AGGREGATION_LEVELS,
    DATE_COLUMN,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL
)

def build_pipeline():
    """Builds the complete data processing pipeline"""
    
    # Feature engineering steps
    feature_engineering = Pipeline([
        ('type_converter', TypeConverter(NUMERICAL_COLS, CATEGORICAL_COLS)),
        ('date_features', DateTimeFeatures(DATE_COLUMN)),
        ('aggregator', FeatureAggregator(AGGREGATION_LEVELS)),
        ('outlier_handler', OutlierHandler(NUMERICAL_COLS))
    ])
    
    # Preprocessing steps
    all_numeric_cols = NUMERICAL_COLS + [
        f'{level}_{feat}' 
        for level in AGGREGATION_LEVELS
        for feat in ['TransactionCount', 'TotalAmount', 'AvgAmount', 'AmountStd', 'AmountSkew']
    ] + ['TransactionHour', 'TransactionDay', 'TransactionDayOfWeek', 
         'TransactionMonth', 'TransactionYear', 'IsWeekend']
    
    all_categorical_cols = CATEGORICAL_COLS
    
    preprocessor = build_preprocessor(all_numeric_cols, all_categorical_cols)
    
    # Complete pipeline
    pipeline = Pipeline([
        ('feature_engineering', feature_engineering),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline