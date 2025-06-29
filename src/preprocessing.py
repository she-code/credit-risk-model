
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
class TypeConverter(BaseEstimator, TransformerMixin):
    """Converts columns to specified data types"""
    
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert numeric columns
        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Convert categorical columns
        for col in self.categorical_cols:
            X[col] = X[col].astype('category')
            
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handles outliers using IQR method"""
    
    def __init__(self, numeric_cols, threshold=1.5):
        self.numeric_cols = numeric_cols
        self.threshold = threshold
        self.iqr = {}
        self.bounds = {}
        
    def fit(self, X, y=None):
        for col in self.numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.iqr[col] = iqr
            self.bounds[col] = (q1 - self.threshold*iqr, q3 + self.threshold*iqr)
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.numeric_cols:
            lower, upper = self.bounds[col]
            X[col] = np.where(X[col] < lower, lower, 
                             np.where(X[col] > upper, upper, X[col]))
        return X

def build_preprocessor(numeric_cols, categorical_cols):
    """Builds the preprocessing pipeline"""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor