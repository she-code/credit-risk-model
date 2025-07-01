import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAggregator(BaseEstimator, TransformerMixin):
    """Creates aggregate features at specified levels"""

    def __init__(self, aggregation_levels):
        self.aggregation_levels = aggregation_levels
        self.aggregate_features = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for level in self.aggregation_levels:
            # Group by aggregation level
            group = X.groupby(level)

            # Create aggregate features
            X[f"{level}_TransactionCount"] = X[level].map(group.size())
            X[f"{level}_TotalAmount"] = X[level].map(group["Amount"].sum())
            X[f"{level}_AvgAmount"] = X[level].map(group["Amount"].mean())
            X[f"{level}_AmountStd"] = X[level].map(group["Amount"].std())
            X[f"{level}_AmountSkew"] = X[level].map(group["Amount"].skew())

            # Fill NA values created by aggregation
            X.fillna({f"{level}_AmountStd": 0, f"{level}_AmountSkew": 0}, inplace=True)

        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extracts datetime features from specified column"""

    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])

        X["TransactionHour"] = X[self.date_column].dt.hour
        X["TransactionDay"] = X[self.date_column].dt.day
        X["TransactionDayOfWeek"] = X[self.date_column].dt.dayofweek
        X["TransactionMonth"] = X[self.date_column].dt.month
        X["TransactionYear"] = X[self.date_column].dt.year
        X["IsWeekend"] = X[self.date_column].dt.dayofweek.isin([5, 6]).astype(int)

        return X.drop(columns=[self.date_column])
