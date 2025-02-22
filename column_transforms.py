# If we do column transforms on the raw data, the data csv file would be large because we create new columns from each column.
# we generally only need to use transforms for a subset of all rows
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lags=3):
        self.lags = lags

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        X_transformed = pd.DataFrame(X)
        for lag in range(1, self.lags + 1):
            X_transformed[f'lag_{lag}'] = X_transformed.shift(lag)
        
        # X_transformed.dropna(inplace=True) # Drop NaNs from shifting
        return X_transformed.fillna(0)
    
class ZscoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ewm_period_m = 63, ewm_period_std_dev = 63):
        self.ewm_period_m = ewm_period_m
        self.ewm_period_std_dev = ewm_period_std_dev

    def fit(self, _):
        return self  # No fitting needed

    def transform(self, X):
        assert(X.isna().any().any() == False) # a good series
        X_norm = (X - X.ewm(span=self.ewm_period_m).mean()) / X.ewm(span=self.ewm_period_std_dev, min_periods=3).std()
        #X_norm.dropna(inplace=True) # Drop NaNs from shifting
        X_norm = X_norm.clip(-4, 4)
        return X_norm.fillna(0)
    

class VolScaler(BaseEstimator, TransformerMixin):
    def __init__(self, ewm_period_std_dev = 63):
        self.ewm_period_std_dev = ewm_period_std_dev

    def fit(self, _):
        return self  # No fitting needed

    def transform(self, X):
        assert(X.isna().any().any() == False) # a good series
        X_norm = X / X.ewm(span=self.ewm_period_std_dev, min_periods=3).std()
        #X_norm.dropna(inplace=True) # Drop NaNs from shifting
        X_norm = X_norm.clip(-4, 4)
        return X_norm.fillna(0)
    

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_bound = X.quantile(self.lower_quantile)
        self.upper_bound = X.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        return X.clip(self.lower_bound, self.upper_bound, axis=1)
    

