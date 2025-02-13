import pandas as pd
import numpy as np
eps = 1e-8 # zero in the float world

def value_weighted_mse(full_dataframe, forecast_column, actual_column, start_date, end_date):
    grouped_by_date = full_dataframe.loc[start_date:end_date].groupby(level=0)
    return grouped_by_date\
        .apply(lambda x: np.average((x[forecast_column] - x[actual_column])**2, weights=x['market_cap']))\
        .mean()

def value_weighted_mape(full_dataframe, forecast_column, actual_column, start_date, end_date):
    grouped_by_date = full_dataframe.loc[start_date:end_date].groupby(level = 0)
    return grouped_by_date\
        .apply(lambda x: 100 * np.average(
            np.where(
                np.abs(x[actual_column]) > eps, 
                np.abs(x[forecast_column] - x[actual_column])/x[actual_column], 
                0
            ),
            weights=x['market_cap']))\
        .mean()

def mse(full_dataframe, forecast_column, actual_column, start_date, end_date):
    grouped_by_date = full_dataframe.loc[start_date:end_date].groupby(level=0)
    return grouped_by_date.apply(lambda x: ((x[forecast_column] - x[actual_column]) ** 2).mean()).mean()

def mape(full_dataframe, forecast_column, actual_column, start_date, end_date):
    eps = 1e-10  # Small value to prevent division by zero
    grouped_by_date = full_dataframe.loc[start_date:end_date].groupby(level=0)
    return grouped_by_date.apply(
        lambda x: 100 * np.mean(
            np.where(
                np.abs(x[actual_column]) > eps, 
                np.abs(x[forecast_column] - x[actual_column]) / np.abs(x[actual_column]), 
                0
            )
        )
    ).mean()

def rolling_time_series_split(n, train_window, test_window, step):
    for i in range(0, n - train_window - test_window, step):
        train_indices = range(i, i+train_window)
        test_indices = range(i+train_window, i+train_window+test_window)
        yield train_indices, test_indices, i

def expanding_window_time_series_split(n, train_window, test_window, step):
    for i in range(train_window, n - test_window, step):
        train_indices = range(0, i)
        test_indices = range(i, i+test_window)
        yield train_indices, test_indices, i