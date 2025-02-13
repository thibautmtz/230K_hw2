import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import hashlib
import os
import datetime    
from datetime import timedelta
import utils 
import column_transforms
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from shorten_filenames import shorten_filename
import xgboost as xgb
from sklearn.linear_model import Lasso

beginning_oos = end_validation = datetime.date(2017, 2, 28)
end_oos = datetime.date(2022, 3, 31)
beginning_validation = beginning_oos - timedelta(days=365*2)

df = pd.read_csv("features.csv", index_col=[0,1])
df.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(df.index.get_level_values(0)).date, df.index.get_level_values(1)]
)
dates_index = pd.Series(df.loc[:end_validation, :].index.get_level_values(0).unique(), )

def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 1, log = True)

    accounting_ratios = [
        'log_bm',
        'roa', 
        'roe',
        'log_pcf',
    ]
    size_and_value = [
        'log_size',
        'log_age_lb', 
    ]
    mom_reversal_vol = [
        'mom',
        'vol', 
        'price', 
        'strev', 
    ]

    baselines = [
        'ols_3m_d', 
        'ols_1y_d',
        'ols_5y_m', 
    ]
    
    market_features = [
        'rf', 
        'rm', 
        'market_cap',
        'capm_return'
    ]

    include_accounting_ratios = trial.suggest_categorical('include_accounting_ratios', [True, False])
    include_size_and_value = trial.suggest_categorical('include_size_and_value', [True, False])
    include_mom_reversal_vol = trial.suggest_categorical('include_mom_reversal_vol', [True, False])
    include_market_features = trial.suggest_categorical('include_market_features', [True, False])
    append = trial.suggest_categorical('append', [True, False])

    features_to_use = baselines
    if include_accounting_ratios:
        features_to_use += accounting_ratios
    if include_size_and_value:
        features_to_use += size_and_value
    if include_mom_reversal_vol:
        features_to_use += mom_reversal_vol
    if include_market_features:
        features_to_use += market_features

    

    zscore_level_features = trial.suggest_categorical('zscore_level_features', [True, False])
    features_to_zscore = []

    if zscore_level_features:
        suggesed_features = ['log_bm', 'roa', 'roe', 'log_pcf', 'log_size', 'rm', 'rf', 'market_cap']
        features_to_zscore += [feature for feature in suggesed_features if feature in features_to_use]
    
    winsorize = trial.suggest_categorical('winsorize', [True, False])
    features_to_winsorize = []
    if winsorize:
        suggesed_features = ['log_bm', 'roa', 'roe', 'log_pcf', 'log_size', 'rm', 'rf', 'market_cap', 'mom', 'vol', 'price', 'strev', 'capm_return']
        features_to_winsorize += [feature for feature in suggesed_features if feature in features_to_use]

    for feature in features_to_use:
        if feature not in df.columns:
            print(f"Missing {feature}")
            break
        
    mape_scores = []

    for train_indices, val_indices, _ in utils.rolling_time_series_split(len(dates_index), 360, 60, 24):
        begin_train_xv = dates_index.loc[train_indices[0]]
        end_train_xv = dates_index.loc[train_indices[-1]]
        begin_validation_xv = dates_index.loc[val_indices[0]]
        end_validation_xv = dates_index.loc[val_indices[-1]]

        preprocessor = ColumnTransformer([
            ('zscore', column_transforms.ZscoreTransformer(ewm_period_m=12, ewm_period_std_dev=30, append = append), features_to_zscore),   # Standardize numerical columns
            ('winsorize', column_transforms.Winsorizer(), features_to_winsorize),   # Winsorize numerical columns               
        ], remainder = 'passthrough')  
        
        model = Lasso(alpha=alpha)

        df_X_train = df.loc[begin_train_xv:end_train_xv, features_to_use]
        df_X_train_transformed = preprocessor.fit_transform(df_X_train)

        df_y_train = df.loc[begin_train_xv:end_train_xv, 'f_ols_1y_d']

        model.fit(df_X_train_transformed, df_y_train)

        # Get the forecast for beginning_validation to end_validation period
        df_X_forecast = df.loc[begin_validation_xv:end_validation_xv, features_to_use]
        df_X_forecast_transformed = preprocessor.transform(df_X_forecast)
        df['forecast'] = pd.Series(model.predict(df_X_forecast_transformed), index = df_X_forecast.index) # we don't need to reindex the forecast

        mape = utils.value_weighted_mape(df, 'forecast', 'f_ols_1y_d', begin_validation_xv, end_validation_xv)
        mape_scores.append(mape)
    
    return np.mean(np.array(mape_scores))

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dir_if_not_exists('./outputs/contour_plots')
create_dir_if_not_exists('./outputs')

class Logger:
    def __init__(self):
        self.filename = f'./outputs/optuna_training_lasso_{datetime.datetime.now().isoformat()}.csv'
        self.buffer = ['name, value\n']
    def __call__(self, study, trial):
        name = ' '.join([f'{k}={v}' for k, v in trial.params.items()])
        self.buffer.append(f'{name},{trial.value}\n')
        if len(self.buffer) >= 20:
            with open(self.filename, 'a') as f:
                for line in self.buffer:
                    f.write(line)
            self.buffer = []

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=1200, n_jobs = -1, timeout = 3600, show_progress_bar=True, callbacks = [Logger()])


all_params = (list(study.best_params.keys()))
param_pairs = []

for i in range(len(all_params)):
    for j in range(i+1, len(all_params)):
        param_pairs.append([all_params[i], all_params[j]])


for i, param_pair in enumerate(param_pairs):
    fig = plt.figure()
    ax = optuna.visualization.matplotlib.plot_contour(study, params=param_pair)
    proposed_filename = f'output_lasso_{param_pair}_{datetime.datetime.now().isoformat()}'
    shortened_filename = shorten_filename(proposed_filename)
    plt.savefig(f"./outputs/contour_plots/{shortened_filename}.png")
    plt.close()
