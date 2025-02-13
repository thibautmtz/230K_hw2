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
import xgboost as xgb
from shorten_filenames import shorten_filename

beginning_oos = end_validation = datetime.date(2017, 2, 28)
end_oos = datetime.date(2022, 3, 31)
beginning_validation = beginning_oos - timedelta(days=365*2)

df = pd.read_csv("features.csv", index_col=[0,1])
df.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(df.index.get_level_values(0)).date, df.index.get_level_values(1)]
)

def objective(trial):
    max_depth = trial.suggest_int('max_depth', 1, 8)
    n_estimators = trial.suggest_int('n_estimators', 1, 8)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1, log = True)
    eta = trial.suggest_float('eta', 0, 1)

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
        
    preprocessor = ColumnTransformer([
        ('zscore', column_transforms.ZscoreTransformer(ewm_period_m=7, ewm_period_std_dev=30, append = False), features_to_zscore),   
        ('_', column_transforms.Winsorizer(), features_to_winsorize),  
    ], remainder = 'passthrough').set_output(transform = 'pandas')

    model = xgb.XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, eta = eta, reg_alpha = reg_alpha)

    df_X_train = df.loc[:beginning_validation, features_to_use]
    df_X_train_transformed = preprocessor.fit_transform(df_X_train)

    df_y_train = df.loc[:beginning_validation, 'f_ols_1y_d']

    model.fit(df_X_train_transformed, df_y_train)

    # Get the forecast for beginning_validation to end_validation period
    df_X_forecast = df.loc[beginning_validation:end_validation, features_to_use]
    df_X_forecast_transformed = preprocessor.transform(df_X_forecast)

    df['forecast'] = pd.Series(model.predict(df_X_forecast_transformed), index = df_X_forecast.index) # we don't need to reindex the forecast

    mape = utils.value_weighted_mape(df, 'forecast', 'f_ols_1y_d', beginning_validation, end_validation)
    return mape
     

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dir_if_not_exists('./outputs/contour_plots')
create_dir_if_not_exists('./outputs')

class Logger:
    def __init__(self):
        self.filename = f'./outputs/optuna_training_xgb_{datetime.datetime.now().isoformat()}.csv'
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
study.optimize(objective, n_trials=10, n_jobs = -1, timeout = 3600, show_progress_bar=True, callbacks = [Logger()])


all_params = (list(study.best_params.keys()))
param_pairs = []

for i in range(len(all_params)):
    for j in range(i+1, len(all_params)):
        param_pairs.append([all_params[i], all_params[j]])


for i, param_pair in enumerate(param_pairs):
    fig = plt.figure()
    ax = optuna.visualization.matplotlib.plot_contour(study, params=param_pair)
    proposed_filename = f'output_{param_pair}_{datetime.datetime.now().isoformat()}'
    shortened_filename = shorten_filename(proposed_filename)
    plt.savefig(f"./outputs/contour_plots/.png")
    plt.close()
