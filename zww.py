import warnings

from Tools.scripts.dutree import display

warnings.simplefilter('ignore')

import gc

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

train = pd.read_csv('train_dataset.csv')
# train
test = pd.read_csv('evaluation_public.csv')
# test
df = pd.concat([train, test])

roll_cols = ['JS_NH3',
             'CS_NH3',
             'JS_TN',
             'CS_TN',
             'JS_LL',
             'CS_LL',
             'MCCS_NH4',
             'MCCS_NO3',
             'JS_COD',
             'CS_COD',
             'JS_SW',
             'CS_SW',
             'B_HYC_NH4',
             'B_HYC_XD',
             'B_HYC_MLSS',
             'B_HYC_JS_DO',
             'B_HYC_DO',
             'B_CS_MQ_SSLL',
             'B_QY_ORP',
             'N_HYC_NH4',
             'N_HYC_XD',
             'N_HYC_MLSS',
             'N_HYC_JS_DO',
             'N_HYC_DO',
             'N_CS_MQ_SSLL',
             'N_QY_ORP']

df['time'] = pd.to_datetime(df['time'])
for i in range(1, 5):
    df[[ii + f'_roll_{i}_mean_diff' for ii in roll_cols]] = df[roll_cols].rolling(i, min_periods=1).sum().diff()

df[[ii + '_roll_8_mean' for ii in roll_cols]] = df[roll_cols].rolling(8, min_periods=1).mean()
df[[ii + '_roll_16_mean' for ii in roll_cols]] = df[roll_cols].rolling(16, min_periods=1).mean()

df[[ii + '_roll_16_mean_diff' for ii in roll_cols]] = df[[ii + '_roll_16_mean' for ii in roll_cols]].diff()
df[[ii + '_roll_8_mean_diff' for ii in roll_cols]] = df[[ii + '_roll_8_mean' for ii in roll_cols]].diff()

df[[ii + '_roll_8_std' for ii in roll_cols]] = df[roll_cols].rolling(8, min_periods=1).std()
train = df.iloc[:train.shape[0]]
test = df.iloc[train.shape[0]:]

N_col = ['N_HYC_NH4',
         'N_HYC_XD',
         'N_HYC_MLSS',
         'N_HYC_JS_DO',
         'N_HYC_DO',
         'N_CS_MQ_SSLL',
         'N_QY_ORP']

B_col = ['B_HYC_NH4',
         'B_HYC_XD',
         'B_HYC_MLSS',
         'B_HYC_JS_DO',
         'B_HYC_DO',
         'B_CS_MQ_SSLL',
         'B_QY_ORP']

NB_col = ['A_' + ii[2:] for ii in ['B_HYC_NH4',
                                   'B_HYC_XD',
                                   'B_HYC_MLSS',
                                   'B_HYC_JS_DO',
                                   'B_HYC_DO',
                                   'B_CS_MQ_SSLL',
                                   'B_QY_ORP']]
train[NB_col] = train[B_col].values / (train[N_col].values + 1e-3)
test[NB_col] = test[B_col].values / (test[N_col].values + 1e-3)
# NB_col
# 1. 数据说明里表示，北生化池和南生化池在生产过程中不会互相影响, 可以先试下分开两部分
# 2. 只用有 label 的数据

train_B = train[[i for i in train.columns if (i != 'Label2' and not i.startswith('N_'))]].copy()
train_N = train[[i for i in train.columns if (i != 'Label1' and not i.startswith('B_'))]].copy()

train_B = train_B[train_B['Label1'].notna()].copy().reset_index(drop=True)
train_N = train_N[train_N['Label2'].notna()].copy().reset_index(drop=True)

test_B = test[[i for i in test.columns if not i.startswith('N_')]].copy()
test_N = test[[i for i in test.columns if not i.startswith('B_')]].copy()


# 时间特征
def add_datetime_feats(df):
    df['time'] = pd.to_datetime(df['time'])
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek

    return df


train_B = add_datetime_feats(train_B)
train_N = add_datetime_feats(train_N)
test_B = add_datetime_feats(test_B)
test_N = add_datetime_feats(test_N)


# 做点比率数值特征
def add_ratio_feats(df, type_='B'):
    df['JS_CS_NH3_ratio'] = df['JS_NH3'] / (df['CS_NH3'] + 1e-3)
    df['JS_CS_TN_ratio'] = df['JS_TN'] / (df['CS_TN'] + 1e-3)
    df['JS_CS_LL_ratio'] = df['JS_LL'] / (df['CS_LL'] + 1e-3)
    df['MCCS_NH4_NH3_ratio'] = df['MCCS_NH4'] / (df['CS_NH3'] + 1e-3)
    df['MCCS_NO3_NH3_ratio'] = df['MCCS_NO3'] / (df['CS_NH3'] + 1e-3)
    df['JS_CS_COD_ratio'] = df['JS_COD'] / (df['CS_COD'] + 1e-3)
    df['JS_CS_SW_ratio'] = df['JS_SW'] / (df['CS_SW'] + 1e-3)
    df['HYC_DO_ratio'] = df[f'{type_}_HYC_JS_DO'] / (df[f'{type_}_HYC_DO'] + 1e-3)
    df['CS_MQ_LL_ratio'] = df[f'{type_}_CS_MQ_SSLL'] / (df['CS_LL'] + 1e-3)

    return df


train_B = add_ratio_feats(train_B, type_='B')
train_N = add_ratio_feats(train_N, type_='N')
test_B = add_ratio_feats(test_B, type_='B')
test_N = add_ratio_feats(test_N, type_='N')

# target log1p 转换

B_max, B_min = train_B['Label1'].max(), train_B['Label1'].min()
N_max, N_min = train_N['Label2'].max(), train_N['Label2'].min()

train_B['Label1'] = np.log1p(train_B['Label1'])
train_N['Label2'] = np.log1p(train_N['Label2'])


def run_lgb(df_train, df_test, ycol, n_splits=5, seed=2022):
    use_feats = [col for col in df_test.columns if col not in ['time', 'Label1', 'Label2', 'label']]
    model = lgb.LGBMRegressor(num_leaves=32, objective='mape',
                              max_depth=16,
                              learning_rate=0.1,
                              n_estimators=10000,
                              subsample=0.8,
                              feature_fraction=0.6,
                              reg_alpha=0.5,
                              reg_lambda=0.25,
                              random_state=seed,
                              metric=None)
    oof = []
    prediction = df_test[['time']]
    prediction[ycol] = 0
    df_importance_list = []
    from tscv import GapKFold
    cv = GapKFold(n_splits=n_splits, gap_before=0, gap_after=0)
    for fold_id, (trn_idx, val_idx) in enumerate(cv.split(df_train[use_feats])):
        X_train = df_train.iloc[trn_idx][use_feats]
        Y_train = df_train.iloc[trn_idx][ycol]
        X_val = df_train.iloc[val_idx][use_feats]
        Y_val = df_train.iloc[val_idx][ycol]
        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=100,
                              eval_metric='rmse',
                              early_stopping_rounds=100)
        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = df_train.iloc[val_idx][['time', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)
        pred_test = lgb_model.predict(df_test[use_feats], num_iteration=lgb_model.best_iteration_)
        prediction[ycol] += pred_test / n_splits
        df_importance = pd.DataFrame({
            'column': use_feats,
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)
        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['importance'].agg(
        'mean').sort_values(ascending=False).reset_index()
    print(df_importance.head(50))
    df_oof = pd.concat(oof).reset_index(drop=True)
    df_oof[ycol] = np.expm1(df_oof[ycol])
    df_oof['pred'] = np.expm1(df_oof['pred'])
    prediction[ycol] = np.expm1(prediction[ycol])

    return df_oof, prediction


df_oof_B, pred_B = run_lgb(train_B, test_B, ycol='Label1', n_splits=10)
df_oof_N, pred_N = run_lgb(train_N, test_N, ycol='Label2', n_splits=10)

def calc_score(df1, df2):
    rmse_1 = np.sqrt(mean_squared_error(df1['pred'], (df1['Label1'])))
    rmse_2 = np.sqrt(mean_squared_error(df2['pred'], (df2['Label2'])))
    loss = (rmse_1+rmse_2)/2
    print(rmse_1,rmse_2)
    score = (1 / (1 + loss)) * 1000
    return score

calc_score(df_oof_B, df_oof_N)


sub = pd.read_csv('sample_submission.csv')
sub['Label1'] = pred_B['Label1'].values
sub['Label2'] = pred_N['Label2'].values
sub

# 0.7814041946461452
# 0.7864133808960185
sub.to_csv('zww.csv', index=False)

