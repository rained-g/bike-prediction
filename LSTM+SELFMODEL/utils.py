import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# columns = [
#         # 'instant',
#         'dteday',
#         "season",
#         'yr',
#         'mnth',
#         'hr',
#         'holiday',
#         'weekday',
#         'workingday',
#         'weathersit',
#         'temp',
#         'atemp',
#         'hum',
#         'windspeed',
#     ]
def create_dataset(dataset, lookback=96, lookafter=96, columns=None, device=None):
    n = dataset.shape[0]
    X, y = [], []
    temp = []
    dataset = dataset.values
    for i in range(n -lookback-lookafter - 1):
        X.append(dataset[i:i + lookback, 0:len(columns)])
        y.append(dataset[i + lookback:i + lookback + lookafter, -1])
    # print(X,y)
    return torch.Tensor(np.array(X)).to(device), torch.Tensor(np.array(y)).to(device)

# 将日期列转换为星期几
def get_weekday(date_str):
    date = datetime.datetime.strptime(date_str, "%Y/%m/%d")
    weekday = date.weekday()
    weekdays = [0,1,2,3,4,5,6]
    return weekdays[weekday]

def get_dataset(lookback=96, lookafter=96, columns=None, device=None):

    datasets = pd.read_csv('../data/train_data.csv', header=0)
    datasets_test = pd.read_csv('../data/test_data.csv', header=0)

    # datasets = datasets.drop(['instant', 'casual', 'registered'], axis=1)
    # datasets_test = datasets_test.drop(['instant', 'casual', 'registered'], axis=1)

    datasets = datasets[columns]
    datasets_test = datasets_test[columns]

    scaler = MinMaxScaler(feature_range=(0, 1))

    datasets = datasets.dropna()
    datasets_test = datasets_test.dropna()
    if 'dteday' in columns:
        datasets['dteday'] = datasets['dteday'].apply(get_weekday)
        datasets_test['dteday'] = datasets_test['dteday'].apply(get_weekday)

    # datasets['cnt'] = scaler.fit_transform(datasets['cnt'])
    cnt_column = datasets['cnt']
    cnt_column = cnt_column.values.reshape(-1, 1)
    cnt_column_scaled = scaler.fit_transform(cnt_column)
    datasets['cnt'] = cnt_column_scaled

    cnt_column = datasets_test['cnt']
    cnt_column = cnt_column.values.reshape(-1, 1)
    cnt_column_scaled = scaler.fit_transform(cnt_column)
    datasets_test['cnt'] = cnt_column_scaled
    # datasets_test['cnt'] = scaler.fit_transform(datasets_test['cnt'])

    trainX, trainY = create_dataset(datasets, lookback, lookafter, columns=columns,device=device)

    testX, testY = create_dataset(datasets_test, lookback, lookafter, columns=columns, device=device)

    return trainX, trainY, testX, testY