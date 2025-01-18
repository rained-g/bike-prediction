from argparse import Namespace
from cProfile import label

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from utils import *
from Module_LSTM import Module_LSTM

def plot_data(origin_plot, train_plot, lookafter=96):
    X = []
    y = []
    n = len(origin_plot)
    # origin_plot = origin_plot.cpu().numpy()
    # train_plot = train_plot.cpu().numpy()

    # 检查并转换 origin_plot
    if isinstance(origin_plot, torch.Tensor):
        origin_plot = origin_plot.cpu().numpy()
    # 检查并转换 train_plot
    if isinstance(train_plot, torch.Tensor):
        train_plot = train_plot.cpu().numpy()

    for i in range(0, n, lookafter):
        X.extend(origin_plot[i].flatten())
        y.extend(train_plot[i].flatten())
    plt.plot(range(len(X)), X)
    plt.plot(range(len(y)), y)

    plt.show()

def eval(args):
    lookback = 96
    lookafter = 96


    columns = [
        # 'instant',
        # 'dteday',
        "season",
        'yr',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday',
        'weathersit',
        'temp',
        'atemp',
        'hum',
        'windspeed',
        # 'casual',
        # 'registered',
        'cnt'
    ]

    datasets = pd.read_csv('../data/train_data.csv', header=0)
    datasets_test = pd.read_csv('../data/test_data.csv', header=0)

    # datasets = datasets.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)
    # datasets_test = datasets_test.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = datasets.dropna()
    datasets_test = datasets_test.dropna()

    cnt = datasets.cnt.to_numpy()
    scaler_cnt = MinMaxScaler(feature_range=(0, 1))
    scaler_cnt.fit_transform(cnt.reshape(-1, 1))


    # datasets_scaled = scaler.fit_transform(datasets)
    # datasets_test_scaled = scaler.fit_transform(datasets_test)

    # trainX, trainY = create_dataset(datasets_scaled, lookback, lookafter, columns=columns,device=device)
    # testX, testY = create_dataset(datasets_test_scaled, lookback, lookafter, columns=columns, device=device)
    trainX, trainY, testX, testY = get_dataset(lookback, lookafter=args.lookafter, columns=columns, device=device)
    # model = Module_LSTM(input_size=len(columns)).to(device)

    # model = torch.load('../models/1/LSTM_240_R[50]_L[2]_size[256].pth').to(device)
    model = torch.load(f'../models/{args.i}/{args.model_name}_{args.lookafter}_R[50]_L[2]_size[256].pth').to(device)
    # model = torch.load('../models/1/LSTMTransformer1_240_R[50]_L[2]_size[256].pth').to(device)

    with torch.no_grad():
        y_pred = model(trainX)
        test_y_pred = model(testX)
        loss_fn = nn.MSELoss()
        loss_fn2 = nn.L1Loss()
        train_plot = scaler_cnt.inverse_transform(y_pred.detach().cpu().numpy())
        test_plot = scaler_cnt.inverse_transform(test_y_pred.detach().cpu().numpy())
        # origin_plot = np.squeeze(scaler_cnt.inverse_transform(datasets_scaled[:,-1].reshape(-1, 1)), axis=1)
        # test_origin_plot = np.squeeze(scaler_cnt.inverse_transform(testY.reshape(-1, 1).detach().cpu().numpy()), axis=1)
        # print(train_plot.shape[0]+97, datasets_scaled.shape)
        # print(train_plot.shape, datasets_scaled.shape)
        train_mse = loss_fn(y_pred.cpu(), trainY.cpu())
        train_mae = loss_fn2(y_pred.cpu(), trainY.cpu())
        train_mape = (loss_fn2(y_pred.cpu(), trainY.cpu())) / y_pred.cpu().mean()

        test_mse = loss_fn(test_y_pred.cpu(), testY.cpu())
        test_mae = loss_fn2(test_y_pred.cpu(), testY.cpu())
    # print(" train MSE %.4f, test MSE %.4f train MAE %.4f test MAE %.4f train MAPE %.4f" %
    #       (train_mse, test_mse, train_mae, test_mae, train_mape))
        # pred_plot = np.array(train_plot.tolist() + test_plot.tolist())

    # print(origin_plot)
    trainY = scaler_cnt.inverse_transform(trainY.detach().cpu().numpy())
    testY = scaler_cnt.inverse_transform(testY.detach().cpu().numpy())
    # for i in range(1, 2, 240):
    #     plt.plot(range(trainY[i].shape[0]), trainY[i], label='truth')
    #     plt.plot(range(train_plot[i].shape[0]), train_plot[i], label='pred')
    #     plt.title('train')
    #     plt.show()
    train_mse = loss_fn(torch.from_numpy(train_plot), torch.from_numpy(trainY))
    test_mse = loss_fn(torch.from_numpy(test_plot), torch.from_numpy(testY))
    train_mae = loss_fn2(torch.from_numpy(train_plot), torch.from_numpy(trainY))
    test_mae = loss_fn2(torch.from_numpy(test_plot), torch.from_numpy(testY))

    print(" train MSE %.4f, test MSE %.4f train MAE %.4f test MAE %.4f train MAPE %.4f" %
          (train_mse, test_mse, train_mae, test_mae, train_mape))
    # print(test_mse1)

    # for i in range(0, 1, 96):
    #     plt.plot(range(testY[i].shape[0]), testY[i], label='truth')
    #     plt.plot(range(test_plot[i].shape[0]), test_plot[i], label='pred')
    #     plt.title('test')
    #     plt.show()
    # plot_data(trainY, train_plot, lookafter=lookafter)
    # plot_data(testY, test_plot, lookafter=lookafter)
    # plot_data(testY, test_plot[::-1], lookafter=lookafter)


    # plt.plot(range(test_plot.shape[0]), test_origin_plot)
    # plt.plot(range(test_plot.shape[0]), test_plot)


    # plt.plot(origin_plot[690:1890])

    pass
        # torch.save(model, f'./LSTM_{lookback}')

    return train_mse, test_mse, train_mae, test_mae

if __name__ == '__main__':
    args = Namespace( lookafter=96, i=0, model_name='LSTM')
    eval(args)