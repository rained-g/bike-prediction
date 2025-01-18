import datetime
import re
from argparse import Namespace

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from torch import cuda, optim, nn
from torch.autograd import backward
from torch.utils import data
from tqdm import tqdm

from Module_LSTM import Module_LSTM
from utils import create_dataset, get_dataset
from LSTMTransformer import LSTMTransformer




def LT_train(args):

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



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainX, trainY, testX, testY = get_dataset(args.lookback, args.lookafter, columns=columns,device=device)

    # model = Module_LSTM(input_size=len(columns), output_size=args.lookafter,num_layers=args.num_layers, hidden_size=args.hidden_size).to(device)
    model = LSTMTransformer(input_size=len(columns), hidden_size=args.hidden_size, lstm_layers=args.num_layers, out_feature=args.lookafter).to(device)

    load_model_round = 0
    if len(args.model_path) != 0:
        model = torch.load(args.model_path).to(device)
        match = re.search(r'\[(\d+)\]', args.model_path)
        load_model_round = int(match.group(1))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loss_fn2 = nn.L1Loss()
    loader = data.DataLoader(data.TensorDataset(trainX, trainY), shuffle=True, batch_size=10)
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        train_rmse = 0
        train_mae = 0
        num = 0
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_rmse += np.sqrt(loss_fn(y_pred.detach().cpu(), y_batch.detach().cpu()))
            train_mae += loss_fn2(y_pred.detach().cpu(), y_batch.detach().cpu())
            num += 1

        # print(f"epoch train_rmse:{train_rmse / num} train_mae:{train_mae / num}")

        # Validation
        # if epoch % 10 != 0:
        #     continue
        model.eval()
        with torch.no_grad():
            y_pred = model(trainX)
            y_pred_test = model(testX)
            train_mse = loss_fn(y_pred.cpu(), trainY.cpu())
            train_mae = loss_fn2(y_pred.cpu(), trainY.cpu())
            train_mape = (loss_fn2(y_pred.cpu(), trainY.cpu())) / y_pred.cpu().mean()

            test_mse = loss_fn2(y_pred_test.cpu(), testY.cpu())
            test_mae = loss_fn2(y_pred_test.cpu(), testY.cpu())
            print(type(y_pred.cpu()), y_pred.cpu().mean())
        print("Epoch %d: train MSE %.4f, test MSE %.4f train MAE %.4f test MAE %.4f train MAPE %.4f" %
              (epoch, train_mse, test_mse, train_mae, test_mae, train_mape))
        torch.save(model,
                   f'../models/temp/LSTMTransformer1_{args.lookafter}_R[{load_model_round + args.n_epochs}]_L[{args.num_layers}]_size[{args.hidden_size}]_R[{epoch}].pth')

    torch.save(model, f'../models/{args.i}/LSTMTransformer1_{args.lookafter}_R[{load_model_round + args.n_epochs}]_L[{args.num_layers}]_size[{args.hidden_size}].pth')
if __name__ == '__main__':
    lookback = 96
    lookafter = 240
    # model_path = f'../models/LSTM_96_R[50]_L[2]_size[256].pth'
    model_path = f''
    num_layers = 2
    hidden_size = 256
    n_epochs = 50
    lr = 5e-4

    args = Namespace(lr=lr, lookback=lookback, lookafter=lookafter,model_path=model_path, num_layers=num_layers, hidden_size=hidden_size, n_epochs=n_epochs, i=0)

    LT_train(args)
