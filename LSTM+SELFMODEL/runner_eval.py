from argparse import Namespace

import numpy as np

from train import train
from train_LSTMTransformer import LT_train
from eval import eval
if __name__ == '__main__':
    lookback = 96
    lookafter = 240
    # model_path = f'../models/LSTM_96_R[50]_L[2]_size[256].pth'
    model_path = f''
    num_layers = 2
    hidden_size = 256
    n_epochs = 50
    lr = 1e-4
    train_mse_all = []
    test_mse_all = []
    train_mae_all = []
    test_mae_all = []
    test_mse1_all = []
    for i in range(1, 6):
        # args = Namespace(lr=lr, lookback=lookback, lookafter=lookafter, model_path=model_path, num_layers=num_layers,
        #                  hidden_size=hidden_size, n_epochs=n_epochs, i=i)
        # train(args)
        args = Namespace(lr=lr, lookback=lookback, lookafter=240, model_path=model_path, num_layers=num_layers,
                         hidden_size=hidden_size, n_epochs=n_epochs, i=i, model_name='LSTM')

        train_mse, test_mse, train_mae, test_mae = eval(args)

        train_mse_all.append(train_mse)
        test_mse_all.append(test_mse)
        train_mae_all.append(train_mae)
        test_mae_all.append(test_mae)

    print(f'train_mse:{np.mean(train_mse_all)} {np.std(train_mse_all)}, test_mse:{np.mean(test_mse_all)} {np.std(test_mse_all)},'
          f' train_mae:{np.mean(train_mae_all)} {np.std(train_mae_all)}, test_mae:{np.mean(test_mae_all)} {np.std(test_mae_all)}'
          )