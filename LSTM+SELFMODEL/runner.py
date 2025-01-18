from argparse import Namespace
from train import train
from train_LSTMTransformer import LT_train

if __name__ == '__main__':
    lookback = 96
    lookafter = 240
    # model_path = f'../models/LSTM_96_R[50]_L[2]_size[256].pth'
    model_path = f''
    num_layers = 2
    hidden_size = 256
    n_epochs = 50
    lr = 1e-4

    for i in range(1, 6):
        # args = Namespace(lr=lr, lookback=lookback, lookafter=lookafter, model_path=model_path, num_layers=num_layers,
        #                  hidden_size=hidden_size, n_epochs=n_epochs, i=i)
        # train(args)
        args = Namespace(lr=lr, lookback=lookback, lookafter=96, model_path=model_path, num_layers=num_layers,
                         hidden_size=hidden_size, n_epochs=n_epochs, i=i)
        LT_train(args)

    for i in range(1, 6):
        # args = Namespace(lr=lr, lookback=lookback, lookafter=lookafter, model_path=model_path, num_layers=num_layers,
        #                  hidden_size=hidden_size, n_epochs=n_epochs, i=i)
        # train(args)
        args = Namespace(lr=lr, lookback=lookback, lookafter=240, model_path=model_path, num_layers=num_layers,
                         hidden_size=hidden_size, n_epochs=n_epochs, i=i)
        LT_train(args)