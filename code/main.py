import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import os
import time
import joblib
import pickle
import argparse
from utils import get_data, plot_multiple_curves
from train_model import CV_train
from models import ModelSelector, HGT



def get_parser():
    parser = argparse.ArgumentParser(description='HG-LGBM')

    parser.add_argument('--dataset', type=str, default="HMDAD", help='Dataset name to use')
    parser.add_argument('--save_file', type=str, default='HMDAD/save_file_HMDAD/', help='Result save path')

    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=512, help='Number of hidden channels')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--self_encode_len', type=int, default=256, help='Self-encoding length')

    # Training parameters
    parser.add_argument('--kfold', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--print_epoch', type=int, default=20, help='Print frequency')
    parser.add_argument('--globel_random', type=int, default=222, help='Global random seed')
    parser.add_argument('--maskMDI', action='store_true', help='Whether to mask MDI')

    return parser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# Configuration class
class Config:
    def __init__(self):
        parser = get_parser()
        args = parser.parse_args()

        self.save_file = args.save_file
        self.kfold = args.kfold
        self.maskMDI = args.maskMDI
        self.hidden_channels = args.hidden_channels
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.self_encode_len = args.self_encode_len
        self.globel_random = args.globel_random
        self.epochs = args.epochs
        self.print_epoch = args.print_epoch
        self.dataset = args.dataset
        self.other_args = {'arg_name': [], 'arg_value': []}



def set_attr(config, param_grid):
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for key in param_keys:
            setattr(config, key, param[key])
            config.other_args['arg_name'].append(key)
            config.other_args['arg_value'].append(param[key])
        yield config


if __name__ == '__main__':

    set_seed(521)

    param_search = {
        'hidden_channels': [64],
        'num_heads': [8],
        'num_layers': [6],
    }

    params_all = Config()
    param_generator = set_attr(params_all, param_search)
    data_list = []

    while True:
        try:
            params = next(param_generator)
        except StopIteration:
            break

        data_tuple = get_data(params=params)
        data_idx, auc_name = CV_train(params, data_tuple)

    for i in range(1, 6):
        kf = i
        file_name = f'data/model_{kf}_fold.dict'

        # Wait for file to exist
        while not os.path.exists(file_name):
            time.sleep(1)

        data_load = joblib.load(file_name)

        # Model selection and training
        selector = ModelSelector()
        X_train, X_test = data_load['train_data'], data_load['test_data']
        y_train, y_test = data_load['y_train'], data_load['y_test']

        models = selector.get_models([])
        ls_dict = selector.train_with_grid_search(
            X_train, np.reshape(y_train, (-1,)), X_test, np.reshape(y_test, (-1,)), models
        )
        data_list.append(ls_dict)

        data_all = data_list if len(data_list) > 1 else data_list[0]
        os.makedirs(params_all.save_file, exist_ok=True)
        save_path = os.path.join(params_all.save_file, '8head_2layer_5cv_data_1000.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(data_all, f)

    selector = ModelSelector()
    avg_performance = selector.calculate_average_performance(data_list)


    if not os.path.exists('results'):
        os.makedirs('results')

    avg_perf_path = 'results/average_performance.pkl'
    with open(avg_perf_path, 'wb') as f:
        pickle.dump(avg_performance, f)
