import argparse
import logging
import os
import random

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import tqdm
from sklearn import preprocessing

from model import new_model
from script import dataloader, utility, earlystopping


def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=87, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=9)
    parser.add_argument('--n_pred', type=int, default=1,
                        help='the number of time interval for prediction, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=12)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='H_STGCN')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.3)  # useless
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=369)
    parser.add_argument('--epochs', type=int, default=329, help='epochs, default as 10000')  # p:default 366
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=25)  # 24
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')

    args = parser.parse_args()

    print('Training configs: {}'.format(args))

    set_env(args.seed)

    device = torch.device('cpu')

    return args, device


def data_prepare(args, device):
    adj, n_vertex = dataloader.load_adj(args.dataset)  # my:adj[8, 8], n_vertex=8

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'water flow.csv')).shape[0]
    train_num = int(data_col * 37 / 61)
    # train_num = 650

    len_test = int(math.floor(data_col - train_num))

    len_train = int(train_num)

    train, test = dataloader.load_data_f(args.dataset, len_train, )  # my:train, val, test =[514,8], [109,8],[109,8]
    # 数据标准化处理
    A = train
    # zscore = preprocessing.StandardScaler()
    # zscore = preprocessing.Normalizer()
    zscore = preprocessing.MinMaxScaler()
    train = zscore.fit_transform(train)
    test = zscore.transform(test)

    B = train

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred,
                                                 device)  # my:x_train[]499,1,12,8], y_train[499,8]
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred,
                                               device)  # my: x_test[94,1,12,8], y_test[94,8]

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, test_iter


def prepare_model(args, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'H_STGCN':
        edge_index, edge_attr = get_edge_edgeweight()
        # 将张量放到GPU上
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        model = new_model.H_STGCN(args, n_vertex, edge_index, edge_attr).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(loss, args, optimizer, scheduler, es, model, train_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):  # batch_size=32
            xarr = x.cpu().numpy()
            yarr = y.cpu().numpy()
            # print(x.shape) x[32, 1, 12, 8]  y[32,8]
            model_x = model(x)
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            ypre_arr = y_pred.detach().cpu().numpy()
            l = loss(y_pred, y[:, -1].unsqueeze(-1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_sum / n, gpu_mem_alloc))

    model.eval()
    classification = 'train'

    train_MSE = utility.evaluate_model(model, loss, train_iter)
    train_MAE, train_RMSE, train_MAPE, train_NSE = utility.evaluate_metric(model, train_iter, zscore, classification)
    print(
        f'Dataset {args.dataset:s} | Train loss {train_MSE:.6f} | MAE {train_MAE:.6f} | RMSE {train_RMSE:.6f} | MAPE {train_MAPE:.8f}| NSE {train_NSE:.8f}')


@torch.no_grad()
def test(zscore, loss, model, test_iter, args):
    model.eval()
    classification = 'test'
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_MAPE, test_NSE = utility.evaluate_metric(model, test_iter, zscore, classification)
    print(
        f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | '
        f'RMSE {test_RMSE:.6f} | MAPE {test_MAPE:.8f}| NSE {test_NSE:.8f}')


def get_edge_edgeweight():
    data_list = []
    df = pd.read_csv('./data/nzg_sd/daoshuam.csv')

    edge = []
    edge_weight = []
    A = np.array(df)[:, 1:]

    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            edge.append([r, c])
            edge_weight.append([abs(A[r, c])])
            # edge_weight.append([A[r,c]])
    # print(edge)
    # print(edge_weight)
    # 边索引矩阵
    edge_index = torch.tensor(np.array(edge), dtype=torch.long)
    edge_index = edge_index.permute(1, 0)
    # 边属性矩阵
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)
    return edge_index, edge_attr


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args, device = get_parameters()
    n_vertex, zscore, train_iter, test_iter = data_prepare(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, n_vertex)
    train(loss, args, optimizer, scheduler, es, model, train_iter)
    test(zscore, loss, model, test_iter, args)
