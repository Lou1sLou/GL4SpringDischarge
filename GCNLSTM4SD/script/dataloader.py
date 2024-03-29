import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_adj(dataset_name):
    df = pd.read_csv('data/nzg_sd/daoshuam.csv')
    adj = np.array(df)[:, :]
    adj = sp.csc_matrix(adj)
    n_vertex = 8

    # 打印稀疏矩阵
    return adj, n_vertex


def load_data_spring_1(dataset_name, len_train):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'wf.csv'))
    train = vel[: len_train]  # my:train[514,8]
    test = vel[len_train:]  # my:[109,8]
    return train, test


def load_data_spring_2(dataset_name, len_train):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'newds8.csv'))
    train = vel[: len_train]  # my:train[514,8]
    test = vel[len_train:]  # my:[109,8]
    return train, test


def load_data_spring_3(dataset_name, len_train):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'wa.csv'))
    train = vel[: len_train]  # my:train[514,8]
    test = vel[len_train:]  # my:[109,8]
    return train, test


# 完整数据集
def load_data_f(dataset_name, len_train):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'water flow.csv'))
    # vel = vel.iloc[:, :7]
    train = vel[: len_train]  # my:train[514,8]
    test = vel[len_train:]  # my:[109,8]
    return train, test


def load_data_spring_4(dataset_name, len_train):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    train = vel[: len_train]  # my:train[514,8]
    test = vel[len_train:]  # my:[109,8]
    return train, test


def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]  # my:8
    len_record = len(data)  # my:514
    num = len_record - n_his - n_pred  # my:499 ,n_his=12,n_pred=3

    x = np.zeros([num, 2, n_his, n_vertex])  # my:[499,1,12,8]
    y = np.zeros([num, n_vertex])  # my:[499,8]

    for i in range(num):
        head = i
        tail = i + n_his

        # 8 vertexes 2 dim
        for a in range(2):
            if a == 0:
                x[i, a, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
            else:
                ax = data[head: tail][:, -1]
                ax = np.tile(ax, (8, 1)).transpose(1, 0)
                x[i, a, :, :] = ax.reshape(1, n_his, n_vertex)
            # my:i=0, x.shape=[499,1,12,8]  [num,1,n_his,n_vertex]
        # if i % 2 == 0:
        #     x[i / 2, 0, :, :] = data[head: tail: 2].reshape(1, n_his, n_vertex)
        # else:
        #     x[i / 2, 1, :, :] = data[head + 1: tail: 2].reshape(1, n_his, n_vertex)

        y[i] = data[tail + n_pred - 1]  # my:i=0, y.shape=[499,8],n_his=12,n_pred=3

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
