import os
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
import torch
import matplotlib.pyplot as plt
import csv


def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    # adj = 0.5 * (dir_adj + dir_adj.transpose())

    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id

    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
            or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso


def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)  # my:gso[8, 8]
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')  # my:id[8, 8]
    eigval_max = max(eigsh(A=gso, k=6, which='LM', return_eigenvectors=False))

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso


def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device,
                                       requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y[:, -1].unsqueeze(-1))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n

        return mse


def evaluate_metric(model, data_iter, scaler, classification):
    model.eval()
    with torch.no_grad():
        # mae, sum_y, mape, mse = [], [], [], []
        Y, Y_pre = [], []
        Y_ture, Y_pre_ture = [], []
        MAPE_LEN = 0
        # MAE, RMSE, NSE = 0., 0., 0., 0.
        MAPE = 0.

        # Open the CSV file in write mode
        with open('predictions.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            for y_ture, y_pred in zip(Y, Y_pre):
                # Write the true and predicted values to the CSV file
                writer.writerow([y_ture, y_pred])

            for x, y in data_iter:
                # 查看原始数据（归一化之前）
                y_ture = scaler.inverse_transform(y.cpu().numpy())[:, -1]
                y_pred_ture = scaler.inverse_transform(model(x).view(len(x), -1).expand(-1, 8).cpu().numpy())[:, -1]
                # y_pred_ture = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy())[:, -1]

                # Write the true and predicted values to the CSV file
                writer.writerow([y_ture, y_pred_ture])

                Y_ture.extend(y_ture)
                Y_pre_ture.extend(y_pred_ture)

                y_1 = y.cpu().numpy()[:, -1]
                a = model(x)
                y_pred_1 = model(x).view(len(x), -1).cpu().numpy()[:, -1]
                Y.extend(y_1)
                Y_pre.extend(y_pred_1)
        print(Y[-1],Y_pre[-1],Y_ture[-1],Y_pre_ture[-1])
  # + Y_pre_ture
        # print(Y_ture[-1])
        # print(Y_pre_ture[-1])
        # N = len(Y_ture)   # Number of data points
        #
        # print(N)
        #
        # plt.figure()
        # plt.plot(range(1, N + 1), Y_ture, label='True Values')
        # plt.plot(range(1, N + 1), Y_pre_ture, label='Predicted Values')
        # plt.xlabel('Data Points')
        # plt.ylabel('Values')
        # plt.legend()
        # plt.title('Testing')
        # plt.grid(False)
        #
        # # plt.show()
        #
        # x = [0, 1]
        # y = [0, 1]
        #
        # plt.figure()
        # # plt.plot(x, y, marker=' ', linestyle='-')
        # plt.scatter(Y_ture, Y_pre_ture, )
        #
        # # 获取当前图的坐标轴范围
        # x_min, x_max = plt.xlim()
        # y_min, y_max = plt.ylim()
        #
        # # 计算直线的坐标点
        # line_x = np.array([x_min, x_max])
        # line_y = line_x  # 斜率为1
        #
        # # 绘制斜率为1的直线
        # plt.plot(line_x, line_y, color='black', linestyle='--', label='True Line')
        #
        # plt.xlabel('True Values')
        # plt.ylabel('Predicted Values')
        # plt.legend()
        # plt.title(f'True vs. Predicted Values')
        #
        # # 创建原始数据和模型预测的散点图
        # # plt.figure()
        # # plt.scatter(Y, Y_pre, label='True vs. Predicted (After Scaling)')
        # # plt.xlabel('True Values')
        # # plt.ylabel('Predicted Values')
        # # plt.legend()
        # # plt.title(f'True vs. Predicted Values (After Scaling)')
        # plt.show()

        # 查看原始数据（归一化之前）
        Y_ture_arr = np.array(Y_ture)
        Y_pre_ture_arr = np.array(Y_pre_ture)

        # 将结果保存到文件
        if classification == 'train':
            save_array_to_file('C:\\Users\\Lou1sLou\\Desktop\\STGCN_With_NodeWeight\\data_save\\train_gt{}_{}_{}.txt'.format(datetime.now().day, datetime.now().hour,
                                                                        datetime.now().minute), Y_ture_arr)
            save_array_to_file('C:\\Users\\Lou1sLou\\Desktop\\STGCN_With_NodeWeight\\data_save\\train_pre{}_{}_{}.txt'.format(datetime.now().day, datetime.now().hour,
                                                                         datetime.now().minute), Y_pre_ture_arr)
        else:
            save_array_to_file('C:\\Users\\Lou1sLou\\Desktop\\STGCN_With_NodeWeight\\data_save\\test_gt{}_{}_{}.txt'.format(datetime.now().day, datetime.now().hour,
                                                                       datetime.now().minute), Y_ture_arr)
            save_array_to_file('C:\\Users\\Lou1sLou\\Desktop\\STGCN_With_NodeWeight\\data_save\\test_pre{}_{}_{}.txt'.format(datetime.now().day, datetime.now().hour,
                                                                        datetime.now().minute), Y_pre_ture_arr)

        # Y = np.array(Y)
        # Y_pre = np.array(Y_pre)
        LEN = len(Y_ture_arr)
        MAE = mae(Y_ture_arr, Y_pre_ture_arr)
        RMSE = rmse(Y_ture_arr, Y_pre_ture_arr)
        NSE = nse(Y_ture_arr, Y_pre_ture_arr)
        for i in range(LEN):
            if Y[i] == 0:
                continue
            else:
                MAPE += mape(Y_ture_arr[i], Y_pre_ture_arr[i])
                MAPE_LEN += 1

        # y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
        # y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)

        # d = np.abs(y - y_pred)
        # mae += d.tolist()
        # sum_y += y.tolist()
        # mape += (d / y).tolist()
        # mse += (d ** 2).tolist()

        return MAE, RMSE, MAPE / MAPE_LEN, NSE


def nse(y_tru, y_pre):
    if type(y_tru) is torch.Tensor and type(y_pre) is torch.Tensor:
        result = 1 - torch.div(torch.sum(torch.square(y_tru - y_pre), dim=-1),
                               torch.sum(torch.square(y_tru - torch.mean(y_tru, dim=-1, keepdim=True))))
        return result
    elif type(y_tru) is np.ndarray and type(y_pre) is np.ndarray:
        result = 1 - np.divide(np.sum(np.square(y_tru - y_pre), axis=-1),
                               np.sum(np.square(y_tru - np.mean(y_tru, axis=-1))))
        return result
    else:
        return -1


def rmse(y_tru, y_pre):
    if type(y_tru) is torch.Tensor and type(y_pre) is torch.Tensor:
        return torch.sqrt(torch.mean(torch.square(y_tru - y_pre), dim=-1))
    elif type(y_tru) is np.ndarray and type(y_pre) is np.ndarray:
        return np.sqrt(np.mean(np.square(y_tru - y_pre), axis=-1))
    else:
        return -1


def mae(y_tru, y_pre):
    if type(y_tru) is torch.Tensor and type(y_pre) is torch.Tensor:
        return torch.mean(torch.abs(y_tru - y_pre), dim=-1)
    elif type(y_tru) is np.ndarray and type(y_pre) is np.ndarray:
        return np.mean(np.abs(y_tru - y_pre), axis=-1)
    else:
        return -1


# def mape(y_tru, y_pre):
#     if type(y_tru) is torch.Tensor and type(y_pre) is torch.Tensor:
#         return torch.mean(torch.abs(torch.div((y_tru - y_pre), y_tru)), dim=-1) * 100
#     elif type(y_tru) is np.ndarray and type(y_pre) is np.ndarray:
#         y_tru[y_tru() == 0] = 1e-8
#         return np.mean(np.abs(np.divide((y_tru - y_pre), y_tru)), axis=-1) * 100
#     else:
#         return -1
#
# import torch

def mape(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    epsilon = 1e-8
    # mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    mape = (torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape.item()


# def mape(y_true, y_pred, epsilon=1e-7):
#     y_true = torch.from_numpy(y_true)
#     y_pred = torch.from_numpy(y_pred)
#     mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
#     return mape.item()

def save_array_to_file(file_path, array):
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    with open(os.path.join("data_save", file_path), "w") as file:
        for item in array:
            file.write(str(item) + '\n')
