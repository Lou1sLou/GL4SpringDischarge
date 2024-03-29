import numpy as np
import scipy.sparse as sp
import csv

from torch_geometric.nn import GCN

adj = np.array([[1, 110, 0, 0, 0, 0, 0, 405],
                [110, 1, 0, 0, 220, 0, 0, 370],
                [0, 0, 1, 0, 0, 0, 0, 590],
                [455, 540, 350, 1, 0, 0, 0, 840],
                [0, 220, 0, 0, 1, 0, 0, 470],
                [0, 0, 0, 0, 385, 1, 0, 850],
                [0, 0, 0, 0, 0, 385, 1, 1230],
                [0, 0, 0, 0, 0, 0, 0, 1]])


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # 给A加上一个单位矩阵
    return sparse_to_tuple(adj_normalized)


if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # 采用三元组(row, col, data)的形式存储稀疏邻接矩阵
    rowsum = np.array(adj.sum(1))  # 按行求和得到rowsum, 即每个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # (行和rowsum)^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized  # L=I-D^{-1/2} A D^{-1/2}
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')  # 得到laplacian矩阵的最大特征值
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])  # 对laplacian矩阵进行scale处理

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    return sparse_to_tuple(t_k)
