import torch
import torch.nn as nn
import numpy as np
import random
import sys
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class H_STGCN(torch.nn.Module):
    def __init__(self, args, n_vertex, edge_index, edge_attr, num_nodes=15, num_features=2, nhid=64, num_classes=1,
                 dropout_ratio=0.3, window=16, real_time=False):
        super(H_STGCN, self).__init__()

        self.num_nodes = n_vertex
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.window = args.n_his
        self.num_features = num_features
        self.nhid = nhid
        self.dropout = dropout_ratio
        self.real_time = real_time
        self.num_classes = num_classes
        self.GCN_num = 2
        self.conv_1 = nn.Conv2d(self.num_features, self.nhid, (1, 1))
        # self.conv_2 = nn.Conv2d(self.num_features, self.nhid, (2, 2)) L

        self.conv1 = GCNConv(self.nhid, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self._batch_norm_1 = nn.BatchNorm1d(self.nhid)
        self._batch_norm_2 = nn.BatchNorm1d(self.nhid)

        self.dec_drop = nn.Dropout(self.dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_nodes * self.nhid, self.nhid),
            nn.ReLU()
        )

        self.LSTM = nn.LSTM(input_size=self.nhid,
                            hidden_size=40,
                            num_layers=1,
                            batch_first=False,
                            bidirectional=True,)

        self.classifier = nn.Sequential(
            nn.Linear(80, 16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(16, self.num_classes)
        )

    def _graph_convolution_1(self, X, edge_index, edge_weight):
        # X[256,32] edge_index[56,2] edge_weight[56,]
        X = F.relu(self.conv1(X, edge_index, edge_weight))
        X = self._batch_norm_1(X)
        # X = self.dec_drop(X)
        return X

    def _graph_convolution_2(self, X, edge_index, edge_weight):
        X = F.relu(self.conv2(X, edge_index, edge_weight))
        X = self._batch_norm_2(X)
        # X = self.dec_drop(X)
        return X

    # def _graph_convolution_3(self, X, edge_index, edge_weight):
    #     X = F.relu(self.conv2(X, edge_index, edge_weight))
    #     X = self._batch_norm_2(X)
    #     # X = self.dec_drop(X)
    #     return X

    def Hierarchical_GCN(self, x, edge_index, edge_weight=None, batch=None):

        x1 = self._graph_convolution_1(x, edge_index, edge_weight) + x
        x2 = self._graph_convolution_2(x1, edge_index, edge_weight) + x1
        # x3 = self._graph_convolution_3(x1, edge_index, edge_weight) + x2

        return x2  # (batch * 8) * (nhid*3)

    def forward(self, data):

        x, edge_index, edge_attr = data, self.edge_index, self.edge_attr

        # print("xsize: ", x.size())

        x = x.reshape(-1, self.num_nodes, x.size(1), x.size(2)).permute(0, 2, 1, 3)  # [256, 1, 8, 16]  our[256, 1, 15, 16]  st[32,1,8,12]
        # batchsize256,node15,window16

        # x_shape: [-1, 1, 5, 8]
        # x = x.reshape(-1, 1, 8, 5)

        x = self.conv_1(x)  # [80,1,8,5]->[80,num_features,8,5]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, x.size(2), x.size(3))  # our[340,16,32] st[256,12,32]

        OUT = []
        for l in range(x.size(1)):
            if edge_attr == None:
                H = self.Hierarchical_GCN(x[:, l, :], edge_index, None)
            else:
                H = self.Hierarchical_GCN(x[:, l, :], edge_index, edge_attr.flatten())

            OUT.append(H)

        x_concat = torch.stack(OUT, dim=0)  # time * (batch * 8) * (nhid * 3)

        X = x_concat.reshape(self.window, -1, self.num_nodes * self.nhid)  # time * batch * (8 * nhid *3)

        X = self.fc1(X)  # [16, 256, 32]  batchsize=256   node=16   n.hid=32

        r_out, (h_n, h_c) = self.LSTM(X, None)

        x_step = r_out[-1, :, :]

        dec_score = self.classifier(x_step)

        return dec_score