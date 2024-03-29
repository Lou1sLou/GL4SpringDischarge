import numpy as np
import pandas as pd
import copy

# 读取原始距离矩阵
dist_mat = pd.read_csv('data/metr-la/adj_m.csv', index_col=None)

# 加一个很小的数,避免除零错误
eps = 1e-8
dist_mat += eps

# 归一化到0-1
dist_mat = dist_mat / np.max(dist_mat)

# 取负对数得到亲和度矩阵
aff_mat = -np.log(dist_mat)

# 标准化到0均值,1方差
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
aff_mat = scaler.fit_transform(aff_mat)

aff_mat_T = copy.deepcopy(aff_mat)
aff_mat_T = aff_mat_T.T

# 对称化
aff_mat = (aff_mat + aff_mat_T) / 2

# 添加自连接
aff_mat += np.identity(aff_mat.shape[0])

# 写出预处理后的邻接矩阵
pd.DataFrame(aff_mat).to_csv('data/metr-la/adj_mmm.csv')
