import pandas as pd

# 读取CSV文件，将第一行作为标题
file_path = 'data/metr-la/adj_m.csv'  # 替换成你的文件路径
data = pd.read_csv(file_path, header=0)

# 提取数据部分（排除标题行）
numeric_data = data.iloc[:, 1:]  # 假设数据从第二列开始，如果不是，请调整索引

# 将字符串类型转换为数值类型（假设数据中有非数值字符）
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# 取非0数的倒数
inverse_nonzero = numeric_data.applymap(lambda x: 1/x if x != 0 else 0)

# 按列进行归一化操作
normalized_data = (inverse_nonzero - inverse_nonzero.min()) / (inverse_nonzero.max() - inverse_nonzero.min())

# 打印归一化后的数据
normalized_data.to_csv('normalized_data.csv', index=False)

