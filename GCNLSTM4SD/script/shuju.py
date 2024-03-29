import csv

# 打开 txt 文件
with open("data_true.txt", "r") as f:
    # 读取所有数据
    data = f.readlines()

# 创建 csv 文件
with open("output_true.csv", "w", newline="") as f:
    # 创建 csv 写入器
    writer = csv.writer(f)

    # 遍历所有数据
    for line in data:
        # 将每行数据按空格分割
        items = line.split()

        # 将每个数据写入 csv 文件
        for item in items:
            writer.writerow([item])