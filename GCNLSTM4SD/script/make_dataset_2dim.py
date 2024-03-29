import csv

# 读取原始CSV文件
with open('./data/metr-la/Xinan_PS_Dataset.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = list(csv_reader)

# 创建一个新的CSV文件来保存修改后的数据
with open('./data/metr-la/Xinan_2d.csv', 'w', newline='') as new_csv_file:
    csv_writer = csv.writer(new_csv_file)

    # 写入原始数据集的标题行
    csv_writer.writerow(data[0])

    for row in data[1:]:
        # 写入原始数据行
        csv_writer.writerow(row)

        csv_writer.writerow(row[11:12] * 12)

