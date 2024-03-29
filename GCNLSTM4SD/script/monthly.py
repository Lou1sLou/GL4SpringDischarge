import csv

in_file = 'data/water flow(原版).csv'
out_file = 'data/dataset1.csv'

selected_lines = [i for i in range(0, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集1.csv已保存选取的行')


out_file = 'data/dataset2.csv'

selected_lines = [i for i in range(1, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集2.csv已保存选取的行')


out_file = 'data/dataset3.csv'

selected_lines = [i for i in range(2, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集3.csv已保存选取的行')


out_file = 'data/dataset4.csv'

selected_lines = [i for i in range(3, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集4.csv已保存选取的行')


out_file = 'data/dataset5.csv'

selected_lines = [i for i in range(4, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集5.csv已保存选取的行')


out_file = 'data/dataset6.csv'

selected_lines = [i for i in range(5, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集6.csv已保存选取的行')


out_file = 'data/dataset7.csv'

selected_lines = [i for i in range(6, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集7.csv已保存选取的行')


out_file = 'data/dataset8.csv'

selected_lines = [i for i in range(7, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集8.csv已保存选取的行')


out_file = 'data/dataset9.csv'

selected_lines = [i for i in range(8, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集9.csv已保存选取的行')



out_file = 'data/dataset10.csv'

selected_lines = [i for i in range(9, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集10.csv已保存选取的行')


out_file = 'data/dataset11.csv'

selected_lines = [i for i in range(10, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集11.csv已保存选取的行')


out_file = 'data/dataset12.csv'

selected_lines = [i for i in range(11, 720, 12)]

with open(in_file, 'r') as f_input, open(out_file, 'w', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    for i, row in enumerate(reader):
        if i in selected_lines:
            writer.writerow(row)

print('数据集12.csv已保存选取的行')