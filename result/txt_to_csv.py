import csv
import re  # 导入正则表达式模块
# 读取文本文件
input_file = 'result/Q3.txt'
output_file = 'result/Q3.csv'

# 正则表达式用于提取数据
pattern = r'X=\((.*?)\), Y=\((.*?)\), Z=\((.*?)\), P=(\d+), Q=(\d+), R=(\d+), profit: (\d+)'

# 打开输出文件并写入数据
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # 写入表头
    header = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y1', 'y2', 'y3', 'z1', 'z2', 'z3', 'P', 'Q', 'R', 'Profit']
    writer.writerow(header)

    # 逐行读取输入文件
    for line in infile:
        # 使用正则表达式提取数据
        match = re.search(pattern, line)
        if match:
            X = [int(x) for x in match.group(1).split(', ')]
            Y = [int(y) for y in match.group(2).split(', ')]
            Z = [int(z) for z in match.group(3).split(', ')]
            P = int(match.group(4))
            Q = int(match.group(5))
            R = int(match.group(6))
            profit = int(match.group(7))

            # 将数据写入csv
            row = X + Y + Z + [P, Q, R, profit]
            writer.writerow(row)