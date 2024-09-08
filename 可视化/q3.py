import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
file_path = 'result/Q3.csv'
data = pd.read_csv(file_path)

# 定义要观察的变量组
variable_groups = [('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'),  # x1 到 x8 作为一组
                   ('y1', 'y2', 'y3'),  # y1 到 y3 作为一组
                   ('z1', 'z2', 'z3'),  # z1 到 z3 作为一组
                   ('P', 'Q')]  # P 和 Q 作为一组
individual_variables = ['R']  # R 作为单独变量

# 创建 images 文件夹（如果不存在）
output_dir = 'image/q3可视化'
os.makedirs(output_dir, exist_ok=True)

# 针对变量组绘制箱形图和小提琴图
for group in variable_groups:
    # 定义变量名
    group_name = '_'.join(group)
    
    # 定义条件列表（所有变量为0，所有变量为1，或者混合情况）
    conditions = [
        (data[list(group)] == 0).all(axis=1),
        (data[list(group)] == 1).all(axis=1),
        (data[list(group)] != data[list(group)].iloc[0]).any(axis=1)
    ]

    # 定义选择列表
    choices = [f'All {group_name}=0', f'All {group_name}=1', f'Mixed {group_name}']

    # 使用 numpy 的 select 函数来创建一个新的分类标签
    data[f'condition_{group_name}'] = np.select(conditions, choices, default='Other')

    # 绘制箱形图并保存
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=f'condition_{group_name}', y='Profit', data=data)
    plt.title(f'Profit Distribution by Conditions ({group_name}) (Box Plot)')
    plt.xlabel(f'Conditions ({group_name})')
    plt.ylabel('Profit')
    box_plot_file = os.path.join(output_dir, f'box_plot_{group_name}.png')
    plt.savefig(box_plot_file)
    plt.close()

    # 绘制小提琴图并保存
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=f'condition_{group_name}', y='Profit', data=data)
    plt.title(f'Profit Distribution by Conditions ({group_name}) (Violin Plot)')
    plt.xlabel(f'Conditions ({group_name})')
    plt.ylabel('Profit')
    violin_plot_file = os.path.join(output_dir, f'violin_plot_{group_name}.png')
    plt.savefig(violin_plot_file)
    plt.close()

# 针对单个变量绘制箱形图和小提琴图
for variable in individual_variables:
    # 定义条件列表
    conditions = [
        (data[variable] == 0),
        (data[variable] == 1)
    ]

    # 定义选择列表
    choices = [f'{variable}=0', f'{variable}=1']

    # 使用 numpy 的 select 函数来创建一个新的分类标签
    data[f'condition_{variable}'] = np.select(conditions, choices, default='Other')

    # 绘制箱形图并保存
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=f'condition_{variable}', y='Profit', data=data)
    plt.title(f'Profit Distribution by Conditions ({variable}) (Box Plot)')
    plt.xlabel(f'Conditions ({variable})')
    plt.ylabel('Profit')
    box_plot_file = os.path.join(output_dir, f'box_plot_{variable}.png')
    plt.savefig(box_plot_file)
    plt.close()

    # 绘制小提琴图并保存
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=f'condition_{variable}', y='Profit', data=data)
    plt.title(f'Profit Distribution by Conditions ({variable}) (Violin Plot)')
    plt.xlabel(f'Conditions ({variable})')
    plt.ylabel('Profit')
    violin_plot_file = os.path.join(output_dir, f'violin_plot_{variable}.png')
    plt.savefig(violin_plot_file)
    plt.close()

print(f"Charts saved to {output_dir}")