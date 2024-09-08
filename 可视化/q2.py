import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 加载数据
file_path = 'result/result_q2/result_q2_m1.csv'
data = pd.read_csv(file_path)

# 定义要观察的变量
variables = [('x3', 'x4'), ('x1', 'x2')]

# 创建 images 文件夹（如果不存在）
output_dir = 'image/q2可视化'
os.makedirs(output_dir, exist_ok=True)

for var1, var2 in variables:
    # 定义条件列表
    conditions = [
        (data[var1] == 0) & (data[var2] == 0),
        (data[var1] == 0) & (data[var2] == 1),
        (data[var1] == 1) & (data[var2] == 0),
        (data[var1] == 1) & (data[var2] == 1)
    ]

    # 定义选择列表
    choices = [f'{var1}=0, {var2}=0', f'{var1}=0, {var2}=1', f'{var1}=1, {var2}=0', f'{var1}=1, {var2}=1']

    # 使用 numpy 的 select 函数来创建一个新的分类标签
    data[f'condition_{var1}_{var2}'] = np.select(conditions, choices, default='Other')

    # 绘制箱形图并保存
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=f'condition_{var1}_{var2}', y='profit', data=data)
    plt.title(f'Profit Distribution by Conditions ({var1}, {var2}) (Box Plot)')
    plt.xlabel(f'Conditions ({var1}, {var2})')
    plt.ylabel('Profit')
    box_plot_file = os.path.join(output_dir, f'box_plot_{var1}_{var2}.png')
    plt.savefig(box_plot_file)
    plt.close()

    # 绘制小提琴图并保存
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=f'condition_{var1}_{var2}', y='profit', data=data)
    plt.title(f'Profit Distribution by Conditions ({var1}, {var2}) (Violin Plot)')
    plt.xlabel(f'Conditions ({var1}, {var2})')
    plt.ylabel('Profit')
    violin_plot_file = os.path.join(output_dir, f'violin_plot_{var1}_{var2}.png')
    plt.savefig(violin_plot_file)
    plt.close()


# 定义要观察的变量
variable = 'x5'

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
sns.boxplot(x=f'condition_{variable}', y='profit', data=data)
plt.title(f'Profit Distribution by Conditions ({variable}) (Box Plot)')
plt.xlabel(f'Conditions ({variable})')
plt.ylabel('Profit')
box_plot_file = os.path.join(output_dir, f'box_plot_{variable}.png')
plt.savefig(box_plot_file)
plt.close()

# 绘制小提琴图并保存
plt.figure(figsize=(10, 6))
sns.violinplot(x=f'condition_{variable}', y='profit', data=data)
plt.title(f'Profit Distribution by Conditions ({variable}) (Violin Plot)')
plt.xlabel(f'Conditions ({variable})')
plt.ylabel('Profit')
violin_plot_file = os.path.join(output_dir, f'violin_plot_{variable}.png')
plt.savefig(violin_plot_file)
plt.close()
# 准备数据用于热力图
pivot_table = data.pivot_table(index=['x1', 'x2', 'x3'], columns=['x4', 'x5'], values='profit')
# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Profit Heatmap by Configuration')
plt.ylabel('Configurations (x1, x2, x3)')
plt.xlabel('Configurations (x4, x5)')
plt.tight_layout()

# 保存图像到文件
output_path = os.path.join(output_dir, 'profit_heatmap.png')
plt.savefig(output_path)
plt.show()

print(f"Charts saved to {output_dir}")