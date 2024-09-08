import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# 加载数据
file_path = 'result/result_q4_2/result_m1.csv'
data = pd.read_csv(file_path)

# 准备数据用于热力图
pivot_table = data.pivot_table(index=['x1', 'x2', 'x3'], columns=['x4', 'x5'], values='profit')

# 创建 images 文件夹（如果不存在）
output_dir = 'image/q4可视化'
os.makedirs(output_dir, exist_ok=True)

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