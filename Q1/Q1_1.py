import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
# 生成高斯分布随机数函数
def generate_binomial_random(n, p, size):
    np.random.seed(0)  # 设置随机种子
    random_values = np.random.binomial(n, p, size)  # 生成二项分布的随机数
    return random_values.astype(int)  # 转换为整数类型

# 生成n个随机数据，这些数据满足二项分布，n为m，p为0.1
def generate_random_data(m, n):
    return generate_binomial_random(m, 0.1, n)

def T(m,n,mean,std):
    return (mean - 0.1*m) / (std / np.sqrt(n))

# 用一个数据框记录m和n的值与它们的积
result = pd.DataFrame(columns=['m', 'n', 'm*n'])

for m in range(1, 20):
    for n in range(10, 101):
        print('n=', n)
        data = generate_random_data(m, n)
        # print('data=', data)
        mean = np.mean(data)
        # 修正样本均方差
        std = np.std(data, ddof=1)
        print('T=', T(m,n,mean,std), 't.ppf=', t.ppf(0.95, n-1))
        if T(m,n,mean,std) > t.ppf(0.95, n-1):
            print(f"m={m}, n={n}, m*n={m*n}")
            new_row = {'m': m, 'n': n, 'm*n': m*n}
            #将新的行数据转换为Series
            new_series = pd.Series(new_row)
            result = pd.concat([result, new_series.to_frame().T], ignore_index=True)
            break
        
# 打印出最小的m*n值
print('最小的m*n值为：')
min_value = result.loc[result['m*n'].idxmin()] 
print(min_value)

# 画出所有m*n与m的关系
plt.figure(figsize=(10, 6))  # 设置图片大小
plt.plot(result['m'], result['m*n'], label='m*n 值', color='b', marker='o')  # 添加线和标记

# 标注最小值
plt.scatter(min_value['m'], min_value['m*n'], color='r')  # 标注最小值点
plt.text(min_value['m'], min_value['m*n'], f'Min: ({min_value["m"]}, {min_value["m*n"]})', 
         fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='r')

# 美化图表
plt.title('m 与 m*n 的关系', fontsize=16)  # 添加标题
plt.xlabel('m 值', fontsize=14)  # x轴标签
plt.ylabel('m*n 值', fontsize=14)  # y轴标签
plt.grid(True)  # 添加网格
plt.legend()  # 添加图例

# 保存为PDF矢量图格式
plt.savefig('image/Q1_1.pdf', format='pdf')  # 保存为PDF格式矢量图
plt.show()  # 显示图表


