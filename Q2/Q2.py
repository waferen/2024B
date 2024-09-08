import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题

# 定义零配件1的购买及检测成本
# 两个参数，x1为0-1变量，表示是否检测，n表示购买数量，m表示第m种情况
def cost_spare1(x1,n,m):
    # cost = data['零配件 1 购买单价'][m] * n+ x1*(data['零配件 1 检测成本'][m] * n+data['零配件 1 次品率'][m] * n *data['零配件 1 购买单价'][m])
    cost = data['零配件 1 购买单价'][m] * n+ x1*(data['零配件 1 检测成本'][m] * n)
    num_of_spare1 = n-x1*int(n*data['零配件 1 次品率'][m])
    return cost,num_of_spare1

# 定义零配件2的购买及检测成本
def cost_spare2(x2,n,m):
    cost = data['零配件 2 购买单价'][m] * n+ x2*(data['零配件 2 检测成本'][m] * n)
    num_of_spare2 = n-x2*n*data['零配件 2 次品率'][m]
    return cost,num_of_spare2


def Num_of_reject(x1,x2,num_of_finished_products,m):
       # 如果x1=1,x2=1,则不合格的只由成品组装导致
    if x1==1 and x2==1:
        num_of_reject_final=int(num_of_finished_products*data['成品次品率'][m])# 次品个数
    # 如果x1=1,x2=0,则需要先确认零配件2的次品个数，再在剩下的正品零件中再减去装配产生的次品
    if x1==1 and x2==0 :
        num_of_reject_2 = int(num_of_finished_products*data['零配件 2 次品率'][m])
        num_of_reject_final = int((num_of_finished_products-num_of_reject_2)*data['成品次品率'][m])+num_of_reject_2
    # 如果x1=0,x2=1,则需要先确认零配件1的次品个数，再在剩下的正品零件中再减去装配产生的次品
    if x1==0 and x2==1:
        num_of_reject_1 = int(num_of_finished_products*data['零配件 1 次品率'][m])
        num_of_reject_final = int((num_of_finished_products-num_of_reject_1)*data['成品次品率'][m])+num_of_reject_1 
    # 如果x1=0,x2=0,由于两个都会产生次品，只有(1-零件1次品率)*(1-零件2次品率)*num_of_finished_products的零件组是正品,其余都是次品。在正品零件组中减去装配产生的次品
    if x1==0 and x2==0:
        num_of_qualify = int((1-data['零配件 1 次品率'][m])*(1-data['零配件 2 次品率'][m])*(1-data['成品次品率'][m])*num_of_finished_products)# 正品数量
        num_of_reject_final = int(num_of_finished_products-num_of_qualify)
    return num_of_reject_final



# 定义成品装配及检测拆解的成本,n1，n2分别表示零配件1和零配件2进入装配环节的数量,x4只有当x3=1时才有意义
def cost_finished_products(n1,n2,m,x1,x2,x3,x4):

    num_of_finished_products = min(n1,n2) # 可以产生成品的零件组数量

    cost_assemble = num_of_finished_products*data['成品装配成本'][m] # 装配成本，成本4
    
    cost_test = num_of_finished_products*data['成品检测成本'][m] # 检测成本，成本5

    num_of_reject = Num_of_reject(x1,x2,num_of_finished_products,m) # 次品个数

    cost_chaijie = num_of_reject*data['不合格成品拆解成本'][m] # 拆解成本，成本7

    cost_of_all = cost_assemble+x3*(cost_test+x4*cost_chaijie) # 总成本

    num_of_finished_products = num_of_finished_products - x3*num_of_reject # 实际产生的成品数量

    return cost_of_all,num_of_finished_products,x4*num_of_reject



# 定义不合格成品调换损失，以及召回的不合格成品数量,n为成品数量，m为第m种情况
def cost_loss(x1,x2,x3,num_of_finished_products,m):
    # 如果经过成品检测到市场，不合格成品的数量为0
    if x3==1:
        return 0,0
    else:
        num_of_loss = Num_of_reject(x1,x2,num_of_finished_products,m) # 次品个数
        loss = num_of_loss*data['不合格成品调换损失'][m] # 总调换损失
    return loss,num_of_loss


# 定义总利润函数
#x1,x2,x3,x4分别表示零配件1是否检测，零配件2是否检测，成品是否检测，不合格成品是否拆解,召回的成品是否拆解
def profit(x1,x2,x3,x4,x5,n1,n2,m):
    m-=1
    # 第一轮
    # 零配件的购买及检测成本
    cost1,num_of_spare1 = cost_spare1(x1,n1,m)
    cost2,num_of_spare2 = cost_spare2(x2,n2,m)
    
    # 零配件使用后剩余的零件数量，由于可能有些零配件检测出次品数不同，可能出现两种配件数量不匹配的情况
    n1-=min(num_of_spare1,num_of_spare2)
    n2-=min(num_of_spare1,num_of_spare2)
    
    # 成品装配及检测拆解的成本,以及实际产生的成品数量，拆解的不合格成品数量
    cost3,num_of_finished_products,num_of_chaijie = cost_finished_products(num_of_spare1,num_of_spare2,m,x1,x2,x3,x4)
    
    # 不合格成品调换损失，以及召回的不合格成品数量
    cost4,num_of_loss = cost_loss(x1,x2,x3,num_of_finished_products,m)

    total_cost = cost1+cost2+cost3+cost4+x5*num_of_loss*data['不合格成品拆解成本'][m]

    total_profit = data['市场售价'][m]*(num_of_finished_products-num_of_loss)-total_cost

    # 拆解所得零件和召回的零件进行第二轮，默认全部检测且不再进行拆解
    n1 += num_of_chaijie+x5*num_of_loss
    n2 += num_of_chaijie+x5*num_of_loss
    if n1>0 and n2>0:
        cost1,num_of_spare1 = cost_spare1(1,n1,m)
        cost2,num_of_spare2 = cost_spare2(1,n2,m)

        cost3,num_of_finished_products,num_of_chaijie = cost_finished_products(num_of_spare1,num_of_spare2,m,1,1,1,0)

        cost4,num_of_loss = cost_loss(1,1,1,num_of_finished_products,m)

        total_cost = cost1+cost2+cost3+cost4

        total_profit += data['市场售价'][m]*(num_of_finished_products-num_of_loss)-total_cost
    return total_profit

# 主程序

# 导入itertools用于生成组合
import itertools

# 载入数据
data = pd.read_csv('data/Q2.csv')
# 将相关列转换为浮点数，确保可以进行数学运算
# 假设数据列 '零配件 1 次品率' 和其他次品率列可能以百分比或字符串形式存在
data['零配件 1 次品率'] = pd.to_numeric(data['零配件 1 次品率'].str.rstrip('%'), errors='coerce') / 100
data['零配件 2 次品率'] = pd.to_numeric(data['零配件 2 次品率'].str.rstrip('%'), errors='coerce') / 100
data['成品次品率'] = pd.to_numeric(data['成品次品率'].str.rstrip('%'), errors='coerce') / 100

# 确保所有其他涉及到数值运算的列都是数值类型
data['零配件 1 购买单价'] = pd.to_numeric(data['零配件 1 购买单价'], errors='coerce')
data['零配件 2 购买单价'] = pd.to_numeric(data['零配件 2 购买单价'], errors='coerce')
data['市场售价'] = pd.to_numeric(data['市场售价'], errors='coerce')

print(data)


# 生成所有有效的组合
combinations = list(itertools.product([0, 1], repeat=5))
valid_combinations = [combo for combo in combinations if combo[3] == 0 or combo[2] == 1]

# 循环m取值1到6
for m in range(1, 7):
    # 存储每种m值对应的组合及利润
    results = []
    for combo in valid_combinations:
        p = profit(*combo, n1=1000, n2=1000, m=m)
        results.append({'x1': combo[0], 'x2': combo[1], 'x3': combo[2], 'x4': combo[3], 'x5': combo[4], 'profit': p})

    # 创建 DataFrame，并将 x1, x2, x3, x4, x5 分开
    df_results = pd.DataFrame(results)

    # 将结果保存到CSV文件，文件名根据m值不同
    csv_filename = f'result/result_q2/result_q2_m{m}.csv'
    df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    # 找到最大利润
    max_profit = df_results['profit'].max()
    print(f"第 {m} 种情况的最大利润为: {max_profit}")
    print('此时，x1, x2, x3, x4, x5的取值为：')
    print(df_results.loc[df_results['profit'].idxmax()])

    # 为绘图准备横坐标的组合字符串
    df_results['combination'] = df_results[['x1', 'x2', 'x3', 'x4', 'x5']].astype(str).agg(''.join, axis=1)

    # 绘制利润图，横坐标为组合字符串
    plt.figure(figsize=(10, 5))
    plt.bar(df_results['combination'], df_results['profit'])
    plt.title(f"Profit per Combination (Scenario {m})")
    plt.xlabel("Combination (x1, x2, x3, x4, x5)")
    plt.ylabel("Profit")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # 保存图片，文件名根据m值不同
    pdf_filename = f'image/image_q2/Q2_scenario_{m}.png'
    plt.savefig(pdf_filename, format='png')
