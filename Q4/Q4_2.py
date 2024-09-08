import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题
# 定义数据
data = [
    {
        '情况': 1,
        '零配件 1 次品率': 0.1,
        '零配件 1 购买单价': 4,
        '零配件 1 检测成本': 2,
        '零配件 2 次品率': 0.1,
        '零配件 2 购买单价': 18,
        '零配件 2 检测成本': 3,
        '成品次品率': 0.1,
        '成品装配成本': 6,
        '成品检测成本': 3,
        '市场售价': 56,
        '不合格成品调换损失': 6,
        '不合格成品拆解成本': 5
    },
    {
        '情况': 2,
        '零配件 1 次品率': 0.2,
        '零配件 1 购买单价': 4,
        '零配件 1 检测成本': 2,
        '零配件 2 次品率': 0.2,
        '零配件 2 购买单价': 18,
        '零配件 2 检测成本': 3,
        '成品次品率': 0.2,
        '成品装配成本': 6,
        '成品检测成本': 3,
        '市场售价': 56,
        '不合格成品调换损失': 6,
        '不合格成品拆解成本': 5
    },
    {
        '情况': 3,
        '零配件 1 次品率': 0.1,
        '零配件 1 购买单价': 4,
        '零配件 1 检测成本': 2,
        '零配件 2 次品率': 0.1,
        '零配件 2 购买单价': 18,
        '零配件 2 检测成本': 3,
        '成品次品率': 0.1,
        '成品装配成本': 6,
        '成品检测成本': 3,
        '市场售价': 56,
        '不合格成品调换损失': 30,
        '不合格成品拆解成本': 5
    },
    {
        '情况': 4,
        '零配件 1 次品率': 0.2,
        '零配件 1 购买单价': 4,
        '零配件 1 检测成本': 1,
        '零配件 2 次品率': 0.2,
        '零配件 2 购买单价': 18,
        '零配件 2 检测成本': 1,
        '成品次品率': 0.2,
        '成品装配成本': 6,
        '成品检测成本': 2,
        '市场售价': 56,
        '不合格成品调换损失': 30,
        '不合格成品拆解成本': 5
    },
    {
        '情况': 5,
        '零配件 1 次品率': 0.1,
        '零配件 1 购买单价': 4,
        '零配件 1 检测成本': 8,
        '零配件 2 次品率': 0.2,
        '零配件 2 购买单价': 18,
        '零配件 2 检测成本': 1,
        '成品次品率': 0.1,
        '成品装配成本': 6,
        '成品检测成本': 2,
        '市场售价': 56,
        '不合格成品调换损失': 10,
        '不合格成品拆解成本': 5
    },
    {
        '情况': 6,
        '零配件 1 次品率': 0.05,
        '零配件 1 购买单价': 4,
        '零配件 1 检测成本': 2,
        '零配件 2 次品率': 0.05,
        '零配件 2 购买单价': 18,
        '零配件 2 检测成本': 3,
        '成品次品率': 0.05,
        '成品装配成本': 6,
        '成品检测成本': 3,
        '市场售价': 56,
        '不合格成品调换损失': 10,
        '不合格成品拆解成本': 40
    }
]
# 成品抽样检测
import numpy as np

def cost_sample_finished_product(m, data):
    """
    计算成品的抽样检测是否合格及抽样检测的成本。

    :param m: 情况编号
    :param data: 包含所有成品成本信息的字典或DataFrame
    :return: 是否抽样检测合格，抽样检测成本，抽样检测的次品率
    """
    # 获取成品次品率
    defect_rate = data[m]['成品次品率']
    
    # 初始抽样成本
    cost = 0
    
    # 初始抽样数量
    initial_samples = 162
    final_samples = 223
    
    # 计算下界和上界
    lower_bound = 1.28 * np.sqrt(defect_rate * (1 - defect_rate) / initial_samples) + defect_rate
    upper_bound = 1.645 * np.sqrt(defect_rate * (1 - defect_rate) / final_samples) + defect_rate
    
    # 获取成品检测成本
    unit_cost = data[m]['成品检测成本']
    
    # 模拟抽样过程
    # 初始抽样
    # print(defect_rate)
    sampled_defects = np.random.binomial(initial_samples, defect_rate)
    cost += initial_samples * unit_cost
    
    # 检查是否需要继续抽样
    if sampled_defects < lower_bound * initial_samples:
        # 不再抽样，检测合格
        total_samples = initial_samples
        total_defects = sampled_defects
    elif sampled_defects > upper_bound * initial_samples:
        # 检测不合格
        total_samples = initial_samples
        total_defects = sampled_defects
    else:
        # 继续抽样到223个
        additional_samples = final_samples - initial_samples
        additional_defects = np.random.binomial(additional_samples, defect_rate)
        cost += additional_samples * unit_cost
        
        # 更新总抽样数和次品数
        total_samples = final_samples
        total_defects = sampled_defects + additional_defects
    
    # 计算抽样检测的次品率
    sample_defect_rate = total_defects / total_samples
    
    # 最终判断是否合格
    if total_defects < lower_bound * total_samples:
        # 抽样检测合格
        return True, cost, sample_defect_rate
    elif total_defects > upper_bound * total_samples:
        # 抽样检测不合格
        return False, cost, sample_defect_rate
    else:
        # 如果在边界之间，默认认为合格
        return True, cost, sample_defect_rate
# 零配件的抽样检测
def cost_sample(part_number, m, data):
    """
    计算给定零件的抽样检测是否合格及抽样检测的成本，并返回抽样检测的次品率。

    :param part_number: 零件编号
    :param m: 情况编号
    :param data: 包含所有零件成本信息的字典或DataFrame
    :return: 是否抽样检测合格，抽样检测成本，抽样检测的次品率
    """
    np.random.seed(0)
    # 获取次品率
    defect_rate = data[m][f'零配件 {part_number} 次品率']
    
    # 初始抽样成本
    cost = 0
    
    # 抽样到162个
    initial_samples = 162
    final_samples = 223
    
    # 计算下界和上界
    lower_bound = 1.28 * np.sqrt(defect_rate * (1 - defect_rate) / initial_samples) + defect_rate
    upper_bound = 1.645 * np.sqrt(defect_rate * (1 - defect_rate) / final_samples) + defect_rate
    
    # 模拟抽样过程
    # 假设实际抽样过程中每个样本的检测成本为1单位成本
    unit_cost = data[m][f'零配件 {part_number} 检测成本']
    
    # 初始抽样
    sampled_defects = np.random.binomial(initial_samples, defect_rate)
    cost += initial_samples * unit_cost
    
    # 检查是否需要继续抽样
    if sampled_defects < lower_bound * initial_samples:
        # 不再抽样，检测合格
        total_samples = initial_samples
        total_defects = sampled_defects
    elif sampled_defects > upper_bound * initial_samples:
        # 检测不合格
        total_samples = initial_samples
        total_defects = sampled_defects
    else:
        # 继续抽样到223个
        additional_samples = final_samples - initial_samples
        additional_defects = np.random.binomial(additional_samples, defect_rate)
        cost += additional_samples * unit_cost
        
        # 更新总抽样数和次品数
        total_samples = final_samples
        total_defects = sampled_defects + additional_defects
    
    # 计算抽样检测的次品率
    sample_defect_rate = total_defects / total_samples
    
    # 最终判断是否合格
    if total_defects < lower_bound * total_samples:
        # 抽样检测合格
        return True, cost, sample_defect_rate
    elif total_defects > upper_bound * total_samples:
        # 抽样检测不合格
        return False, cost, sample_defect_rate
    else:
        # 如果在边界之间，默认认为合格
        return True, cost, sample_defect_rate


# 零件的购买及检测成本
# 两个参数，x1为0-1变量，表示是否检测，n表示购买数量，m表示第m种情况
def cost_spare(part_number, x1, n, m, data):
    """
    计算给定零件的成本和有效数量。

    :param part_number: 数字，表示零件的编号
    :param x1: 0-1-2变量，表示进行全检（1），不检测（0），抽检（2）
    :param n: 购买数量
    :param m: 第m种情况
    :param data: 包含所有零件成本信息的字典
    :return: 成本和有效数量的元组
    """
    # 构建键名
    purchase_cost_key = f'零配件 {part_number} 购买单价'
    inspection_cost_key = f'零配件 {part_number} 检测成本'
    defect_rate_key = f'零配件 {part_number} 次品率'
    # print(data)
    # print(data[m][purchase_cost_key])
    if x1 == 1:
        # 全检
        cost = data[m][purchase_cost_key] * n + data[m][inspection_cost_key] * n
        num_of_spare = n - int(n * data[m][defect_rate_key])
    elif x1 == 0:
        # 不检测
        cost = data[m][purchase_cost_key] * n
        num_of_spare = n
    elif x1 == 2:
        # 抽检
        is_pass, sample_cost,sample_defect_rate = cost_sample(part_number, m, data)
        if is_pass:
            # 抽检合格
            num_of_spare = n
            cost = data[m][purchase_cost_key] * n + sample_cost
            data[m][defect_rate_key]= sample_defect_rate
        else:
            # 抽检不合格，需要退换
            if data[m][defect_rate_key]>=0.05: data[m][defect_rate_key] = data[m][defect_rate_key] - 0.01  # 次品率下降1个百分点
            num_of_spare = n 
            cost = data[m][purchase_cost_key] * n + sample_cost + 0.1* n  # 退换的物流及时间成本这里设置为一个0.1
    else:
        raise ValueError("x1 必须为 0、1 或 2")

    return cost, num_of_spare
# 成品次品数量
def Num_of_reject(x1, x2, num_of_finished_products, m, data):
    """
    根据不同的零件状态计算成品中的次品数量。

    :param x1: 零配件1的状态，1表示可能有次品，0或2表示无次品
    :param x2: 零配件2的状态，1表示可能有次品，0或2表示无次品
    :param num_of_finished_products: 成品的数量
    :param m: 情况编号
    :param data: 包含所有零件次品率和成品次品率的字典或DataFrame
    :return: 成品中的次品数量
    """
    # 获取相关的次品率
    reject_rate_final = data[m]['成品次品率']
    reject_rate_part1 = data[m][f'零配件 1 次品率']
    reject_rate_part2 = data[m][f'零配件 2 次品率']
    # 将x1和x2的值标准化，将2视为0
    x1 = 0 if x1 == 2 else x1
    x2 = 0 if x2 == 2 else x2

    # 根据零件状态计算次品数量
    if x1 == 1 and x2 == 1:
        # 不考虑零配件次品，仅由成品组装导致
        num_of_reject_final = int(num_of_finished_products * reject_rate_final)
    elif x1 == 1 and x2 == 0:
        # 考虑零配件2的次品，然后加上剩余零件由成品组装导致的次品
        num_of_reject_2 = int(num_of_finished_products * reject_rate_part2)
        num_of_reject_final = int((num_of_finished_products - num_of_reject_2) * reject_rate_final) + num_of_reject_2
    elif x1 == 0 and x2 == 1:
        # 考虑零配件1的次品，然后加上剩余零件由成品组装导致的次品
        num_of_reject_1 = int(num_of_finished_products * reject_rate_part1)
        num_of_reject_final = int((num_of_finished_products - num_of_reject_1) * reject_rate_final) + num_of_reject_1
    else:  # x1 == 0 and x2 == 0
        # 考虑两者的次品率，计算正品数量，剩余为次品
        num_of_qualify = int((1 - reject_rate_part1) * (1 - reject_rate_part2) * (1 - reject_rate_final) * num_of_finished_products)
        num_of_reject_final = int(num_of_finished_products - num_of_qualify)

    return num_of_reject_final


# 成品的检测及拆解成本
def cost_finished_products(n1, n2, m, x1, x2, x3, x4, data):
    """
    计算成品装配及检测拆解的成本。

    :param n1: 零配件1的数量
    :param n2: 零配件2的数量
    :param m: 第m种情况
    :param x1: 零配件1的状态，0表示不检测，1表示全检，2表示抽检
    :param x2: 零配件2的状态，0表示不检测，1表示全检，2表示抽检
    :param x3: 成品的状态，0表示不检测，1表示全检，2表示抽检
    :param x4: 不合格成品的状态，0表示不拆解，1表示拆解
    :param data: 包含所有成本信息的字典
    :return: 总成本，实际产生的成品数量，拆解的成品数量
    """
    # 可以产生成品的零件组数量
    num_of_finished_products = min(n1, n2)

    # 计算次品数量
    num_of_reject = Num_of_reject(x1, x2, num_of_finished_products, m, data)

    # 装配成本
    cost_assemble = num_of_finished_products * data[m]['成品装配成本']


      # 检测成本
    if x3 == 1:
        # 全检
        cost_test = num_of_finished_products * data[m]['成品检测成本']
    elif x3 == 0:
        # 不检测
        cost_test = 0
    elif x3 == 2:
        # 抽检
        is_pass, sample_cost, sample_defect_rate = cost_sample_finished_product(m, data)
        if is_pass:
            # 抽检合格
            cost_test = sample_cost
            # 抽检合格后次品率降低
            if data[m]['成品次品率']>=0.05: data[m]['成品次品率'] = sample_defect_rate - 0.01  # 降低1个百分点
        else:
            # 抽检不合格，需要全检
            cost_test = sample_cost + num_of_finished_products * data[m]['成品检测成本']
            # 更新次品率
            data[m]['成品次品率'] = sample_defect_rate
    else:
        raise ValueError("x3 必须为 0、1 或 2")

    # 计算次品数量
    num_of_reject = Num_of_reject(x1, x2, num_of_finished_products, m, data)

    # 拆解成本
    if x4 == 1:
        # 需要拆解
        cost_chaijie = num_of_reject * data[m]['不合格成品拆解成本']
    elif x4 == 0:
        # 不需要拆解
        cost_chaijie = 0
    else:
        raise ValueError("x4 必须为 0 或 1")

    # 总成本
    cost_of_all = cost_assemble + cost_test + cost_chaijie

    # 实际产生的成品数量
    num_of_finished_products -= num_of_reject

    return cost_of_all, num_of_finished_products, num_of_reject

def cost_loss(x1, x2, x3, num_of_finished_products, m, data):
    """
    计算不合格成品的调换损失及其数量。

    :param x1: 零配件1的状态，0表示不检测，1表示全检，2表示抽检
    :param x2: 零配件2的状态，0表示不检测，1表示全检，2表示抽检
    :param x3: 成品的状态，0表示不检测，1表示全检，2表示抽检
    :param num_of_finished_products: 成品的数量
    :param m: 第m种情况
    :param data: 包含所有成本信息的字典
    :return: 调换损失，不合格成品的数量
    """
    # 如果成品经过全检，不合格成品的数量为0
    if x3 == 1:
        return 0, 0
    elif x3 == 0:
        # 不检测
        num_of_reject = Num_of_reject(x1, x2, num_of_finished_products, m, data)
        loss = num_of_reject * data[m]['不合格成品调换损失']
        return loss, num_of_reject
    elif x3 == 2:
        # 抽检
        is_pass, sample_cost, sample_defect_rate = cost_sample_finished_product(m, data)
        if is_pass:
            # 抽检合格
            # 抽检合格后次品率降低
            if data[m]['成品次品率']>0.05: data[m]['成品次品率'] = sample_defect_rate - 0.01  # 降低1个百分点
            num_of_reject = Num_of_reject(x1, x2, num_of_finished_products, m, data)
            loss = num_of_reject * data[m]['不合格成品调换损失']
            return loss, num_of_reject
        else:
            # 抽检不合格，需要全检
            # 更新次品率
            data[m]['成品次品率'] = sample_defect_rate
            num_of_reject = Num_of_reject(x1, x2, num_of_finished_products, m, data)
            loss = num_of_reject * data[m]['不合格成品调换损失']
            return loss, num_of_reject
    else:
        raise ValueError("x3 必须为 0、1 或 2")

# 定义总利润函数

def profit(x1, x2, x3, x4, x5, n1, n2, m, data):
    m -= 1
    
    # 第一轮
    # 零配件的购买及检测成本
    cost1, num_of_spare1 = cost_spare(1, x1, n1, m, data)
    cost2, num_of_spare2 = cost_spare(2, x2, n2, m, data)
    
    # 零配件使用后剩余的零件数量
    used_spares = min(num_of_spare1, num_of_spare2)
    n1 -= used_spares
    n2 -= used_spares
    
    # 成品装配及检测拆解的成本,以及实际产生的成品数量，拆解的不合格成品数量

    cost3, num_of_finished_products, num_of_chaijie = cost_finished_products(used_spares, used_spares, m, x1, x2, x3, x4, data)
    
    # 不合格成品调换损失，以及召回的不合格成品数量
    cost4, num_of_loss = cost_loss(x1, x2, x3, num_of_finished_products, m, data)

    # 计算总成本
    total_cost = cost1 + cost2 + cost3 + cost4 + x5 * num_of_loss * data[m]['不合格成品拆解成本']

    # 计算总利润
    total_profit = data[m]['市场售价'] * (num_of_finished_products - num_of_loss) - total_cost

    # 拆解所得零件和召回的零件进行第二轮，默认全部检测且不再进行拆解
    n1 += num_of_chaijie + x5 * num_of_loss
    n2 += num_of_chaijie + x5 * num_of_loss
    if n1 > 0 and n2 > 0:
        # 第二轮
        cost1, num_of_spare1 = cost_spare(1, 1, n1, m, data)
        cost2, num_of_spare2 = cost_spare(2, 1, n2, m, data)

        cost3, num_of_finished_products, num_of_chaijie = cost_finished_products(min(num_of_spare1, num_of_spare2), min(num_of_spare1, num_of_spare2), m, 1, 1, 1, 0, data)

        cost4, num_of_loss = cost_loss(1, 1, 1, num_of_finished_products, m, data)

        total_cost = cost1 + cost2 + cost3 + cost4

        total_profit += data[m]['市场售价'] * (num_of_finished_products - num_of_loss) - total_cost

    return total_profit

# 主程序

import itertools
import pandas as pd
import matplotlib.pyplot as plt

# 假设以下函数已经定义好并且可以在环境中使用
# profit
# 这些函数的定义不在这里给出，但它们应该按照之前描述的方式工作

# 生成 x1, x2, x3 的所有可能的组合
combinations_123 = list(itertools.product([0, 1, 2], repeat=3))  # x1, x2, x3

# 生成 x4, x5 的所有可能的组合
combinations_45 = list(itertools.product([0, 1], repeat=2))  # x4, x5

# 组合成完整的5元组
full_combinations = [combo123 + combo45 for combo123 in combinations_123 for combo45 in combinations_45]
# 应用有效性检查
valid_combinations = [combo for combo in full_combinations if combo[3] == 0 or combo[2] == 1]

# 循环m取值1到6
for m in range(1, 7):
    # 选择对应的数据
    current_data = data

    # 存储每种m值对应的组合及利润
    results = []
    for combo in valid_combinations:
        # 每次循环前重置数据
        data_copy = current_data.copy()
        
        p = profit(combo[0], combo[1], combo[2], combo[3], combo[4], n1=1000, n2=1000, m=m, data=data_copy)
        results.append({
            'x1': combo[0],
            'x2': combo[1],
            'x3': combo[2],
            'x4': combo[3],
            'x5': combo[4],
            'profit': p
        })

    # 创建 DataFrame，并将 x1, x2, x3, x4, x5 分开
    df_results = pd.DataFrame(results)

    # 确保结果目录存在
    os.makedirs('result/result_q4_2', exist_ok=True)
    os.makedirs('image/image_q4_2', exist_ok=True)

    # 将结果保存到CSV文件，文件名根据m值不同
    csv_filename = f'result/result_q4_2/result_m{m}.csv'
    df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    # 找到profit列的最大值
    max_profit = df_results['profit'].max()

    # 使用mask找到所有profit等于最大值的行
    max_profit_mask = df_results['profit'] == max_profit

    # 获取所有最大值出现的索引
    max_profit_indices = df_results.index[max_profit_mask].tolist()

    # 获取最大值最后一次出现的索引
    last_max_profit_index = max_profit_indices[-1]

    # 打印最大利润及其对应的x1, x2, x3, x4, x5的值
    print(f"第 {m} 种情况的最大利润为: {max_profit}")
    print('此时，x1, x2, x3, x4, x5的取值为：')
    print(df_results.loc[last_max_profit_index])
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
    pdf_filename = f'image/image_q4_2/Q4_2_scenario_{m}.png'
    plt.savefig(pdf_filename, format='png')
    plt.close()  # 关闭图像以节省内存

print("所有场景的利润分析完成并已保存。")