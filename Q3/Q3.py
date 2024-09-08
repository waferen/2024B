import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import itertools
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题

scenarios = {
        # 零配件数据
        "part1_defect_rate": 0.10, "part1_cost": 2, "part1_test_cost": 1,
        "part2_defect_rate": 0.10, "part2_cost": 8, "part2_test_cost": 1,
        "part3_defect_rate": 0.10, "part3_cost": 12, "part3_test_cost": 2,
        "part4_defect_rate": 0.10, "part4_cost": 2, "part4_test_cost": 1,
        "part5_defect_rate": 0.10, "part5_cost": 8, "part5_test_cost": 1,
        "part6_defect_rate": 0.10, "part6_cost": 12, "part6_test_cost": 2,
        "part7_defect_rate": 0.10, "part7_cost": 8, "part7_test_cost": 1,
        "part8_defect_rate": 0.10, "part8_cost": 12, "part8_test_cost": 2,
        # 半成品数据
        "semi_product1_defect_rate": 0.10, "semi_product1_assembly_cost": 8, "semi_product1_test_cost": 4, "semi_product1_disassembly_cost": 6,
        "semi_product2_defect_rate": 0.10, "semi_product2_assembly_cost": 8, "semi_product2_test_cost": 4, "semi_product2_disassembly_cost": 6,
        "semi_product3_defect_rate": 0.10, "semi_product3_assembly_cost": 8, "semi_product3_test_cost": 4, "semi_product3_disassembly_cost": 6,
        # 成品数据
        "finished_product_defect_rate": 0.10,"": 8, "finished_product_test_cost": 6, "finished_product_disassembly_cost": 10,
        "market_price": 200, "replacement_loss": 40,
        "num_parts1": 100, "num_parts2": 100
    }


# 零件是否检测的0-1变量列表x_values，零件的数量列表n_parts，半成品的编号semi_product_number，半成品是否检测的0-1变量y_value，半成品是否拆解的0-1变量z_value/
# 成品是否检测的0-1变量p_value，成品是否拆解的0-1变量q_value

# 零部件的购买检测费用和剩余数量
def calculate_cost_and_spare(part_number, x, n, scenarios):

    # 构造零件信息的键
    part_info_key = f"{part_number}_defect_rate"
    # 获取零件的次品率
    defect_rate = scenarios[part_info_key]
    # 获取零件的购买单价
    purchase_price = scenarios[f"{part_number}_cost"]
    # 获取零件的检测成本
    test_cost = scenarios[f"{part_number}_test_cost"]

    # print(f"零件{part_number}的购买单价为{purchase_price}，次品率为{defect_rate}，检测成本为{test_cost},购买数量为{n}")
    # 总成本包括购买成本和检测成本
    total_cost = purchase_price * n + x * (test_cost * n)
    
    # 可用的零件数量为订购数量检测出的次品数量
    usable_quantity = n - x * int(n * defect_rate)
    return total_cost, usable_quantity

# 半成品次品数量
def num_of_reject_semi_product(semi_product_number, x_values, num_of_semi_products, scenarios):
    """
    计算指定半成品组装后的次品数量。

    参数:
    semi_product_number (int): 半成品的编号，用于确定使用的零件。
    x_values (list of int): 对应零件是否检测的0-1变量列表。
    num_of_semi_products (int): 半成品的数量。
    scenarios (dict): 包含所有零件和半成品数据的字典。

    返回:
    int: 次品数量。
    """
    # 定义每个半成品对应的零件编号
    semi_product_parts = {
        1: ['part1', 'part2', 'part3'],
        2: ['part4', 'part5', 'part6'],
        3: ['part7', 'part8']
    }

    # 获取当前半成品的次品率
    defect_rate_semi_product = scenarios[f"semi_product{semi_product_number}_defect_rate"]

    # 初始化合格组合的概率
    qualified_combination_rate = 1

    # 获取当前半成品对应的零件编号列表
    part_numbers = semi_product_parts.get(semi_product_number, [])

    # 计算每个零件的有效次品率，并更新合格组合的概率
    for i, part_number in enumerate(part_numbers):
        defect_rate = scenarios[f"{part_number}_defect_rate"]
        # 如果零件进行了检测，则其次品率为0
        effective_defect_rate = 0 if x_values[i] == 1 else defect_rate
        qualified_combination_rate *= (1 - effective_defect_rate)

    # 计算合格组合数量
    num_of_qualified_combinations = int(qualified_combination_rate * num_of_semi_products)

    # 计算合格零件组合中由于装配过程产生的次品数量
    num_of_unqualified_combinations = int(num_of_qualified_combinations * defect_rate_semi_product)

    # 计算由不合格零件产生的次品数
    num_of_unqualified_parts = num_of_semi_products - num_of_qualified_combinations
    
    # 计算总次品数量
    num_of_reject_final = num_of_unqualified_combinations + num_of_unqualified_parts 

    return num_of_reject_final


# 半成品的装配、检测和拆解成本
def cost_semi_product(n_parts, semi_product_number, x_values, y_value, z_value, scenarios):
    """
    计算半成品装配、检测和拆解的成本。

    参数:
    n_parts (list of int): 每个零件的数量列表，例如 [n1, n2, n3]。
    semi_product_number (int): 半成品的编号，用于确定使用的零件。
    x_values (list of int): 对应零件是否检测的0-1变量列表。
    y_value (int): 半成品是否检测的0-1变量。
    z_value (int): 半成品是否拆解的0-1变量。
    scenarios (dict): 包含所有零件和半成品数据的字典。

    返回:
    tuple: 包含三个元素的元组，分别为总成本、实际产生的半成品数量,拆解的不合格成品数量,半成品次品数量。
    """

    # 定义每个半成品对应的零件编号
    semi_product_parts = {
        1: ['part1', 'part2', 'part3'],
        2: ['part4', 'part5', 'part6'],
        3: ['part7', 'part8']
    }

    # 获取当前半成品对应的零件编号列表
    part_numbers = semi_product_parts.get(semi_product_number, [])

    # 可以产生成品的零件组数量
    num_of_semi_products = min(n_parts)

    # 计算半成品的装配成本
    cost_assemble = num_of_semi_products * scenarios[f'semi_product{semi_product_number}_assembly_cost']

    # 计算半成品的检测成本
    cost_test = num_of_semi_products * scenarios[f'semi_product{semi_product_number}_test_cost'] if y_value == 1 else 0

    # 计算半成品中的次品数量
    num_of_reject = num_of_reject_semi_product(semi_product_number, x_values, num_of_semi_products, scenarios)

    # 计算不合格半成品的拆解成本
    cost_chaijie = num_of_reject * scenarios[f'semi_product{semi_product_number}_disassembly_cost'] if y_value == 1 and z_value == 1 else 0

    # 总成本
    cost_of_all = cost_assemble + cost_test + cost_chaijie

    # 实际产生的半成品数量
    num_of_semi_products_actual = num_of_semi_products - num_of_reject if y_value == 1 else num_of_semi_products

    # 拆解的不合格半成品数量
    num_of_chaijie = num_of_reject if y_value == 1 and z_value == 1 else 0

    return cost_of_all, num_of_semi_products_actual, num_of_chaijie, num_of_reject




# 成品次品数量
def num_of_reject_finished_product(x_values, y_values, num_of_finished_products, scenarios):
    """
    计算成品的次品数量。

    参数:
    x_values (list of int): 对应零件是否检测的0-1变量列表。
    y_values (list of int): 对应半成品是否检测的0-1变量列表。
    num_of_finished_products (int): 成品的数量。
    scenarios (dict): 包含所有零件和半成品数据的字典。

    返回:
    int: 次品数量。
    """
    # 定义每个半成品对应的零件编号
    semi_product_parts = {
        1: ['part1', 'part2', 'part3'],
        2: ['part4', 'part5', 'part6'],
        3: ['part7', 'part8']
    }

    # 固定的半成品编号列表
    semi_product_numbers = [1, 2, 3]

    # 初始化合格组合的概率
    qualified_combination_rate = 1

    # 计算每个半成品及其零件的有效次品率，并更新合格组合的概率
    for i, semi_product_number in enumerate(semi_product_numbers):
        # 获取当前半成品对应的零件编号列表
        part_numbers = semi_product_parts.get(semi_product_number, [])

        # 计算每个零件的有效次品率，并更新合格组合的概率
        part_effective_defect_rate = 1
        for j, part_number in enumerate(part_numbers):
            defect_rate_part = scenarios[f"{part_number}_defect_rate"]
            # 如果零件进行了检测，则其次品率为0
            effective_defect_rate = 0 if x_values[j] == 1 else defect_rate_part
            part_effective_defect_rate *= (1 - effective_defect_rate)

        # 获取当前半成品的次品率
        defect_rate_semi_product = scenarios[f"semi_product{semi_product_number}_defect_rate"]
        # 如果半成品进行了检测，则其次品率为0
        effective_defect_rate_semi_product = 0 if y_values[i] == 1 else defect_rate_semi_product
        # 更新合格组合的概率
        qualified_combination_rate *= (1 - part_effective_defect_rate * effective_defect_rate_semi_product)

    # 计算有效组合数量
    num_of_qualified_combinations = int(qualified_combination_rate * num_of_finished_products)

    # 计算合格零件组合中由于装配过程产生的次品数量
    num_of_unqualified_combinations = int(num_of_qualified_combinations * scenarios['finished_product_defect_rate'])

    # 计算成品中的次品数量
    num_of_reject_final = num_of_unqualified_combinations + num_of_finished_products - num_of_qualified_combinations

    return num_of_reject_final



# 成品的装配、检测和拆解成本
def cost_finished_product(n_semi_products, x_values, y_values, p_value, q_value, scenarios):
    """
    计算成品装配、检测和拆解的成本。

    参数:
    n_semi_products (list of int): 每个半成品的数量列表，例如 [n1, n2, n3]。
    semi_product_numbers (list of int): 半成品的编号列表，例如 [1, 2, 3]。
    y_values (list of int): 对应半成品是否检测的0-1变量列表。
    p_value (int): 成品是否检测的0-1变量。
    q_value (int): 成品是否拆解的0-1变量。
    scenarios (dict): 包含所有零件和半成品数据的字典。

    返回:
    tuple: 包含四个元素的元组，分别为总成本、实际产生的成品数量、拆解的不合格成品数量、成品次品数。
    """

    # 可以产生成品的半成品组数量
    num_of_finished_products = min(n_semi_products)

    # 计算成品的装配成本
    cost_assemble = num_of_finished_products * scenarios['']

    # 计算成品的检测成本
    cost_test = num_of_finished_products * scenarios['finished_product_test_cost'] if p_value == 1 else 0

    # 计算成品中的次品数量
    num_of_reject = num_of_reject_finished_product(x_values, y_values, num_of_finished_products, scenarios)

    # 计算不合格成品的拆解成本
    cost_chaijie = num_of_reject * scenarios['finished_product_disassembly_cost'] if p_value == 1 and q_value == 1 else 0

    # 实际产生的成品数量
    num_of_finished_products_actual = num_of_finished_products - num_of_reject if p_value == 1 else num_of_finished_products

    # 总成本
    cost_of_all = cost_assemble + cost_test + cost_chaijie

    # 拆解的不合格成品数量
    num_of_chaijie = num_of_reject if p_value == 1 and q_value == 1 else 0

    return cost_of_all, num_of_finished_products_actual, num_of_chaijie, num_of_reject



def cost_loss(x_values, y_values, p_value, num_of_finished_products, scenarios):
    """
    计算不合格成品的调换损失和召回的不合格成品数量。

    参数:
    x_values (list of int): 对应零件是否检测的0-1变量列表。
    y_values (list of int): 对应半成品是否检测的0-1变量列表。
    p_value (int): 成品是否检测的0-1变量。
    num_of_finished_products (int): 成品的数量。
    scenarios (dict): 包含所有零件和半成品数据的字典。

    返回:
    tuple: 包含两个元素的元组，分别为总调换损失和召回的不合格成品数量。
    """

    # 如果经过成品检测到市场，不合格成品的数量为0
    if p_value == 1:
        return 0, 0
    else:
        # 计算成品的次品数量
        num_of_reject = num_of_reject_finished_product(x_values, y_values, num_of_finished_products, scenarios)
        
        # 计算总调换损失
        # 使用 scenarios 中的 replacement_loss 字段
        loss = num_of_reject * scenarios['replacement_loss']

    return loss, num_of_reject


def profit(x_values, y_values, z_values, p_value, q_value, r_value, n_parts, scenarios):
    """
    计算总利润，并记录每种零件被拆解得到的数量。

    参数:
    x_values (list of int): 八个零件是否检测的0-1变量列表。
    y_values (list of int): 三个半成品是否检测的0-1变量列表。
    z_values (list of int): 三个半成品是否拆解的0-1变量列表。
    p_value (int): 成品是否检测的0-1变量。
    q_value (int): 成品是否拆解的0-1变量。
    r_value (int): 市场退回的不合格成品是否拆解的0-1变量。
    n_parts (list of int): 八个零件的数量列表。
    scenarios (dict): 包含所有零件和半成品数据的字典。

    返回:
    float: 总利润。
    """
    # 定义每个半成品对应的零件编号
    semi_product_parts = {
        1: ['part1', 'part2', 'part3'],
        2: ['part4', 'part5', 'part6'],
        3: ['part7', 'part8']
    }

    # 初始化拆解得到的零件数量
    disassembled_parts = {f'part{i + 1}': 0 for i in range(len(n_parts))}

    # 第一轮
    # 零配件的购买及检测成本
    costs = []
    num_of_spares = []
    for i in range(len(n_parts)):
        part_key = f'part{i + 1}'
        cost, num_of_spare = calculate_cost_and_spare(part_key, x_values[i], n_parts[i], scenarios)
        # print(f'零配件{i + 1}的购买及检测成本为{cost}，剩余数量为{num_of_spare}')
        costs.append(cost)
        num_of_spares.append(num_of_spare)

    # 零配件使用后剩余的零件数量
    num_of_semi_products = min(num_of_spares)
    for i in range(len(n_parts)):
        n_parts[i] -= num_of_semi_products

    # 半成品装配及检测拆解的成本,以及实际产生的半成品数量，拆解的不合格半成品数量
    semi_product_costs = []
    semi_product_nums = []
    for semi_product_num in range(1, 4):
        cost_semi, num_of_semi_products_actual, num_of_chaijie, _ = cost_semi_product(
            [num_of_semi_products] * 3, semi_product_num, x_values, y_values[semi_product_num - 1], z_values[semi_product_num - 1], scenarios)
        
        # 更新拆解得到的零件数量
        parts_to_add = semi_product_parts[semi_product_num]
        for part in parts_to_add:
            disassembled_parts[part] += num_of_chaijie
        
        semi_product_costs.append(cost_semi)
        semi_product_nums.append(num_of_semi_products_actual)

    # 最少的半成品数量
    num_of_semi_products_actual = min(semi_product_nums)

    # 成品装配及检测拆解的成本,以及实际产生的成品数量，拆解的不合格成品数量
    cost3, num_of_finished_products, num_of_chaijie, disassembled_parts_ = cost_finished_product(
        [num_of_semi_products_actual] * 3, x_values, y_values, p_value, q_value, scenarios)
    
    # 更新拆解得到的零件数量
    parts_to_add= ['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8']
    for part in parts_to_add:
        disassembled_parts[part] += num_of_chaijie

    # 不合格成品调换损失，以及召回的不合格成品数量
    cost4, num_of_loss = cost_loss(x_values, y_values, p_value, num_of_finished_products, scenarios)

    # 总成本
    total_cost = sum(costs) + sum(semi_product_costs) + cost3 + cost4 + r_value * num_of_loss * scenarios['finished_product_disassembly_cost']

    # 总利润
    total_profit = scenarios['market_price'] * (num_of_finished_products - num_of_loss) - total_cost



    # 拆解所得零件和召回的零件进行第二轮，默认全部检测且不再进行拆解
    for i in range(len(n_parts)):
         n_parts[i] += disassembled_parts.get(f'part{i + 1}', 0) + r_value * num_of_loss

    if all(n > 0 for n in n_parts):
        costs = []
        num_of_spares = []
        for i in range(len(n_parts)):
            part_key = f'part{i + 1}'
            cost, num_of_spare = calculate_cost_and_spare(part_key, 1, n_parts[i], scenarios)
            costs.append(cost)
            num_of_spares.append(num_of_spare)

        num_of_semi_products = min(num_of_spares)
        for i in range(len(n_parts)):
            n_parts[i] -= num_of_semi_products

        # 半成品装配及检测拆解的成本,以及实际产生的半成品数量，拆解的不合格半成品数量
        semi_product_costs = []
        semi_product_nums = []
        for semi_product_num in range(1, 4):
            cost_semi, num_of_semi_products_actual, num_of_chaijie, disassembled_parts_ = cost_semi_product(
                [num_of_semi_products] * 3, semi_product_num, [1] * 8, [1] * 3, 0, scenarios)
            
            # 更新拆解得到的零件数量
            parts_to_add = semi_product_parts[semi_product_num]
            for part in parts_to_add:
                disassembled_parts[part] += num_of_chaijie
            
            semi_product_costs.append(cost_semi)
            semi_product_nums.append(num_of_semi_products_actual)

        # 最少的半成品数量
        num_of_semi_products_actual = min(semi_product_nums)

        # 成品装配及检测拆解的成本,以及实际产生的成品数量，拆解的不合格成品数量
        cost3, num_of_finished_products, num_of_chaijie, disassembled_parts_ = cost_finished_product(
            [num_of_semi_products_actual] * 3, [1] * 8, [1] * 3, 1, 0, scenarios)
        
        

        # 不合格成品调换损失，以及召回的不合格成品数量
        cost4, num_of_loss = cost_loss([1] * 8, [1] * 3, 1, num_of_finished_products, scenarios)

        # 总成本
        total_cost = sum(costs) + sum(semi_product_costs) + cost3 + cost4

        # 更新总利润
        total_profit += scenarios['market_price'] * (num_of_finished_products - num_of_loss) - total_cost

    return total_profit


def main():
    max_profit = float('-inf')
    best_strategy = None
    
    # 假设的零件数量
    n_parts = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

    # 生成所有可能的参数组合
    x_values_combinations = list(itertools.product([0, 1], repeat=8))
    y_values_combinations = list(itertools.product([0, 1], repeat=3))
    z_values_combinations = list(itertools.product([0, 1], repeat=3))
    p_value_combinations = [0, 1]
    q_value_combinations = [0, 1]
    r_value_combinations = [0, 1]

    all_strategies = []

    # 计算所有策略的利润
    for x_values in x_values_combinations:
        for y_values in y_values_combinations:
            for z_values in z_values_combinations:
                for p_value in p_value_combinations:
                    for q_value in q_value_combinations:
                        for r_value in r_value_combinations:
                            # 零件数量重置
                            n_parts = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
                            current_profit = profit(x_values, y_values, z_values, p_value, q_value, r_value, n_parts, scenarios)
                            all_strategies.append((x_values, y_values, z_values, p_value, q_value, r_value, current_profit))
                            if current_profit > max_profit:
                                max_profit = current_profit
                                best_strategy = (x_values, y_values, z_values, p_value, q_value, r_value)

   #保存所有策略及其利润
    with open('result/Q3.txt', 'w') as f:
        for strategy in all_strategies:
            
            x_values, y_values, z_values, p_value, q_value, r_value, current_profit = strategy
            f.write(f"strategy: X={x_values}, Y={y_values}, Z={z_values}, P={p_value}, Q={q_value}, R={r_value}, profit: {current_profit}\n")
        
        print(f"\nmax_profit: {max_profit}\n")
        print(f"best_strategy: X={best_strategy[0]}, Y={best_strategy[1]}, Z={best_strategy[2]}, P={best_strategy[3]}, Q={best_strategy[4]}, R={best_strategy[5]}")

if __name__ == "__main__":
    main()