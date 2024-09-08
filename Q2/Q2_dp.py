# 导入库
import numpy as np

# 定义情景数据
scenarios = [
    {
        "part1_defect_rate": 0.10, "part1_cost": 4, "part1_test_cost": 2,
        "part2_defect_rate": 0.10, "part2_cost": 18, "part2_test_cost": 3,
        "product_defect_rate": 0.10, "assembly_cost": 6, "product_test_cost": 3,
        "market_price": 56, "replacement_loss": 6, "disassembly_cost": 5,
        "num_parts1": 100, "num_parts2": 100
    },
    {
        "part1_defect_rate": 0.20, "part1_cost": 4, "part1_test_cost": 2,
        "part2_defect_rate": 0.20, "part2_cost": 18, "part2_test_cost": 3,
        "product_defect_rate": 0.20, "assembly_cost": 6, "product_test_cost": 3,
        "market_price": 56, "replacement_loss": 6, "disassembly_cost": 5,
        "num_parts1": 100, "num_parts2": 100
    },
    {
        "part1_defect_rate": 0.10, "part1_cost": 4, "part1_test_cost": 2,
        "part2_defect_rate": 0.10, "part2_cost": 18, "part2_test_cost": 3,
        "product_defect_rate": 0.10, "assembly_cost": 6, "product_test_cost": 3,
        "market_price": 56, "replacement_loss": 30, "disassembly_cost": 5,
        "num_parts1": 100, "num_parts2": 100
    },
    {
        "part1_defect_rate": 0.20, "part1_cost": 4, "part1_test_cost": 1,
        "part2_defect_rate": 0.20, "part2_cost": 18, "part2_test_cost": 1,
        "product_defect_rate": 0.20, "assembly_cost": 6, "product_test_cost": 2,
        "market_price": 56, "replacement_loss": 30, "disassembly_cost": 5,
        "num_parts1": 100, "num_parts2": 100
    },
    {
        "part1_defect_rate": 0.10, "part1_cost": 4, "part1_test_cost": 8,
        "part2_defect_rate": 0.20, "part2_cost": 18, "part2_test_cost": 1,
        "product_defect_rate": 0.10, "assembly_cost": 6, "product_test_cost": 2,
        "market_price": 56, "replacement_loss": 10, "disassembly_cost": 5,
        "num_parts1": 100, "num_parts2": 100
    }
]

# 计算每个零件的总成本
def batch_cost(num_parts, defect_rate, part_cost, test_cost, test):
    if test:
        # 如果测试零件，去除次品，计算测试后的成本
        return num_parts * part_cost + num_parts * test_cost * (1 - defect_rate)
    else:
        # 不测试，所有零件直接使用
        return num_parts * part_cost

# 计算成品利润，包括检测、拆解的选择
def product_profit(p1_defect, p2_defect, product_defect, num_parts1, num_parts2, 
                   assembly_cost, product_test_cost, disassembly_cost, 
                   replacement_loss, market_price, test_product, disassemble):
    # 计算组合次品率
    combined_defect_rate = 1 - (1 - p1_defect) ** num_parts1 * (1 - p2_defect) ** num_parts2*(1 - product_defect)
    
    # 计算收入
    if test_product:
        # 检测后只卖合格产品
        revenue = (1 - combined_defect_rate) * market_price
        if disassemble:
            # 进行拆解，计算拆解成本
            disassembly_cost_total = combined_defect_rate * disassembly_cost
            return revenue - (assembly_cost + product_test_cost + disassembly_cost_total)
        else:
            # 不拆解，承担次品带来的损失
            return revenue - (assembly_cost + product_test_cost + combined_defect_rate * replacement_loss)
    else:
        # 不检测成品，直接进入市场，承担次品的市场损失
        revenue = (1 - combined_defect_rate) * market_price
        return revenue - (assembly_cost + combined_defect_rate * replacement_loss)

# 动态规划或遍历计算最大利润
def calculate_max_profit(scenario):
    num_parts1 = scenario["num_parts1"]
    num_parts2 = scenario["num_parts2"]
    
    p1_defect = scenario["part1_defect_rate"]
    p1_cost = scenario["part1_cost"]
    p1_test_cost = scenario["part1_test_cost"]
    
    p2_defect = scenario["part2_defect_rate"]
    p2_cost = scenario["part2_cost"]
    p2_test_cost = scenario["part2_test_cost"]
    
    product_defect = scenario["product_defect_rate"]
    assembly_cost = scenario["assembly_cost"]
    product_test_cost = scenario["product_test_cost"]
    
    market_price = scenario["market_price"]
    replacement_loss = scenario["replacement_loss"]
    disassembly_cost = scenario["disassembly_cost"]
    
    max_profit = float("-inf")
    best_strategy = None
    
    # 遍历所有策略组合
    for x1 in [0, 1]:  # 零配件1检测与否
        for x2 in [0, 1]:  # 零配件2检测与否
            for y in [0, 1]:  # 成品检测与否
                for z in [0, 1]:  # 成品拆解与否
                    # 计算零配件成本
                    part1_total_cost = batch_cost(num_parts1, p1_defect, p1_cost, p1_test_cost, x1)
                    part2_total_cost = batch_cost(num_parts2, p2_defect, p2_cost, p2_test_cost, x2)
                    
                    # 计算成品利润
                    product_total_profit = product_profit(p1_defect if x1 else p1_defect, 
                                                          p2_defect if x2 else p2_defect,
                                                          product_defect, 
                                                          num_parts1, num_parts2,
                                                          assembly_cost, 
                                                          product_test_cost, 
                                                          disassembly_cost, 
                                                          replacement_loss, 
                                                          market_price, 
                                                          y, z)
                    
                    # 总利润
                    total_profit = product_total_profit - (part1_total_cost + part2_total_cost)
                    
                    # 找到最大利润
                    if total_profit > max_profit:
                        max_profit = total_profit
                        best_strategy = (x1, x2, y, z)

    return max_profit, best_strategy

# 计算每个情景下的最大利润和最佳策略
for i, scenario in enumerate(scenarios):
    max_profit, best_strategy = calculate_max_profit(scenario)
    print(f"情景 {i+1}: 最大利润 = {max_profit}, 最优策略 = 零配件1检测 {best_strategy[0]}, 零配件2检测 {best_strategy[1]}, 成品检测 {best_strategy[2]}, 拆解 {best_strategy[3]}")
