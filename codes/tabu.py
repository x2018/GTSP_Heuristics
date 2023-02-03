# -*- coding:utf-8 -*-
# author: xkey
# 禁忌搜索
# 邻域 邻域动作 禁忌表 候选集合 评价函数 特赦规则
import numpy as np
import matplotlib.pyplot as plt 
import math, time, random
from extendTSP import *
'''
注:先在extendTSP.py 中使用随机函数生成实例填入
跑实例修改下述cases的下标即可
TODO list: 模块化代码
消除解为[0...0]的情况 √ (对候选集做检查)
'''

if __name__ == '__main__': 
    ##### 参数及相关数据初始化 #####
    # 初始化城市实例
    city_position, goods_class, city_class = extendTSP_cases[4]
    city_num = len(city_position)             # 城市数目
    goods_num = len(set(goods_class))         # 商品种类数目
    distance = record_distance(city_position) # 得到距离矩阵

    iter_num = 1000       # 迭代次数
    tabu_list = []        # 禁忌表
    tabu_time = []        # 禁忌时间表
    current_tabu_num = 0  # 当前禁忌对象数量
    tabu_limit = 50       # 特赦规则(50次)

    # 候选集
    candidate_num = city_num
    candidate = []
    candidate_value = []

    start_time = time.time()
    ##### 生成随机初始解 #####
    current_solution = []
    for x in city_class:
        current_solution.append(x[random.randrange(len(x))])
    random.shuffle(current_solution)

    current_value = [cal_cost(distance, current_solution, goods_num)] # 计算价值

    best_solution = current_solution.copy() # 最优路径
    bestvalue = current_value[-1]           # 最优值
    print(bestvalue, current_value[-1])
    best_value = [bestvalue] # 记录迭代过程中的最优解

    ##### 开始迭代 #####
    for i in range(iter_num):
        ##### 得到邻域 候选解 #####
        # 初始化本轮候选集
        candidate = []
        candidate_value = []
        temp = 0
        # 随机选取邻域
        while True:
            # 从邻域选择新解 - 同一类城市交换 + 现有的两交换
            seed = np.random.rand()
            if seed > 0.5: # 随机交换卖同一类商品城市的两个节点
                goods = random.randrange(goods_num)
                if len(city_class[goods]) == 1:
                    continue
                for index, city in enumerate(city_class[goods]):
                    if city in current_solution:
                        loc = current_solution.index(city)
                        while True:
                            tmp = random.randrange(len(city_class[goods]))
                            if tmp != index:
                                break
                        candidate.append(current_solution.copy())
                        candidate[temp][loc] = city_class[goods][tmp]
            else: # 交换两个城市
                current = random.sample(range(0, goods_num), 2)
                candidate.append(current_solution.copy())
                candidate[temp][current[0]], candidate[temp][current[1]] = candidate[temp][current[1]], candidate[temp][current[0]] # 交换

            if len(candidate) == temp:
                continue
            # 若不在禁忌表中则放入候选集
            if candidate[temp] not in tabu_list:
                candidate_value.append(cal_cost(distance, candidate[temp], goods_num))
                temp += 1
            else:
                candidate.pop()
            if temp >= candidate_num:
                break

        # 得到候选解中的最优解
        candidate_best = min(candidate_value)
        best_index = candidate_value.index(candidate_best)
        
        current_value.append(candidate_best)
        current_solution = candidate[best_index].copy()
        
        # 与当前最优解进行比较 
        if current_value[-1] < bestvalue:
            best_value.append(current_value[-1])
            bestvalue = current_value[-1]
            best_solution = current_solution.copy()
        else:
            best_value.append(best_value[-1])
        
        # 将最优的加入禁忌表
        tabu_list.append(candidate[best_index])
        tabu_time.append(tabu_limit)
        current_tabu_num += 1

        ##### 更新禁忌表以及禁忌期限 #####
        del_num = 0
        temp = [0 for col in range(goods_num)]
        # 更新步长
        tabu_time = [x-1 for x in tabu_time]
        # 如果达到期限，释放
        for i in range(current_tabu_num):
            if tabu_time[i] == 0:
                del_num += 1
                tabu_list.pop(i)
            
        current_tabu_num -= del_num        
        while 0 in tabu_time:
            tabu_time.remove(0)

    end_time = time.time()
    print("time consuming: %lf s" % (end_time- start_time))

    ##### 显示收敛情况 #####
    print(best_solution, bestvalue)
    plt.plot(best_value)
    plt.title("best value")
    plt.ylabel("path cost")
    plt.xlabel("t")
    plt.show()
    # 当前最优的迭代结果
    plt.plot(current_value)
    plt.title("current best value")
    plt.ylabel("path cost")
    plt.xlabel("t")
    plt.show()

    ##### 显示路线结果 #####
    show_result(best_solution, city_position, city_class)