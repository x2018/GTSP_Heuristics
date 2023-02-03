# -*- coding:utf-8 -*-
# author: xkey
# 蚁群算法(Ant Clony Optimization, ACO)
# 初始种群 信息素浓度 挥发系数
import numpy as np
import matplotlib.pyplot as plt 
import math, time, random
from extendTSP import *
'''
注:先在extendTSP.py 中使用随机函数生成实例填入
跑实例修改下述cases的下标即可
TODO list: 模块化代码
'''

if __name__ == '__main__':
    ##### 参数及相关数据初始化 #####
    # 初始化城市实例
    city_position, goods_class, city_class = extendTSP_cases[4] # extendTSP_generate(39, 25)
    city_num = len(city_position)             # 城市数目
    goods_num = len(set(goods_class))         # 商品种类数目
    distance = record_distance(city_position) # 得到距离矩阵

    ant_num = 60  # 蚂蚁个数
    alpha = 1     # 信息素重要程度因子
    beta = 5      # 启发函数重要程度因子
    rho = 0.1     # 信息素的挥发系数
    Q = 1         # 信息素增加强度系数

    iter_num = 100  # 迭代次数

    start_time = time.time()
    eta_table = 1.0 / ( distance + np.diag([1e10] * city_num)) # 初始化启发矩阵
    pheromone_table = np.ones((city_num, city_num))            # 初始化信息素矩阵

    path_table = np.zeros((ant_num, goods_num)).astype(int)    # 路径表

    current_value = []    # 存放每次迭代后，当前最佳路径长度
    # avg_value = []      # 存放每次迭代后，路径的平均长度
    best_value = []       # 存放每次迭代后，最佳路径长度
    best_solution = None  # 最优路径

    # 开始迭代
    for iter in range(iter_num):
        ##### 初始化蚂蚁起始位置 #####
        if ant_num <= city_num: # 蚂蚁数目不比城市多
            path_table[:, 0] = np.random.permutation(range(city_num))[:ant_num]
        else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
            inited_ants = 0       # 已经分配了的蚂蚁数量
            remain_ants = ant_num # 剩余的蚂蚁数量
            while remain_ants > city_num:
                remain_ants -= city_num
                path_table[inited_ants:inited_ants + city_num, 0] = np.random.permutation(range(city_num))[:]
                inited_ants += city_num
            path_table[inited_ants:, 0] = np.random.permutation(range(city_num))[:remain_ants]

        length = np.zeros(ant_num)  # 记录每只蚂蚁走过的路径长度

        ##### 计算蚂蚁到下一个城市的概率 #####
        for i in range(ant_num):
            visiting = path_table[i, 0]                    # 当前所在的城市
            unvisited = [city for city in range(city_num)] # 未访问的城市集合
            unvisited.remove(visiting)                     # 删除已经访问过的城市元素
            goods_bought = [goods_class[visiting]]         # 记录已买的商品号

            for j in range(1, goods_num):  # 循环goods_num-1次，访问剩余的所有goods_num-1个城市来购买全部种类的商品
                ##### 轮盘法选择下一个城市 #####
                trans_prob = np.zeros(len(unvisited)) # 转移概率列表

                # 计算转移概率(信息素和启发函数共同作用)
                for k in range(len(unvisited)):
                    trans_prob[k] = np.power(pheromone_table[visiting][unvisited[k]], alpha) \
                                * np.power(eta_table[visiting][unvisited[k]], beta)

                # 如果按照轮盘没有结果的话则重新构造放宽限制
                k = -1
                t = 1
                const_trans_cumsum_prob = (trans_prob / sum(trans_prob)).cumsum()     # 概率累积求和
                trans_cumsum_prob = const_trans_cumsum_prob.copy() - np.random.rand() # 随机生成下个城市的转移概率，再用区间比较
                while True:
                    # 寻找下一个要访问的城市
                    for index, prob in enumerate(trans_cumsum_prob):
                        if prob >= 0:
                            if goods_class[unvisited[index]] not in goods_bought:
                                k = unvisited[index]
                                break
                    if k == -1:
                        trans_cumsum_prob = const_trans_cumsum_prob.copy()       # 提速? 似乎有点儿作用
                        trans_cumsum_prob -= np.random.rand() * math.pow(0.8, t) # 随机生成下个城市的转移概率，放宽条件
                        t += 1
                    else:
                        break
                
                path_table[i, j] = k # 记录蚂蚁 i 走的第 j 座城市
                unvisited.remove(k)  # 从未访问城市列表移除城市 k
                goods_bought.append(goods_class[k])

                length[i] +=  distance[visiting][k] # 增加当前路径长度
                visiting = k                        # 将 k 设为当前所在城市
            
            length[i] +=  distance[visiting][path_table[i, 0]] # 增加当前路径长度(最后一个城市到第一个城市的距离)
        
        print("iter:", str(iter) + '/' + str(iter_num)) # , str(i) + '/' + str(ant_num))
        # 更新平均路径
        # avg_value.append(length.mean()) # 记录本轮每只蚂蚁所走的平均路径

        ##### 求出最佳路径 #####
        current_value.append(length.min())
        if iter == 0: # 第一轮直接记录
            best_value.append(length.min())
            best_solution = path_table[length.argmin()].copy()
        else:         # 后面进行比较
            if length.min() > best_value[iter - 1]:
                best_value.append(best_value[-1])
            else:
                best_value.append(length.min())
                best_solution = path_table[length.argmin()].copy()

        ##### 更新信息素(一轮更新一次) #####
        delta_pheromone_table = np.zeros((city_num, city_num))
        for i in range(ant_num): # 更新所有的蚂蚁
            for j in range(goods_num - 1):            
                # 计算 第 i 只蚂蚁改变的信息素      
                # Q/d  其中 d 是从第 j 个城市到第 j + 1 个城市的距离
                delta_pheromone_table[path_table[i, j]][path_table[i, j + 1]] += Q / distance[path_table[i, j]][path_table[i, j + 1]]
            # 最后一个城市到第一个城市
            delta_pheromone_table[path_table[i, goods_num - 1]][path_table[i, 0]] += Q / distance[path_table[i, goods_num - 1]][path_table[i, 0]]
    
        ##### 信息素挥发 #####
        # p = (1 - 挥发速率) * 当前信息素 + 改变的信息素
        # 挥发一部分，增加一部分
        pheromone_table = (1 - rho) * pheromone_table + delta_pheromone_table

    end_time = time.time()
    print("time consuming: %lf s" % (end_time - start_time))

    ##### 显示收敛情况 #####
    best_solution = list(best_solution)
    print(best_solution, best_value[-1]) # , len(best_value)
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