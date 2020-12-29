# -*- coding:utf-8 -*-
# author: xkey
# 模拟退火算法（Simulated Annealing， SA）
import numpy as np
import matplotlib.pyplot as plt 
import math, time, random
from extendTSP import *
'''
注：先在extendTSP.py 中使用随机函数生成实例填入
跑实例修改下述cases的下标即可
TODO list: 模块化代码
'''

if __name__ == '__main__': 
    ##### 参数及相关数据初始化 #####
    # 初始化城市实例
    city_position, goods_class, city_class = extendTSP_cases[4]
    city_num = len(city_position)             # 城市数目
    goods_num = len(set(goods_class))         # 商品种类数目
    distance = record_distance(city_position) # 得到距离矩阵

    alpha = 0.99       # 降温系数
    t_range = (1, 100) # 温度范围
    iter_num = 1000    # 迭代次数
    t = t_range[1]     # 初始化温度 100

    start_time = time.time()
    ##### 生成随机初始解 #####
    new_solution = []
    for x in city_class:
        new_solution.append(x[random.randrange(len(x))])
    random.shuffle(new_solution)

    new_value = cal_cost(distance, new_solution, goods_num) # 计算价值
    
    # 当前的解
    current_solution = new_solution.copy()
    current_value = [new_value]         # 存放每次迭代后，当前最佳路径长度

    best_solution = new_solution.copy() # 最优路径
    bestvalue = new_value               # 最优值
    best_value = [bestvalue]            # 存放每次迭代后，最佳路径长度
    print(bestvalue, new_value, current_value)

    ##### 开始迭代 #####
    while t > t_range[0]: # 温度范围
        for i in np.arange(iter_num):
            # 从邻域选择新解 - 同一类城市交换 + 现有的两交换 + 现有的三交换
            seed = np.random.rand()
            if seed > 0.3 and seed < 0.7: # 随机交换卖同一类商品城市的两个节点
                while True:
                    goods = random.randrange(goods_num)
                    if len(city_class[goods]) > 1: # 有多个城市售卖同类商品
                        break
                for index, city in enumerate(city_class[goods]):
                    if city in new_solution: # 找到当前解中售卖该类商品的城市
                        loc = new_solution.index(city)
                        while True:
                            tmp = random.randrange(len(city_class[goods]))
                            if tmp != index:
                                break
                        new_solution[loc] = city_class[goods][tmp] # 随机替换

            elif seed > 0.7: # 交换路径中的这2个节点的顺序
                while True:# 产生两个不同的随机数
                    loc1 = np.int(np.ceil(np.random.rand()*(goods_num-1)))
                    loc2 = np.int(np.ceil(np.random.rand()*(goods_num-1)))
                    if loc1 != loc2:
                        break
                new_solution[loc1], new_solution[loc2] = new_solution[loc2], new_solution[loc1] # 交换
            else: # 三交换
                while True:
                    loc1 = np.int(np.ceil(np.random.rand()*(goods_num-1)))
                    loc2 = np.int(np.ceil(np.random.rand()*(goods_num-1))) 
                    loc3 = np.int(np.ceil(np.random.rand()*(goods_num-1)))
    
                    if((loc1 != loc2)&(loc2 != loc3)&(loc1 != loc3)):
                        break
    
                # 将loc1, 2, 3按大小排序
                if loc1 > loc2:
                    loc1,loc2 = loc2,loc1
                if loc2 > loc3:
                    loc2,loc3 = loc3,loc2
                if loc1 > loc2:
                    loc1,loc2 = loc2,loc1
    
                # 将[loc1,loc2)区间的数据插入到loc3之后
                tmplist = new_solution[loc1:loc2].copy()
                new_solution[loc1:loc3-loc2+1+loc1] = new_solution[loc2:loc3+1].copy()
                new_solution[loc3-loc2+1+loc1:loc3+1] = tmplist.copy()  

            new_value = cal_cost(distance, new_solution, goods_num) # 计算价值
            if new_value < current_value[-1]: # 直接接受该解
                # 更新 current_solution 和 best_solution
                current_value.append(new_value)
                current_solution = new_solution.copy()

                if new_value < bestvalue:
                    bestvalue = new_value
                    best_solution = new_solution.copy()
                    best_value.append(bestvalue)
                else:
                    best_value.append(best_value[-1])
            else: # 按一定的概率接受该解 (Metroplis准则)
                if np.random.rand() < np.exp(-(new_value-current_value[-1])/t):
                    current_value.append(new_value)
                    current_solution = new_solution.copy()
                    best_value.append(best_value[-1])
                else: # 否则恢复之前状态
                    new_solution = current_solution.copy()
        
        t = alpha * t # 降温
        # print(t, current_value[-1], bestvalue)

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