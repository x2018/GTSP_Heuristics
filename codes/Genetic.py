# -*- coding:utf-8 -*-
# author: xkey
# 遗传算法 - 广义TSP问题
import numpy as np
import matplotlib.pyplot as plt 
import math, time, random
from extendTSP import *
'''
注：先在extendTSP.py 中使用随机函数生成实例填入
跑实例修改下述cases的下标即可
TODO list: 
1 模块化代码; 
2 优化(eg 极值优化 EO) √ 变异交叉时验证下是否使得解的消耗变小
'''

# 计算种群个体的价值(与适应度关联-价值的倒数)
def cal_popvalue(pop):
    global distance
    global goods_num
    pop_value = []
    for p in pop:
        pop_value.append(cal_cost(distance, p, goods_num))
    return pop_value

# 选择 - 轮盘赌
def selection(pop=None, fitvalue=None):
    global popsize
    global goods_num
    newpop = [[] for x in range(popsize)]
    
    # 构造轮盘
    totalfit = sum(fitvalue)
    p_fitvalue = [value / totalfit for value in fitvalue]
    p_fitvalue = np.cumsum(p_fitvalue) # 概率累积求和
    ms = np.sort(np.random.rand(popsize,1)) # 随机数从小到大排列

    fitin = 0
    newin = 0
    while newin < popsize: # 选择个体
        if ms[newin] < p_fitvalue[fitin]: # 如果随机数小于原适应则选择
            newpop[newin] = pop[fitin].copy()
            newin = newin + 1
        else:
            fitin = fitin + 1
    return newpop

# 去重(同类城市)
def rm_repeat(pop):
    global city_class
    repeat_status = False
    for item in city_class:
        if len(item) > 1:
            repeat_status = False
            for city in item:
                if city in pop:
                    if repeat_status == False:
                        repeat_status = True
                    else:
                        pop.remove(city)
    return pop

# 交换两个个体的基因片段+去重
def cross_pop(pop1, pop2):
    global goods_num
    index1 = np.random.randint(0, goods_num - 1)
    index2 = np.random.randint(index1, goods_num - 1)
    tempGene = pop2[index1:index2]  # 交叉的基因片段
    newGene = []
    len = 0
    for city in pop1:
        if len == index1:
            newGene.extend(tempGene)  # 插入基因片段
        if city not in tempGene:
            newGene.append(city)
        len += 1
    newGene = rm_repeat(newGene) # 去重
    return newGene

# 交叉
def crossover(pop=None, pc=None):
    global popsize
    global goods_num
    global distance
    newpop = [[] for x in range(popsize)]
    # 交换两个个体的基因片段+去重
    for i in range(popsize)[::2]: # 交叉p1,p2的部分基因片段
        newpop[i] = pop[i].copy()
        if i != popsize - 1:
            if np.random.rand() < pc:
                newpop[i] = cross_pop(pop[i], pop[i + 1])
                newpop[i + 1] = cross_pop(pop[i + 1], pop[i])
                # 优化 - 若消耗变大则不变异
                if cal_cost(distance, newpop[i], goods_num) > cal_cost(distance, pop[i], goods_num):
                    newpop[i] = pop[i]
            else:
                newpop[i + 1] = pop[i + 1]
    return newpop

# 变异
def mutation(pop=None, pm=None):
    global popsize
    global goods_num
    global city_class
    global distance
    newpop = [[] for x in range(popsize)]

    for i in range(popsize): 
        newpop[i] = pop[i].copy()
        if(random.random() < pm): # 变异概率
            seed = np.random.rand()
            if seed > 0.3 and seed < 0.7: # 随机交换卖同一类商品城市的两个节点
                while True:
                    goods = random.randrange(goods_num)
                    if len(city_class[goods]) > 1: # 有多个城市售卖同类商品
                        break
                for index, city in enumerate(city_class[goods]):
                    if city in newpop[i]: # 找到当前解中售卖该类商品的城市
                        loc = newpop[i].index(city)
                        while True:
                            tmp = random.randrange(len(city_class[goods]))
                            if tmp != index:
                                break
                        newpop[i][loc] = city_class[goods][tmp] # 随机替换

            elif seed > 0.7: # 交换路径中的这2个节点的顺序
                while True:# 产生两个不同的随机数
                    loc1 = np.int(np.ceil(np.random.rand()*(goods_num-1)))
                    loc2 = np.int(np.ceil(np.random.rand()*(goods_num-1)))
                    if loc1 != loc2:
                        break
                newpop[i][loc1], newpop[i][loc2] = newpop[i][loc2], newpop[i][loc1] # 交换
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
                tmplist = newpop[i][loc1:loc2].copy()
                newpop[i][loc1:loc3-loc2+1+loc1] = newpop[i][loc2:loc3+1].copy()
                newpop[i][loc3-loc2+1+loc1:loc3+1] = tmplist.copy() 
            
            # 优化 - 若消耗变大则不变异
            if cal_cost(distance, newpop[i], goods_num) > cal_cost(distance, pop[i], goods_num):
                newpop[i] = pop[i]
    return newpop

# 求最优个体
def best(pop=None, value=None):
    global popsize
    bestindividual = pop[0]
    bestvalue = value[0]

    for i in range(popsize):
        if value[i] < bestvalue:
            bestindividual = pop[i]
            bestvalue = value[i]

    return [bestindividual, bestvalue] # best_index

if __name__ == '__main__': 
    ##### 参数及相关数据初始化 #####
    # 初始化城市实例
    city_position, goods_class, city_class = extendTSP_cases[4]
    city_num = len(city_position)             # 城市数目
    goods_num = len(set(goods_class))         # 商品种类数目
    distance = record_distance(city_position) # 得到距离矩阵

    iter_num = 10000    # 迭代次数
    popsize = city_num  # 种群大小
    pc = 0.1            # 交叉概率
    pm = 0.8            # 变异概率

    start_time = time.time()
    # 初始种群
    pop = []
    pop_value = []
    ##### 生成随机初始种群 #####
    while True:
        current_solution = []
        for x in city_class:
            current_solution.append(x[random.randrange(len(x))])
        random.shuffle(current_solution)
        # if current_solution not in pop:
        pop.append(current_solution)
        if len(pop) >= popsize:
            break

    # 计算价值(与适应度关联，价值的倒数)
    pop_value = cal_popvalue(pop)
    bestindividual, bestvalue = best(pop, pop_value)
    current_value = [bestvalue]    # 存放每次迭代后，当前最佳路径长度
    best_value = [bestvalue]       # 存放每次迭代后，最佳路径长度
    best_solution = bestindividual # 最优路径
    fitvalue = [1/value for value in pop_value] # value 的倒数
    for i in range(iter_num):
        fitvalue = [1/value for value in pop_value] # 计算适应度
        newpop = selection(pop, fitvalue)           # 选择操作
        newpop = crossover(newpop, pc)              # 交叉操作
        newpop = mutation(newpop, pm)               # 变异操作
        pop = newpop                                # 更新种群
        
        # 寻找当前最优解
        pop_value = cal_popvalue(pop) # 计算价值
        bestindividual, bestvalue = best(pop, pop_value)
        current_value.append(bestvalue)
        # 是否为全程最优解
        if bestvalue < best_value[-1]: 
            best_solution = bestindividual
            best_value.append(bestvalue)
        else:
            best_value.append(best_value[-1])

    end_time = time.time()
    print("time consuming: %lf s" % (end_time - start_time))

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