# -*- coding:utf-8 -*-
# 广义TSP - 问题定义
# 给出5个广义TSP实例((9, 5), (17, 11), (24, 15), (31, 16), (39, 25))
# (城市数量， 商品数量)
# 待定义：城市位置(带权完全图 G<V,E> 或 直接二维坐标定义) 和 城市所售商品类别
import random, math
import numpy as np
import matplotlib.pyplot as plt 

###### 随机生成的5个实例 #######
extendTSP_cases = []
##### RAND_CASE 1 #####
# city_num = 9
# goods_num = 5
# tuple: (city_position, goods_class, city_class)
extendTSP_cases.append(([(0, 14), (5, 4), (9, 9), (12, 19), (1, 7), (14, 8), (10, 17), (10, 13), (2, 0)], 
                        [2, 3, 0, 2, 1, 1, 4, 2, 2], 
                        [[2], [4, 5], [0, 3, 7, 8], [1], [6]]))

##### RAND_CASE 2 #####
# city_num = 17
# goods_num = 11
# tuple: (city_position, goods_class, city_class)
extendTSP_cases.append(([(8, 2), (5, 18), (11, 12), (11, 16), (0, 15), (19, 8), (0, 8), (6, 5), (4, 2), (4, 4), (12, 5), (11, 9), (8, 16), (7, 13), (17, 13), (19, 18), (16, 16)], 
                        [1, 9, 0, 7, 3, 4, 5, 3, 3, 6, 10, 3, 2, 8, 0, 2, 2], 
                        [[2, 14], [0], [12, 15, 16], [4, 7, 8, 11], [5], [6], [9], [3], [13], [1], [10]]))

##### RAND_CASE 3 #####
# city_num = 24
# goods_num = 15
# tuple: (city_position, goods_class, city_class)
extendTSP_cases.append(([(1, 18), (0, 17), (17, 2), (11, 2), (8, 10), (19, 3), (4, 15), (12, 0), (6, 10), (1, 1), (13, 0), (0, 5), (11, 5), (13, 5), (17, 18), (7, 14), (1, 8), (6, 2), (17, 16), (2, 13), (4, 3), (10, 17), (3, 10), (3, 3)], 
                        [4, 11, 8, 10, 14, 12, 12, 8, 7, 5, 0, 2, 9, 12, 6, 11, 9, 3, 1, 1, 8, 7, 8, 13], 
                        [[10], [18, 19], [11], [17], [0], [9], [14], [8, 21], [2, 7, 20, 22], [12, 16], [3], [1, 15], [5, 6, 13], [23], [4]]))

##### RAND_CASE 4 #####
# city_num = 31
# goods_num = 16
# tuple: (city_position, goods_class, city_class)
extendTSP_cases.append(([(9, 7), (8, 19), (19, 2), (13, 11), (4, 12), (13, 12), (3, 3), (18, 9), (19, 5), (13, 16), (5, 15), (15, 14), (13, 13), (8, 0), (14, 15), (4, 17), (14, 8), (19, 17), (17, 6), (14, 2), (18, 14), (18, 17), (18, 3), (6, 10), (0, 18), (15, 16), (14, 5), (3, 17), (3, 1), (15, 4), (12, 14)], 
                        [11, 10, 8, 6, 5, 12, 10, 14, 2, 4, 10, 1, 3, 13, 15, 9, 15, 8, 10, 4, 9, 1, 3, 5, 7, 10, 0, 15, 11, 0, 0], 
                        [[26, 29, 30], [11, 21], [8], [12, 22], [9, 19], [4, 23], [3], [24], [2, 17], [15, 20], [1, 6, 10, 18, 25], [0, 28], [5], [13], [7], [14, 16, 27]]))

##### RAND_CASE 5 #####
# city_num = 39
# goods_num = 25
# tuple: (city_position, goods_class, city_class)
extendTSP_cases.append(([(5, 13), (10, 5), (11, 9), (18, 9), (8, 3), (0, 10), (19, 17), (9, 17), (11, 12), (18, 3), (3, 4), (17, 11), (5, 2), (6, 12), (18, 7), (1, 7), (11, 4), (13, 1), (6, 15), (0, 3), (19, 2), (18, 10), (14, 1), (16, 3), (4, 3), (3, 15), (19, 11), (13, 13), (4, 4), (2, 9), (15, 7), (2, 15), (7, 18), (13, 14), (14, 0), (15, 15), (17, 3), (13, 16), (15, 9)], 
                        [11, 7, 2, 21, 11, 16, 6, 21, 9, 8, 9, 24, 23, 0, 6, 17, 18, 18, 5, 15, 4, 10, 12, 24, 12, 7, 20, 22, 3, 11, 19, 0, 4, 14, 13, 21, 1, 12, 9], 
                        [[13, 31], [36], [2], [28], [20, 32], [18], [6, 14], [1, 25], [9], [8, 10, 38], [21], [0, 4, 29], [22, 24, 37], [34], [33], [19], [5], [15], [16, 17], [30], [26], [3, 7, 35], [27], [12], [11, 23]]))

##### 按所卖商品来归类城市 #####
# 按照商品编号来归类城市编号(0开始)
def classify_city(goods_class):
    goods_num = len(set(goods_class))
    city_class = [[] for i in range(goods_num)]
    for index, goods in enumerate(goods_class):
        city_class[goods].append(index)
    return city_class

##### 随机生成广义TSP实例 #####
# params: 城市数量， 商品数量
# return: (city_position, goods_class, city_class)
def extendTSP_generate(city_num, goods_num, x_range = 20, y_range = 20):
    # 坐标范围(20, 20)
    city_position = []
    goods_class = [x for x in range(goods_num)]
    while len(city_position) < city_num:
        rand_position = (random.randrange(x_range), random.randrange(y_range))
        if rand_position not in city_position:
            city_position.append(rand_position)
    while len(goods_class) < city_num:
        goods_class.append(random.randrange(goods_num))
    random.shuffle(goods_class) # 随机打乱标号
    city_class = classify_city(goods_class) # 归类城市
    return (city_position, goods_class, city_class)

##### 记录各个城市间的距离 #####
# 生成一个距离矩阵
def record_distance(city_position):
    num = len(city_position) # 城市数量
    distance = np.zeros((num, num)) # 城市两两之间的距离
    for i in range(num):
        for j in range(i, num):
            distance[i][j] = distance[j][i] = math.sqrt(pow(city_position[i][0] - city_position[j][0], 2) + pow(city_position[i][1] - city_position[j][1], 2))
    return distance

##### 计算当前解的消耗 #####
def cal_cost(distance, solution, goods_num):
    cost = 0
    for j in range(goods_num-1):
        cost += distance[solution[j]][solution[j+1]]
    cost += distance[solution[0]][solution[goods_num - 1]]
    return cost

##### 获取绘图散点信息 #####
def get_plot_points(city_class, city_position):
    x = [] # 横坐标
    y = [] # 纵坐标
    c = [] # 颜色
    for index, item in enumerate(city_class): # 根据商品种类划分点
        x.append([city_position[i][0] for i in item])
        y.append([city_position[i][1] for i in item])
        c.append(colors[index])
    return x,y,c

##### 绘图 #####
def show_result(best_solution, city_position, city_class):
    # 所有点 - 根据商品种类划分颜色
    # x = [x[0] for x in city_position]
    # y = [y[1] for y in city_position]
    x,y,c = get_plot_points(city_class, city_position)

    # 结果上的点
    result = []
    for i in best_solution:
        result.append(city_position[i])
    result.append(city_position[best_solution[0]])
    x_result = [x[0] for x in result]
    y_result = [y[1] for y in result]
    print(result, best_solution)

    for index, x_item in enumerate(x):
        plt.scatter(x_item, y[index], label='Location', color = c[index]) # 散点 edgecolors = 'coral'
    plt.plot(x_result, y_result, label = 'result', color = 'black', alpha = 0.2, marker = '*', markersize=10) # 路径
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title("best result")
    # plt.legend()
    plt.show()

##### 检查解是否有效 #####
def check_valid(goods_class, solution):
    # 检查走过的城市是否与商品种类数量相同
    if len(solution) != len(set(goods_class)): 
        return False
    # 检查是否买重
    else:
        goods_bought = []
        for i in solution:
            if goods_class[i] in goods_bought:
                return False
            else:
                goods_bought.append(goods_class[i])
    return True

# colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
# random.shuffle(colors)
# TODO:找几个颜色区分明显的 √
colors=['black', 'blue', 'green', 'red', 'yellow', 'orange','purple', 'darkblue','lightblue','gold', 'lime', 'maroon',
            'olive', 'silver', 'orchid', 'salmon', 'tomato', 'yellowgreen', 'rosybrown', 'plum', 'peru', 'tan', 'sienna', 'saddlebrown',
            'palevioletred']
'''
{
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
'''