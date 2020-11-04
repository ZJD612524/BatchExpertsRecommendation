# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import math as m
from math import gamma
from numpy import *
from copy import deepcopy
import random
from sklearn.cluster import KMeans
from sklearn import metrics  # 进行性能评估
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import time
start_time = time.time()
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

## 以k-means++算法挑选k值，并以其聚类结果作为初始解进行Sailfish聚类 ##
## 备选方案：更新solution，利用解中每个位置变为数字0-(k-1)的概率！
## 备选方案：先用k-means生成一组解，计算出概率进行初始化

def kmeans(df, k):
    kmeans = KMeans(n_init=20, init = 'k-means++', n_clusters = k, random_state=42, max_iter=400, precompute_distances=True)
    clusterResult = kmeans.fit(df).labels_
    #Silhouette.append(metrics.silhouette_score(x, labels, metric='euclidean'))
    return clusterResult
# ----------------------------------------------------------------------------------------------------

"""
The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm for solving
        constrained engineering optimization problems
"""


def get_global_best(pop, id_fitness):
    sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness], reverse=True)  # 按照轮廓系数从大到小排序
    #print(sorted_pop)
    best = 0
    if len(sorted_pop) > 0:
        best = deepcopy(sorted_pop[0])
    return best


##离散化
def discrete(label_old):
    label_new = []
    ##离散化到range(0,k)
    for i in range(0, len(df)):
        if label_old[i] in np.random.randint(0, k, 1):
            label_new.append(int(i))
        else:
            label_new.extend(np.random.randint(0, k, 1))
    ##保证每个类别都有数据
    for i in range(k):
        if i not in label_new:
            label_new[np.random.randint(0, len(label_new), 1)] = i
    return label_new


def get_initial_solution(k, df):
    initial_solution = kmeans(df, k)
    fitness = fitness_model(initial_solution, k)
    return [initial_solution, fitness]


def create_solution(domain_range, problem_size=0):
    solution = np.random.choice(k, problem_size)
    #solution = np.random.randint(domain_range[0], domain_range[1], problem_size)
    for i in range(k):
        if i not in solution:
            solution[np.random.randint(0, len(solution), 1)] = i
    fitness = fitness_model(solution, k)
    return [solution, fitness]


def center2(df, solution, k):
    """
    计算质心
    :param   group: 分组后样本
    :param   k: 类别个数
    :return: 计算得到的质心
    """
    clusterlabel = np.array(solution)
    clupoint = []
    dff = df.values
    for i in range(k):
        # 计算每个组的新质心
        idx = np.where(clusterlabel == i)
        sum = dff[idx].sum(axis=0)  # 按列相加
        avg_sum = sum / len(dff[idx])
        clupoint.append(avg_sum)
    clupoint = np.asarray(clupoint)
    return clupoint


def fitness_model(solution, k):  # 适应度函数轮廓系数
    """ Assumption that objective function always return the original value """
    global clupoint
    clupoint = center2(df, solution, k)
    #SSE_value = (sum(np.min(cdist(df, clupoint, 'euclidean'), axis=1))) / df.shape[0]
    silhouette = metrics.silhouette_score(df, solution, metric='euclidean')
    return silhouette


#遗传算法：交叉，将个体与当前领导进行交叉操作
def crossover(bestitem,item):
    #num = random.randint(0,len(item[ID_POS]))
    #pos = list(np.random.choice(len(item[ID_POS]), num ,replace=False))
    #print(pos)
    pos=random.randint(0,len(item[ID_POS]))
    for i in range(pos):
        item[ID_POS][pos]=deepcopy(bestitem[ID_POS][pos])
    item[ID_POS] = deepcopy(discrete(item[ID_POS]))
    return item

#遗传算法：变异，当前个体的位置随机变异
def mutation(item):
    L = [i for i in range(0,k)]
    pos=random.randint(0,len(item[ID_POS]))
    for i in range(pos):
        item[ID_POS][pos] = random.choice(L)
    item[ID_POS] = deepcopy(discrete(item[ID_POS]))
    return item

#精英策略
def elite(sf_pop_first,sf_pop_second):
    pop = sf_pop_first + sf_pop_second
    pop_rank = sorted(pop, key=lambda temp: temp[ID_FIT], reverse=True)
    return pop_rank[:len(sf_pop_first)]



def train(k):
    ## Setting parameters
    ##-----------------------------------------------------------------------------------------------
    ID_POS = 0
    ID_FIT = 1
    epoch = 1
    pop_size = 50  # SailFish pop size
    pp = 0.2  # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
    A = 6  # A = 4, 6,...
    epxilon = 0.01  # = 0.0001, 0.001
    problem_size = len(df)
    domain_range = [1, k]
    s_size = int(pop_size / pp)
    ##-----------------------------------------------------------------------------------------------

    # sf_pop = [create_solution(domain_range,problem_size) for _ in range(0, pop_size)]
    # s_pop = [create_solution(domain_range, problem_size) for _ in range(0, s_size)]
    sf_pop = [get_initial_solution(k,df) for _ in range(0, pop_size)]    # 用kmeans++的结果作为初始旗鱼
    s_pop = [get_initial_solution(k,df) for _ in range(0, s_size)]       # 用kmeans++的结果作为初始沙丁鱼
    a = get_global_best(sf_pop, ID_FIT)
    b = get_global_best(s_pop, ID_FIT)
    sf_gbest = deepcopy(a)
    s_gbest = deepcopy(b)
    # print(sf_pop)

    for epoch in range(0, epoch):
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        for i in range(0, pop_size):  # 循环每只旗鱼
            PD = 1 - len(sf_pop) / (len(sf_pop) + len(s_pop))  # 计算猎物的密度，论文公式（8）
            lamda_i = 2 * np.random.uniform() * PD - PD  # 公式（7），得到每一只旗鱼的λ值
            sf_pop[i][ID_POS] = s_gbest[ID_POS] - lamda_i * (np.random.uniform() *
                                                             (sf_gbest[ID_POS] + s_gbest[ID_POS]) / 2 - sf_pop[i][ID_POS])
            sf_pop[i][ID_POS] = discrete(sf_pop[i][ID_POS])

            

        ### 对旗鱼进行交叉和变异操作 ###
        sf_pop_first = deepcopy(sf_pop)  #第一代旗鱼种群
        sf_pop_second = deepcopy(sf_pop) #第二代旗鱼种群
        rand_list = np.random.randint(0, 2, len(sf_pop))
        for num in range(len(sf_pop)):
            if rand_list[num] is 0:
                sf_pop_second[num] = deepcopy(crossover(sf_gbest, sf_pop[num])) #与旗鱼交叉
                sf_pop_second[num]= deepcopy(mutation(sf_pop_second[num]))
            elif rand_list[num] is 1:
                sf_pop_second[num] = deepcopy(crossover(s_gbest, sf_pop[num]))  #与受伤沙丁鱼交叉
                sf_pop_second[num]= deepcopy(mutation(sf_pop_second[num]))
            
        ### 精英策略选取旗鱼种群 ###
        sf_pop = deepcopy(elite(sf_pop_first,sf_pop_second))
        
            
            
        for i in range(0, len(sf_pop)):  # 重新计算每条旗鱼的适应度函数
            sf_pop[i][ID_FIT] = [fitness_model(sf_pop[i][ID_POS], k)]

        ## Calculate AttackPower using Eq.(10)
        AP = A * (1 - 2 * (epoch + 1) * epxilon)
        if AP < 0.5:
            if AP < 0:
                alpha = 1
                beta = 1
            else:
                alpha = int(len(s_pop) * AP)
                beta = int(problem_size * AP)
            #print(alpha)
            #print(beta)

            ### Random choice number of sardines which will be updated their position
            #print(len(s_pop))
            list1 = np.random.choice(range(0, len(s_pop)), alpha)  
            for i in range(0, len(s_pop)):
                if i in list1:
                    #### Random choice number of dimensions in sardines updated
                    list2 = np.random.choice(range(0, problem_size), beta)
                    for j in range(0, problem_size):
                        if j in list2:
                            ##### Update the position of selected sardines and selected their dimensions
                            c = np.random.uniform() * (sf_gbest[ID_POS][j] - s_pop[i][ID_POS][j] + AP)
                            if 0 <= int(c) <= k-1:
                                s_pop[i][ID_POS][j] = int(c)
                            else:
                                s_pop[i][ID_POS][j] = np.random.randint(0, k, 1)
        else:
            ### Update the position of all sardine using Eq.(9)
            for i in range(0, len(s_pop)):
                # s_pop[i][ID_POS] = np.random.uniform() * (sf_gbest[ID_POS] - s_pop[i][ID_POS] + AP)
                d = np.random.uniform() * (np.array(sf_gbest[ID_POS]) - np.array(s_pop[i][ID_POS]) + AP)
                s_pop[i][ID_POS] = discrete(d)

                
            ###对沙丁鱼进行交叉操作###
            for num in range(len(s_pop)):
                s_pop[num]=deepcopy(crossover(sf_gbest,s_pop[num]))
        
        
        ## Recalculate the fitness of all sardine
        for i in range(0, len(s_pop)):
            s_pop[i][ID_FIT] = fitness_model(s_pop[i][ID_POS], k)

        ## Sort the population of sailfish and sardine (for reducing computational cost)
        sf_pop = sorted(sf_pop, key=lambda temp: temp[ID_FIT], reverse=True)
        s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT], reverse=True)
        for i in range(0, pop_size):
            s_size_2 = len(s_pop)
            if s_size_2 == 0:
                s_pop = [create_solution(domain_range, problem_size) for _ in range(0, s_size)]
                s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT], reverse=True)
            for j in range(0, s_size):
                ### If there is a better solution in sardine population.
                if sf_pop[i][ID_FIT] < s_pop[j][ID_FIT]:
                    sf_pop[i] = deepcopy(s_pop[j])
                    del s_pop[j]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size

        ##若s_pop为空，则重新生成一批沙丁鱼
        if len(s_pop) is 0:
            # s_pop = [create_solution(domain_range, problem_size) for _ in range(0, s_size)]
            s_pop = [get_initial_solution(k, df) for _ in range(0, s_size)]

        sf_current_best = deepcopy(get_global_best(sf_pop, ID_FIT))
        s_current_best = deepcopy(get_global_best(s_pop, ID_FIT))

        if sf_current_best[ID_FIT] > sf_gbest[ID_FIT]:
            sf_gbest = deepcopy(sf_current_best)
        #print(s_pop)
        #print(s_current_best)
        if s_current_best[ID_FIT] > s_gbest[ID_FIT]:
            s_gbest = deepcopy(s_current_best)

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        #print(sf_current_best[ID_POS])
        print(sf_current_best[ID_FIT])
        print("Epoch = {}, Fit = {}".format(epoch + 1, sf_gbest[ID_FIT]))

    return sf_gbest[ID_POS], sf_gbest[ID_FIT], clupoint


# -------------------------------------------------------------------------------------------------------
k = 30  # k-means++挑选出的最佳簇数量
ID_POS = 0
ID_FIT = 1
vector_inpath = "./data/new_his_que_btm_30.xlsx"
df = pd.read_excel(vector_inpath, encoding='utf_8_sig', header=None)
print(len(df))
sf_gbest_POS, sf_gbest_FIT, clupoint = train(k)

# 储存簇质心
# print(clupoint  # clupoint为储存所有聚类中心的变量
zhixin_outpath = "./data/Sailfish_cluster_zhixin_new2.xlsx"
clupoint_save = pd.DataFrame(clupoint)
# clunew_save.insert(0, 'id', range(len(clunew)))
clupoint_save.to_excel(zhixin_outpath, header=True, index=False, encoding='utf_8_sig')
print('------------------------------簇质心储存完成----------------------------------')
print('储存路径：%s' % zhixin_outpath)

# 储存聚类结果
cluster_result_outpath = "./data/Sailfish_cluster_result_new2.xlsx"
label_id = pd.DataFrame({'id': np.arange(len(df))})
label_id['cluster_label'] = sf_gbest_POS
label_id.to_excel(cluster_result_outpath, header=True, index=False, encoding='utf_8_sig')
print('-----------------------------聚类标签储存完成---------------------------------')
print('储存路径：%s' % cluster_result_outpath)

use_time = time.time() - start_time
print('运行时间：%s s' % use_time)