import numpy as np
from math import gamma
from copy import deepcopy
import pandas as pd
import random
import time

def create_item(question_num,ans_times):#初始化每一个专家的回答的情况，ans_times代表可回答次数回答，question
    temp1 =  np.zeros(question_num,int)
    for i in random.sample(range(0,question_num),random.randint(1,ans_times)):
        temp1[i]=1
    return temp1

def compare(a,b):
    #如果b优于a，则返回1；否则，返回0.
    if(a[ID_FIT][0] <= b[ID_FIT][0] and a[ID_FIT][1] >= b[ID_FIT][1] and a[ID_FIT][2] <= b[ID_FIT][2]):
        return 1
    else:
        return 0

#遗传算法：将个体与当前领导进行交叉操作
def crossover(bestitem,item):
    pos=random.randint(0,len(item[ID_POS]))
    for i in range(pos):
        item[ID_POS][i]=deepcopy(bestitem[ID_POS][i])
    return item

#遗传算法：变异，当前个体的位置随机变异
def mutation(item):
    for i in range(len(item)):
        item[ID_POS][i]=change_exp(item[ID_POS][i])
    return item

def create_solution(question_num,ans_times,expert_num):
    #计算每个个体的适应度函数值
    solution = [create_item(question_num, ans_times) for _ in range(0, expert_num)]
    fitness = [coverage(solution), expertsrc(solution), success_asd(solution)]
    #返回个体的适应度位置信息和适应度函数
    return [solution, fitness]

def get_best(archive):#获取领导
    temp = [archive[i][ID_FIT] for i in range(len(archive))] #取出每个个体的适应度函数值
    df=pd.DataFrame(temp,columns=['coverage','expert_src','success_ans'])
    df['fenzu1']= pd.cut(df['coverage'].tolist(), cube, labels=[str(i) for i in range(1,cube+1)])  #对f1分组，并打上标签
    df['fenzu2'] = pd.cut(df['expert_src'].tolist(), cube, labels=[str(i) for i in range(1, cube+1)])  #对f2分组，并打上标签
    df['fenzu3'] = pd.cut(df['success_ans'].tolist(), cube, labels=[str(i) for i in range(1, cube+1)])  #对f3分组，并打上标签
    df['fenzu'] = pd.DataFrame(df['fenzu1'].tolist()) + pd.DataFrame(df['fenzu2'].tolist()) + pd.DataFrame(df['fenzu3'].tolist())#连接个体在三个类中的标签
    tmp = {i: df['fenzu'].tolist().count(i) for i in set(df['fenzu'].tolist())}
    the_most_cube = min(zip(tmp.values(), tmp.keys()))[1]
    df = df[df['fenzu'] == the_most_cube]
    a=random.sample(df.index.tolist(),1)[0]
    return archive[a]  #随机返回最不拥挤组的索引

def change_exp(expert_old):  # 随机挑选专家的位置，更改（0，1）
    item_0=[]
    item_1=[]
    for i in range(0,len(expert_old)):
        if expert_old[i] == 0:
            item_0.append(i)
        else:
            item_1.append(i)
    expert_old[random.choice(item_1)] = 0
    expert_old[random.choice(item_0)] = 1
    return expert_old

def binary_pos1 (expert_old):
    #print('expert_old -----------', expert_old)
    sigmoid_pos = 1.0/(1.0 + np.exp(-(np.array(expert_old)-np.array([1.0]*len(expert_old)))))
    #print('sigmoid -----------',sigmoid_pos)
    pos_expert = np.where(sigmoid_pos>np.random.rand(len(expert_old)))[0]
    expert_new = np.zeros(len(expert_old))
    if len(pos_expert)>ans_times:
        #print('0000专家回答问题超过限制',len(pos_expert))
        pos_expert = random.sample(pos_expert.tolist(),ans_times)
        expert_new[pos_expert]=1
    else:
        expert_new[pos_expert]=1
        #print('1111专家回答问题未超过限制', len(pos_expert))
    return expert_new

def binary_pos (expert_old):
    #print('expert_old -----------', expert_old)
    sigmoid_pos = 1.0/(1.0 + np.exp(-(np.array(expert_old)-np.array([1.0]*len(expert_old)))))
    #print('sigmoid -----------',sigmoid_pos)
    pos_expert = np.where(sigmoid_pos>np.random.rand(len(expert_old)))[0]
    expert_new = np.zeros(len(expert_old))
    if len(pos_expert)>ans_times:
        #print('0000专家回答问题超过限制',len(pos_expert))
        pos_expert = random.sample(pos_expert.tolist(),ans_times)
        expert_new[pos_expert]=1
    else:
        expert_new[pos_expert]=1
        #print('1111专家回答问题未超过限制', len(pos_expert))
    return expert_new

def coverage(item_pos):#适应度函数1：覆盖度
    temp = [sum([item_pos[i][j] for i in range(len(item_pos))]) for j in range(len(item_pos[0]))]
    cover = 0
    jj = 0

    for kk in temp:
        if kk!=0:
            cover= (newque_sim.iloc[jj,1]+ cover)
            jj=jj+1
    return cover/1000

def redundancy(item_pos):#适应度函数4：冗余度
    #找出专家-问题矩阵列和不为零的列，即被推荐的问题
    temp = [sum([item_pos[i][j] for i in range(len(item_pos))]) for j in range(len(item_pos[0]))]
    jj = 0
    count = 0
    temp1 = []
    for kk in temp:
        if kk != 0:
            temp1.append(jj)
            count = count +1
        jj = jj + 1
    sumij = 0
    for a in temp1:
        sumj = 0
        for b in temp1:
            sumj = sumj + ques_sim.iloc[a,b]
        sumij = sumij + (1.0-1.0/sumj)
    return sumij/count

def quesnum(item_pos):#被回答的问题个数
    temp = [sum([item_pos[i][j] for i in range(len(item_pos))]) for j in range(len(item_pos[0]))]
    count = 0
    for kk in temp:
        if kk != 0:
            count = count + 1
    return count


def expertsrc(item_pos):#适应度函数2：专家资源
    pos = item_pos
    return sum(map(sum,pos))

def success_asd(item_pos):#适应度函数3：成功回答几率
    success_sum=0 # 成功回答几率
    for j in range(len(item_pos[0])): #对于第j个问题，提取它被专家回答的情况
        foronefield=0
        rank_list = [] # 存储回答问题j的专家领域排名
        exp_list = [item_pos[i][j] for i in range(len(item_pos))]# 问题j的专家回答状态表
        for kk in range(len(item_pos)): # 获得回答问题j的全部专家的领域排名
            if exp_list[kk]!=0:
                rank_list.append(expert_rank.iloc[kk,1:].tolist())
        if rank_list!=[]:
            max_rank= [max([rank_list[h][l] for h in range(0,len(rank_list))]) for l in range(0,len(rank_list[0]))]
            que_field= newque_domain.iloc[j,1:].tolist()
            foronefield = np.sum(np.multiply(np.array(max_rank),np.array(que_field)))
        success_sum = success_sum + foronefield
    return success_sum

def archive_del(archive):# 外部集合的删除策略

    temp = [archive[i][ID_FIT] for i in range(len(archive))]  # 取出每个个体的适应度函数值
    df = pd.DataFrame(temp, columns=['coverage', 'expert_src', 'success_ans'])
    df['fenzu1'] = pd.cut(df['coverage'].tolist(), cube, labels=[str(i) for i in range(1, cube+1)])  # 对f1分组，并打上标签
    df['fenzu2'] = pd.cut(df['expert_src'].tolist(), cube, labels=[str(i) for i in range(1, cube+1)])  # 对f2分组，并打上标签
    df['fenzu3'] = pd.cut(df['success_ans'].tolist(), cube, labels=[str(i) for i in range(1, cube+1)])  # 对f3分组，并打上标签
    df['fenzu'] = pd.DataFrame(df['fenzu1'].tolist()) + pd.DataFrame(df['fenzu2'].tolist()) + pd.DataFrame(df['fenzu3'].tolist()) # 连接个体在三个类中的标签
    tmp = {i: df['fenzu'].tolist().count(i) for i in set(df['fenzu'].tolist())}
    the_most_cube = max(zip(tmp.values(), tmp.keys()))[1]
    df = df[df['fenzu'] == the_most_cube]
    del archive[random.sample(df.index.tolist(), 1)[0]]  #随机删除最拥挤的组的当中的一个个体
    return archive

def archive_renew(item,item_archive,item_archive_num):  #外部集合的更新策略
    for i in range(0,len(item_archive)):

        if compare(item,item_archive[i]):
            #print('=====新个体被支配=====')
            break  # 新个体被外部集中某个个体支配，不进入外部集
        if compare(item_archive[i],item):
            item_archive.append(item)
            del item_archive[i]
            i=i-1
            continue # 新个体支配外部集中某个个体，删除被支配个体

        if (i==len(item_archive)-1):
            if (len(item_archive)<item_archive_num):
                item_archive.append(item)
                #print('=====不被支配，进入外部集=====')
            else:
                #print('=====删除外部集合个体=====')
                item_archive = deepcopy(archive_del(item_archive))
                item_archive.append(item)    # 与外部集合中每个个体互不支配，进入外部集合
    return item_archive

#-----------------------------------------------------------------------------获取数据-----------

excelFile1 = 'data/expert_domain_rank.xlsx' #获取专家排名数据
expert_rank = pd.DataFrame(pd.read_excel(excelFile1))
excelFile2 = 'data/newque_domain_sim.xlsx'#获取问题-领域数据
newque_domain = pd.DataFrame(pd.read_excel(excelFile2))
excelFile3 = 'data/newque_sim_max.xlsx'#获取问题-原始问题集合相似度
newque_sim = pd.DataFrame(pd.read_excel(excelFile3))
excelFile4 = 'data/newque_sim_array.xlsx'
ques_sim = pd.DataFrame(pd.read_excel(excelFile4))

#-----------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------算法参数--
ID_POS=0 #位置信息  {position}
ID_FIT=1 # {fitness1, fitness2, fitness3}
epoch= 50 #迭代次数
A=4 #旗鱼的战斗力参数
epxilon=0.001
pp=0.2
expert_num = expert_rank.shape[0]   #问题数量
question_num = newque_domain.shape[0]  #答案数量
archive_num= 200
ans_times = 5
sf_size= 100 #旗鱼种群数量
s_size=int(sf_size/pp) #沙丁鱼种群数量
cube = 3 #划分外部集合
#-----------------------------------------------------------------------------------------------
print('start')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# 初始化旗鱼种群：初始化位置，并计算每只旗鱼的适应度函数值
sf_pop=[create_solution(question_num,ans_times,expert_num) for _ in range(0, sf_size)]
print('----旗鱼初始化完成-----')
# 初始化沙丁鱼群，并计算每只沙丁鱼的适应度函数值
s_pop=[create_solution(question_num,ans_times,expert_num) for _ in range(0, s_size)]
print('----旗鱼初始化完成-----')

archive_all=[]
archive_all.append(sf_pop[0])
archive_all.append(s_pop[0])

for i in range(0,len(sf_pop)): # 初始化旗鱼外部集合
    #print('11111',i,'旗鱼')
    archive_all= deepcopy(archive_renew(sf_pop[i],archive_all,archive_num))
#print('000000/旗鱼外部集合数量：',len(archive_sf))

for i in range(0,len(s_pop)): # 初始化沙丁鱼外部集合
    #print('22222', i, '沙丁鱼')
    archive_all= deepcopy(archive_renew(s_pop[i],archive_all,archive_num))
#print('000000/沙丁鱼外部集合数量：',len(archive_s))


a = get_best(archive_all)  # 旗鱼中最优的
b = get_best(archive_all)  # 沙丁鱼中最优的

sf_gbest =deepcopy(a)
s_gbest = deepcopy(b)

#print("---0--- sf_gbest = {}".format(sf_gbest[ID_FIT]))
col_1 = []
col_2 = []
col_3 = []
col_4 = []
col_5 = []
col_6 = []
col_7 = []
col_8 = []
for epoch in range(0, epoch):  # 每次迭代
    ## Calculate lamda_i using Eq.(7)
    ## Update the position of sailfish using Eq.(6)
    print('NO.',epoch,'次迭代：')
    print(time.strftime( '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('-----更新旗鱼的位置-----')

    for i in range(0, sf_size):  # 循环每只旗鱼
        PD = 1 - len(sf_pop) / (len(sf_pop) + len(s_pop))  # 计算猎物的密度，论文公式（8）
        lamda_i = 2 * np.random.uniform() * PD - PD  # 公式（7），得到每一只旗鱼的λ值

        ##位置更新需要二元化，逐一更新
        for j in range(0, len(s_gbest[ID_POS])):
            sf_pop[i][ID_POS][j] = s_gbest[ID_POS][j] - lamda_i * (np.random.uniform() *
                        (sf_gbest[ID_POS][j] + s_gbest[ID_POS][j]) / 2 - sf_pop[i][ID_POS][j])  # 公式（6），更新旗鱼的位置
            sf_tmp_row = deepcopy(sf_pop[i][ID_POS][j])
            #print('-------sf_tmp-------------',sf_tmp_row)
            #print('第', i, '只旗鱼：-----第', j, '位专家：--------', min(sf_pop[i][ID_POS][j]),max(sf_pop[i][ID_POS][j]))
            sf_pop[i][ID_POS][j] = deepcopy(binary_pos1(sf_tmp_row))
            #print('第',i,'只旗鱼：-----第',j,'位专家：--------',sum(sf_pop[i][ID_POS][j]))


    a = deepcopy(get_best(archive_all))
    # 对旗鱼进行交叉、变异操作
    for num in range(len(sf_pop)):
        if np.random.uniform()>0.5:
            sf_pop[num] = deepcopy(crossover(a, sf_pop[num]))
            sf_pop[num]= deepcopy( mutation(sf_pop[num]))
        else:
            sf_pop[num] = deepcopy(crossover(b, sf_pop[num]))
            sf_pop[num] = deepcopy(mutation(sf_pop[num]))

    #print("---2--- sf_gbest = {}".format(sf_gbest[ID_FIT]))
    print('-----重新计算旗鱼的适应度函数-----')
    for i in range(0, len(sf_pop)):  #重新计算每条旗鱼的适应度函数
        sf_pop[i][ID_FIT] = [coverage(sf_pop[i][ID_POS]), expertsrc(sf_pop[i][ID_POS]), success_asd(sf_pop[i][ID_POS])]
        #print("------ sf_gbest = {}".format(sf_gbest[ID_FIT]))
    #print("---3--- sf_gbest = {}".format(sf_gbest[ID_FIT]))


    ## Calculate AttackPower using Eq.(10)  计算旗鱼的攻击力
    print('-----更新沙丁鱼的位置-----')
    AP = A * (1 - 2 * (epoch + 1) * epxilon)
    if AP < 0.5:  # 更新部分沙丁鱼的位置
        #print('AP<0.5 NO.',epoch)
        alpha = int(len(s_pop) * AP)
        beta1 = int(expert_num * AP)   #沙丁鱼的行数
        beta2 = int(question_num * AP)  #沙丁鱼的列数
        ### Random choice number of sardines which will be updated their position
        list1 = np.random.choice(range(0, len(s_pop)), alpha)
        for i in range(0, len(s_pop)):
            if i in list1:
                #### Random choice number of dimensions in sardines updated
                list2 = np.random.choice(range(0, expert_num), beta1)
                for j in range(0, expert_num):
                    if j in list2:
                        ##### Update the position of selected sardines and selected their dimensions
                        s_pop[i][ID_POS][j] = binary_pos(s_pop[i][ID_POS][j])

    else:  # 更新全部沙丁鱼的位置
        ### Update the position of all sardine using Eq.(9)
        #print('AP>=0.5 NO.', epoch)
        for i in range(0, len(s_pop)):
            for j in range(0,expert_num):
                s_pop[i][ID_POS][j] = binary_pos(s_pop[i][ID_POS][j])

    a = deepcopy(get_best(archive_all))
    #对沙丁鱼进行交叉、变异操作
    for num in range(len(s_pop)):
        s_pop[num]=deepcopy(crossover(a,s_pop[num]))
        s_pop[num] = deepcopy(mutation(s_pop[num]))


    ## 重新计算所有沙丁鱼的适应度函数
    print('-----重新计算沙丁鱼的适应度函数-----')
    for i in range(0, len(s_pop)):
        s_pop[i][ID_FIT] = [coverage(s_pop[i][ID_POS]), expertsrc(s_pop[i][ID_POS]), success_asd(s_pop[i][ID_POS])]



    ##捕杀:当沙丁鱼群中有个体的适应度函数优于旗鱼时，将被旗鱼捕杀（取代）
    print('-----捕杀-----')
    for i in range(0, sf_size):
        s_size_2 = len(s_pop)
        if s_size_2 == 0:
            s_pop=[create_solution(question_num, ans_times, expert_num) for _ in range(0, s_size)]
        for j in range(0, s_size):
            ### If there is a better solution in sardine population.'
            if compare(sf_pop[i],s_pop[j]):
                sf_pop[i] = deepcopy(s_pop[j])
                del s_pop[j]
            break

    print('-----更新旗鱼外部集合-----')
    for i in range(0, len(sf_pop)):  #更新旗鱼外部集合
        archive_all = deepcopy(archive_renew(sf_pop[i], archive_all, archive_num))
    print('1/旗鱼外部集合数量：', len(archive_all))

    print('-----更新沙丁鱼外部集合-----')
    for i in range(0, len(s_pop)):  # 初始化沙丁鱼外部集合
        archive_all = deepcopy(archive_renew(s_pop[i], archive_all, archive_num))
    print('1/沙丁鱼外部集合数量：', len(archive_all))

    sf_current_best = deepcopy(get_best(archive_all))  # 当前最优的旗鱼位置
    s_current_best = deepcopy(get_best(archive_all))   # 当前最优的沙丁鱼位置

    print("sf_current_best = {}".format(sf_current_best[ID_FIT]))
    if compare(sf_gbest,sf_current_best):
        sf_gbest = deepcopy(sf_current_best)
    if compare(s_gbest,s_current_best):
        s_gbest = deepcopy(s_current_best)
    #print('--------------------------',len(archive_sf))
    print("Epoch = {}, Fit = {}".format(epoch + 1, sf_gbest[ID_FIT]))
    print('已回答问题数量', quesnum(sf_gbest[ID_POS]))

    ##存储
    col_1.append(epoch + 1)
    col_2.append(sf_gbest[ID_FIT][0])
    col_3.append(sf_gbest[ID_FIT][1])
    col_4.append(sf_gbest[ID_FIT][2])
    col_5.append(sf_gbest[ID_POS])
    col_6.append(sf_current_best[ID_FIT][0])
    col_7.append(sf_current_best[ID_FIT][1])
    col_8.append(sf_current_best[ID_FIT][2])
df = pd.DataFrame({'Iteration': col_1, 'Best_覆盖度': col_2, 'Best_专家资源': col_3, 'Best_成功回答概率': col_4, 'sf_gbest': col_5, 'Current_覆盖度': col_6, 'Current_专家资源': col_7, 'Current_成功回答概率': col_8})
df.to_excel('data/GA-MOSFO-result_5_2.xlsx', index=None, encoding='utf_8_sig')
print('储存成功！-- data/GA-MOSFO-result_5_2.xlsx')