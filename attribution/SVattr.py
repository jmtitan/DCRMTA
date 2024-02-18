import itertools
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from loader.txt_processor import load_data, text_save, text_read
import numpy as np
from tqdm.contrib import tzip

def plot(shapley_result):
    ### visualizing the results
    # sns.set_style("white")
    plt.subplots(figsize=(15, 8))
    s = sns.barplot(x='channel', y='shapley_value', data=shapley_result)

    for idx, row in shapley_result.iterrows():
        s.text(row.name, row.shapley_value + 5, round(row.shapley_value, 1), ha='center', color='darkslategray',
               fontweight='semibold')
    plt.title("ADVERTISING CHANNEL'S SHAPLEY VALUE",
              fontdict={'fontfamily': 'san-serif', 'fontsize': 15, 'fontweight': 'semibold', 'color': '#444444'},
              loc='center', pad=10)
    plt.show()
    
def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in itertools.combinations(List, i + 1)]
    return PS

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def v_function(A, C_values):
    '''
    This function computes the worth of each coalition.
    inputs:
            - A : a coalition of channels.
            - C_values : A dictionnary containing the number of conversions that each subset of channels has yielded.
    '''
    subsets_of_A = subsets(A)
    worth_of_A = 0
    for subset in subsets_of_A:
        if subset in C_values:
            worth_of_A += C_values[subset]
    return worth_of_A

def subsets(s):
    '''
    This function returns all the possible subsets of a set of channels.
    input :
            - s: a set of channels.
    '''
    if len(s) == 1:
        return s
    else:
        sub_channels = []
        for i in range(1, len(s) + 1):
            sub_channels.extend(map(list, itertools.combinations(s, i)))
    return sub_channels

def subseq(arr, thred=5):
    finish=[]    # the list containing all the subsequences of the specified sequence
    size = len(arr)    # the number of elements in the specified sequence
    start = 0
    end = 1 << size    # end=2**size
    interv = 1
    if 10 >= size > thred:
        interv = 2**(size - thred)         
    if size > 10:
        start = end - 2**10
        interv = 2 **(10 - thred)
    for index in range(start, end, interv):
        array = []    # remember to clear the list before each loop
        for j in range(size):
            if (index >> j) % 2:
                array.append(arr[j])
        # print(array)
        if array:
            finish.append(array)
    return finish

def calculate_shapley(ui, ci, fi, trainer):
    n = len(ci)# no. of channels
    shapley_values = defaultdict(int)
    if n == 1:
        # 避免空集子集
        shapley_values[0] = trainer.predictor([ci], [ui], [fi])[0]
        return shapley_values
    for idx, ci_t in enumerate(ci):
        # u,c,f 除去 ci对应的元素后，计算转化率
        c_tmp = ci.copy()
        f_tmp = fi.copy()
        del c_tmp[idx]
        del f_tmp[idx]
        # 删除元素后，剩余journey的所有子序列（子集太多太慢）
        Sc = subseq(c_tmp)
        Sf = subseq(f_tmp)
        # Sc = subsets(c_tmp)
        # Sf = subsets(f_tmp)
        # 通过矩阵运算方式，获得添加了删除元素的子集集合
        Sc_p = copy.deepcopy(Sc)
        Sf_p = copy.deepcopy(Sf)
        for ele in Sc_p:
            ele.append(ci[idx])
        for ele in Sf_p:
            ele.append(fi[idx])
        # 计算所有子集集合的转化率
        ctf_v = trainer.predictor(Sc, [ui]*len(Sc), Sf)
        # 计算所有添加触点p的子集集合的转化率
        ctf_vp = trainer.predictor(Sc_p, [ui]*len(Sc_p), Sf_p)
        for sc, sv, svp in zip(Sc, ctf_v, ctf_vp):
            weight = (factorial(len(sc)) * factorial(n - len(sc) - 1) / factorial(n))  # Weight = |S|!(n-|S|-1)!/n!
            contrib = svp - sv  # Marginal contribution = v(S U {i})-v(S)
            shapley_values[idx] += weight * contrib
        shapley_values[idx] = shapley_values[idx] if shapley_values[idx] > 0 else 0
    return shapley_values

def SVattr(SV):
    attr = []
    sumSV = sum(list(SV.values()))
    if sumSV == 0:
        attr = [0] * len(SV.keys())
    else:
        for k in SV.keys():
            attr.append(SV[k] / sumSV)
    return attr

def getROI(attr, C, Y, cost, k=12):######
    roi = [0]*k
    roi_de = [0]*k
    v = 1
    for a, tp, y, pay in tzip(attr, C, Y, cost):
        for i, ci in enumerate(tp):
            roi_de[int(ci)] += pay[i]
            roi[int(ci)] += a[i] * v * y
    for i, de in enumerate(roi_de):
        roi[i] /= de
    return roi

def getBudgetAttr(roi, B):
    summ = sum(roi)
    roi = [i/summ *B for i in roi]
    return roi

def Backeval(cbs, test_seq):
    Blacklist = []
    totcost = 0
    convt_num = 0
    group = {}  # 已计算的event
    for _, ci, costi, yi, flag in test_seq:
        ci = int(ci)
        if flag in Blacklist:
            continue
        else:
            if cbs[ci] > costi: 
                totcost += costi 
                convt_num += yi
                cbs[ci] -= costi
                if flag not in group.keys():
                    group[flag] = [[ci, costi, yi]]
                else:
                    group[flag].append([ci, costi, yi])
            else:
                Blacklist.append(flag)
                
                    
    # print(f'Black list num: {len(Blacklist)}\n')
    return convt_num, totcost
    

def attr_criterion(cbs, test_seq, journey_num):
    convt_num, total_cost = Backeval(cbs, test_seq)
    cpa = 1000 * total_cost / convt_num 
    cvr = convt_num / journey_num
    return cpa, cvr, convt_num



def Attribution(cfgs, data_cfg, trainer):
    print('Attributing...')
    U, C, T, Y, cost, _, cat1_9 = load_data(data_cfg, isTrainSet=False)
    # count = 0
    # attr_seq = []
    # for u,c,f in tzip(U, C, cat1_9):
    #     count += 1
    #     #  计算 Shapely Value
    #     SV = calculate_shapley(u, c, f, trainer)
    #     attr = SVattr(SV)
    #     attr_seq.append(attr)
    # print(count)
    # # # # # 保存归因结果
    attr_text_path = './save/attrib/' + cfgs['model_name'] + '.txt'
    # text_save(attr_text_path, attr_seq)

    attr_seq = text_read(attr_text_path)

    # 计算指标
    test_seq = []
    flag = 0 # 标记journey序列, 用于后续黑名单添加处理
    roi = getROI(attr_seq, C, Y, cost)
    total_Budget = sum([sum(i) for i in cost])
    Budget = total_Budget * cfgs['Budget_proportion'] # 定义budge
    budget_alloc = getBudgetAttr(roi, B=Budget)
    
    for c, y, cost, t in zip(C, Y, cost, T):
        for ci, costi, ti in zip(c, cost, t):
            test_seq.append([ti, ci, costi, y/len(c), flag])
        flag += 1
    test_seq.sort(key=lambda x: x[0], reverse=False)
    cpa, cvr, convt_num = attr_criterion(budget_alloc, test_seq, len(C))
    print('Attribution Evaluating Indicator:')
    print(f'CPA: {cpa} \nCVR: {cvr} \nconvert num: {convt_num}')    