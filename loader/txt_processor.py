import numpy as np
import pickle


def load_config(cfg_file):
    f = open(cfg_file, 'r')
    config_lines = f.readlines()
    cfgs = { }
    for line in config_lines:
        ps = [p.strip() for p in line.split('=')]
        if (len(ps) != 2):
            continue
        try:
            if (ps[1].find(',') != -1):
                str_line = ps[1].split(',')
                cfgs[ps[0]] = list(map(int, str_line))
            elif (ps[1].find('.') == -1):
                cfgs[ps[0]] = int(ps[1])
            else:
                cfgs[ps[0]] = float(ps[1])
        except ValueError:
            cfgs[ps[0]] = ps[1]
            if cfgs[ps[0]] == 'False':
                cfgs[ps[0]] = False
            elif cfgs[ps[0]] == 'True':
                cfgs[ps[0]] = True         

    return cfgs


def load_data(cfgs, isTrainSet = True):
    if cfgs['dataset']=='Mock':
        if isTrainSet:
            file_U = cfgs['file_U_train']
            file_C = cfgs['file_C_train']
            file_T = cfgs['file_T_train']
            file_Y = cfgs['file_Y_train']
            file_cost = cfgs['file_cost_train']
            file_CPO = cfgs['file_CPO_train']
            file_cat1_9 = cfgs['file_cat1_9_train']
        else:
            file_U = cfgs['file_U_test']
            file_C = cfgs['file_C_test']
            file_T = cfgs['file_T_test']
            file_Y = cfgs['file_Y_test']
            file_cost = cfgs['file_cost_test']
            file_CPO = cfgs['file_CPO_test']
            file_cat1_9 = cfgs['file_cat1_9_test']

        f = open(file_U, "rb")
        U = pickle.load(f)
        f.close()

        f = open(file_C, "rb")
        C = pickle.load(f)
        f.close()

        f = open(file_T, "rb")
        T = pickle.load(f)
        f.close()

        f = open(file_Y, "rb")
        Y = pickle.load(f)
        f.close()

        f = open(file_cost, "rb")
        cost = pickle.load(f)
        f.close()

        f = open(file_CPO, "rb")
        CPO = pickle.load(f)
        f.close()

        f = open(file_cat1_9, "rb")
        cat1_9 = pickle.load(f)
        f.close()

        return U, C, T, Y, cost, CPO, cat1_9
    if cfgs['dataset']=='Criteo':
        file_U = cfgs['file_U']
        file_C = cfgs['file_C']
        file_T = cfgs['file_T']
        file_Y = cfgs['file_Y']
        file_cost = cfgs['file_cost']
        file_CPO = cfgs['file_CPO']
        file_cat1_9 = cfgs['file_cat1_9']

        f = open(file_U, "rb")
        U = pickle.load(f)
        f.close()

        f = open(file_C, "rb")
        C = pickle.load(f)
        f.close()

        f = open(file_T, "rb")
        T = pickle.load(f)
        f.close()

        f = open(file_Y, "rb")
        Y = pickle.load(f)
        f.close()

        f = open(file_cost, "rb")
        cost = pickle.load(f)
        f.close()

        f = open(file_CPO, "rb")
        CPO = pickle.load(f)
        f.close()

        f = open(file_cat1_9, "rb")
        cat1_9 = pickle.load(f)
        f.close()

        length = len(U)
        ratio = int(length * cfgs['train_test_ratio'])

        if isTrainSet:
            U = U[:ratio]
            C = C[:ratio]
            T = T[:ratio]
            Y = Y[:ratio]
            cost = cost[:ratio]
            CPO = CPO[:ratio]
            cat1_9 = cat1_9[:ratio]
        else:
            U = U[ratio:]
            C = C[ratio:]
            T = T[ratio:]
            Y = Y[ratio:]
            cost = cost[ratio:]
            CPO = CPO[ratio:]
            cat1_9 = cat1_9[ratio:]

        return U, C, T, Y, cost, CPO, cat1_9


def text_save(filename, data):
    #filename为写入CSV文件的路径，data为要写入数据列表.
    print(f"存入{filename}...")
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','') #去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功")

def text_read(filename):
    attr = []
    print('读取中...')
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            cont = line.split(' ')
            cont = list(map(float, cont))
            attr.append(cont)
    print(f'{filename}读取完毕')
    return attr






