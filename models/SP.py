import sys
sys.path.append(".")
import numpy as np
from loader.txt_processor import load_config, load_data
from pprint import pprint
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error, log_loss


class ConvertionPredictor():
    def __init__(self, data_cfg):
        self.num_campaign = data_cfg['global_campaign_size']
        self.cj_pos = {}
        self.cj_neg = {}
        self.cicj_pos = {}
        self.cicj_neg = {}
        self.pj = {}
        self.pij = {}
        self.contribution = {}
        for i in range(self.num_campaign):
            self.cj_pos[i] = 0
            self.cj_neg[i] = 0
            self.pj[i] = 0
            self.contribution[i] = 0

        for i in range(self.num_campaign):
            for j in range(i+1, self.num_campaign):
                self.cicj_neg[str([i,j])] = 0
                self.cicj_pos[str([i,j])] = 0
                self.pij[str([i,j])] = 0

    def fit(self, C, Y):
        for jn_i, y_i in zip(C, Y):
            if y_i == 1:
                for cj in jn_i:
                    self.cj_pos[cj] += 1
            else:
                for cj in jn_i:
                    self.cj_neg[cj] += 1
        for i in range(self.num_campaign):
            self.pj[i] = self.cj_pos[i] / (self.cj_neg[i]+self.cj_pos[i])
        
        C_sort = []
        for jn_i in C:
            C_sort.append(sorted(jn_i))

        for jn_i, y_i in zip(C_sort, Y):
            if y_i == 1:
                for i in range(len(jn_i)):
                    for j in range(i+1, len(jn_i)):
                        if jn_i[i] == jn_i[j]:
                            continue
                        self.cicj_pos[str([int(jn_i[i]),int(jn_i[j])])] += 1
            else:
                for i in range(len(jn_i)):
                    for j in range(i+1, len(jn_i)):
                        if jn_i[i] == jn_i[j]:
                            continue
                        self.cicj_neg[str([int(jn_i[i]),int(jn_i[j])])] += 1
        
        for i in range(self.num_campaign):
            for j in range(i+1, self.num_campaign):
                self.pij[str([i,j])] = self.cicj_pos[str([i,j])] / (self.cicj_pos[str([i,j])]+self.cicj_neg[str([i,j])])
        
        for i in range(self.num_campaign):
            la = 1 / (2 * (self.num_campaign - 1))
            prop = 0
            for j in range(i+1, self.num_campaign):
                prop += self.pij[str([i,j])] - self.pj[i] - self.pj[j]
            self.contribution[i] = self.pj[i] + la * prop
            # gamma = 2 / self.num_campaign
            # self.contribution[i] = (self.pj[i] + la * prop) * gamma
        
    def predict(self, C):
        pred = []
        for jn_i in C:
            prop = 1
            for ci in jn_i:
                prop *= (1-self.pj[ci])
            pc = 1 - prop
            pred.append(pc)
        return pred
    

class Trainer():
    def __init__(self, cfgs, data_cfg):
        self.model = ConvertionPredictor(data_cfg)
        _, C_train, _, Y_train, _, _, _ = load_data(data_cfg, isTrainSet=True)
        _, C_test, _, Y_test, _, _, _ = load_data(data_cfg, isTrainSet=False)
        self.train_dataset = [C_train, Y_train]
        self.test_dataset = [C_test, Y_test]

    def train_eval(self):
        print('Start training...')
        self.model.fit(self.train_dataset[0], self.train_dataset[1])
        print('pos num of each channel')
        pprint(self.model.cj_pos)
        print('neg num of each channel')
        pprint(self.model.cj_neg)
        print('prop of each channel')
        pprint(self.model.pj)
        print('contribution of each channel')
        pprint(self.model.contribution)
        self.test()

    def test(self):
        print('Start testing...')
        pred = self.model.predict(self.test_dataset[0])
        preds = np.where(np.array(pred) > 0.5, 0, 1)
        labels = self.test_dataset[1]
        C2 = confusion_matrix(labels, preds)
        tn, fp, fn, tp = C2.ravel()
        acc = round((tp + tn) / (tp + fp + fn + tn), 3)
        rec = round((tp) / (tp + fn), 3)
        precision = round((tp) / (tp + fp), 3)
        auc = round(roc_auc_score(labels, preds), 4)
        rmse = round(np.sqrt(mean_squared_error(labels, preds)),4)
        logloss = round(log_loss(labels, preds),4)
        print(f'Finish | Accuracy: {acc} | Recall: {rec} | Precision: {precision}| RMSE: {rmse}| AUC: {auc}, logloss:{logloss}')

    def predictor(self, c, u, cat):
        return self.model.predict(c)
    
if __name__ =='__main__':
    cfgs = load_config('./configs/SP.txt')
    data_cfg = load_config('./data/criteo_cfg.txt')
    trainer = Trainer(cfgs, data_cfg)
    trainer.train_eval()




    

        

    

