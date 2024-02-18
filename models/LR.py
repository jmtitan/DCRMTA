import sys
sys.path.append(".")
import torch
import torch.nn as nn  
from models.utils import Data_Emb, Base_Trainer

class ConvertionPredictor(nn.Module):

    def __init__(self, data_cfg):
        super(ConvertionPredictor, self).__init__()
        self.emb = Data_Emb(data_cfg)
        in_dim = data_cfg['C_embedding_dim'] + sum(data_cfg['cat_embedding_dim_list'])+data_cfg['U_embedding_dim']
        self.linear = nn.Linear(in_dim, 1)
        self.activate = nn.Sigmoid()
    
    def forward(self, U, C, cat):
        tp, u = self.emb(U, C, cat)
        x = tp.sum(1)
        h = torch.cat((x, u), 1)
        h = self.linear(h)
        pred = self.activate(h)
        return pred
    
class Trainer(Base_Trainer):
    def __init__(self, cfgs, data_cfg):
        super().__init__(cfgs, data_cfg)
        if cfgs['pretrained']:
            self.model = ConvertionPredictor(data_cfg).to(cfgs['device'])
            self.model.load_state_dict(torch.load(cfgs['pretrained_model_path']))
        else:
            self.model = ConvertionPredictor(data_cfg).to(cfgs['device'])

    def train_eval(self):
        bceloss = nn.BCELoss(reduction="none")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfgs['predictor_learning_rate'])
        optimizer.zero_grad()
        maxauc = 1e-5
        minloss = 1e5
        la = self.cfgs['var_loss_lambda']
        nita = self.cfgs['bce_loss_nita']
        print('Start Train...')
        for i in range(self.epoches):
            print(f'Epoch: {i} ', end='| ')
            logloss = []
            if self.data_cfg['with_weight']:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, _, y, w = b_data
                    pred = self.model(u, c, cat).flatten()
                    loss_bce = bceloss(pred, y) * w
                    n_ele = loss_bce.numel()
                    loss_bce_mean = loss_bce.mean()
                    var = torch.sum(torch.pow(torch.add(loss_bce, -loss_bce_mean), 2)) / (n_ele - 1)
                    loss_var = torch.pow(var / n_ele, 0.5)
                    loss = nita*loss_bce_mean + la*loss_var             
                    logloss.append(loss_bce_mean.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, _, y = b_data
                    pred = self.model(u, c, cat).flatten()
                    loss_bce = bceloss(pred, y)
                    loss = loss_bce.mean()                 # 后续可能添加
                    logloss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            logloss_mean = sum(logloss) / len(logloss)
            self.log_writer.add_scalar(f"loss/logloss".format(logloss_mean), i)
            auc = self.test(i)
            if logloss_mean < minloss:
                minloss = logloss_mean
            if maxauc < auc:
                maxauc = auc
                if self.cfgs['save_model']:
                    torch.save(self.model.state_dict(), self.cfgs['model_save_path'])
            print(f'logloss_mean: {logloss_mean}')
            self.model.train()
        
        print('Training Finish')
        print(f'best model performance, AUC: {maxauc} | logloss: {minloss}')

    

