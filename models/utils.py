import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loader.data_loader import Custom_Dataset, pred_collate, Datasets, criteo_collate
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error


class Data_Emb():
    def __init__(self, config):
        device = torch.device(config['device'])
        self.channel_embedding = nn.Embedding(
            num_embeddings=config['global_campaign_size']+1,    
            embedding_dim=config['C_embedding_dim']       
        ).to(device)

        self.cat_cnt = len(config['cat_embedding_dim_list'])
        self.cat_embedding_list = nn.ModuleList(
            [nn.Embedding(num_embeddings=config['global_cat_num_list'][i]+1,
            embedding_dim=config['cat_embedding_dim_list'][i]) for i in range(self.cat_cnt)]
        ).to(device)

        self.user_embedding = nn.Embedding(
            num_embeddings=config['global_user_num'],
            embedding_dim=config['U_embedding_dim']
        ).to(device)

    def __call__(self, U, C, cat):
        embeded_C = self.channel_embedding(C)

        embeded_cat_list = []
        for i in range(self.cat_cnt):
            embeded_cat_list.append(self.cat_embedding_list[i](cat[:, :, i]))   # 特征向量 emb

        concated_tsr = embeded_C    #channel emb
        for i in range(self.cat_cnt):
            concated_tsr = torch.cat((concated_tsr, embeded_cat_list[i]), 2)
        
        embeded_U = self.user_embedding(U)
        return concated_tsr, embeded_U

class Base_Trainer:

    def __init__(self, cfgs, data_cfg):
        self.log_writer = SummaryWriter("./log/torch_runs/{}_epoch_{}_weight_{}/".format(cfgs['model_name'], cfgs['predictor_epoch_num'], data_cfg['with_weight']))
        self.batch = cfgs['predictor_batch_size']
        self.epoches = cfgs['predictor_epoch_num']
        self.datald_train = DataLoader(Datasets(isTrainSet=True), batch_size=self.batch, shuffle=True, collate_fn=criteo_collate)
        self.datald_test = DataLoader(Datasets(isTrainSet=False), batch_size=self.batch, shuffle=False, collate_fn=criteo_collate)
        self.cfgs = cfgs
        self.data_cfg = data_cfg
        # Need modify
        # if cfgs['pretrained']:
        #     self.model = None
            # device switch if needed
            # model = torch.load(PATH, map_location='cpu')
            # model = torch.load(PATH, map_location=lambda storage, loc: storage.cuda(0))
            # torch.load(PATH, map_location={'cuda:0':'cuda:1'})  
        # else:
        #     self.model = None

    def train_eval(self):
        bceloss = nn.BCELoss(reduction="none")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfgs['predictor_learning_rate'])
        optimizer.zero_grad()
        minloss = 1e5
        maxauc = 1e-5
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
                    
                    self.log_writer.add_scalar(f"loss/bce".format(loss_bce_mean),idx)
                    self.log_writer.add_scalar(f"loss/total".format(loss_var), idx)
                    logloss.append(loss_bce_mean.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, _, y = b_data
                    pred = self.model(u, c, cat).flatten()
                    loss_bce = bceloss(pred, y).mean() 
                    loss = loss_bce                # 后续可能添加
                    logloss.append(loss_bce.item())
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

    def test(self, idx):
        self.model.eval()
        predictions = []
        labels = []
        for i, b_data in enumerate(self.datald_test):
            if self.data_cfg['with_weight']:
                u, c, cat, _, y, _ = b_data
            else:
                u, c, cat, _, y = b_data
            pred = self.model(u, c, cat)
            pred = pred.squeeze()
            pred_sq = torch.where(pred > 0.5, 1, 0).tolist()
            predictions.extend(pred_sq)
            labels.extend(y.tolist())
        C2 = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = C2.ravel()
        acc = round((tp + tn) / (tp + fp + fn + tn), 3)
        rec = round((tp) / (tp + fn), 3)
        precision = round((tp) / (tp + fp), 3)
        auc = round(roc_auc_score(labels, predictions), 4)
        rmse = round(np.sqrt(mean_squared_error(labels, predictions)),4)
        print(f'Accuracy: {acc} | Recall: {rec} | Precision: {precision}| RMSE: {rmse}| AUC: {auc}', end=' | ')
        self.log_writer.add_scalar(f"Pred/Accuracy", acc, idx)
        self.log_writer.add_scalar(f"Pred/Recall", rec, idx)
        self.log_writer.add_scalar(f"Pred/Precision", precision, idx)
        self.log_writer.add_scalar(f"Pred/AUC", auc, idx)
        self.log_writer.add_scalar(f"Pred/RMSE", rmse, idx)
        return auc
    
    def predictor(self, c, u, cat):
        converts = []
        batch = self.cfgs['predictor_batch_size']
        datald_custom = DataLoader(Custom_Dataset(c, u, cat), batch_size=batch, shuffle=False, collate_fn=pred_collate)
        for i, b_data in enumerate(datald_custom):
            u, c, cat, _ = b_data
            pred = self.model(u, c, cat)
            if len(pred) > 1:
                pred = pred.squeeze()
            else:
                pred = pred[0]
            converts.extend(pred.cpu().detach().numpy().tolist())
        return converts
    