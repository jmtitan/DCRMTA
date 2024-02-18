import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
from loader.data_loader import Custom_Dataset, pred_collate
from models.utils import Base_Trainer, Data_Emb

class ConvertionPredictor(nn.Module):
    def __init__(self, cfgs, data_cfg):
        super(ConvertionPredictor, self).__init__()
        self.cfgs = cfgs
        self.data_cfg = data_cfg
        self.emb = Data_Emb(data_cfg)
        self.time_decay = cfgs['time_decay']
        fvec_in = sum(data_cfg['cat_embedding_dim_list'])
        lstm_in = data_cfg['C_embedding_dim'] + fvec_in
        
        self.lstm = nn.LSTM(
            input_size=lstm_in,   # 49
            hidden_size=cfgs['predictor_hidden_dim'],    # 64
            num_layers=cfgs['predictor_hidden_layer_depth'], # 2
            batch_first=True,
            dropout=cfgs['predictor_drop_rate'],
            bidirectional=False
        )

        self.dense_1 = nn.Sequential(
            nn.Linear(cfgs['predictor_hidden_dim']+lstm_in,256),
            nn.Tanh(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        
        self.dense_2 = nn.Sequential(
            nn.Linear(fvec_in,256),
            nn.ReLU(),
            nn.Dropout(0.2),
 
            nn.Linear(256,64),
            )
        self.dense_3 = nn.Linear(data_cfg['U_embedding_dim'], 64)
        self.dense_final = nn.Sequential(
            nn.Linear(64,1),
            nn.Sigmoid()
            )

        
    def seq_attention(self, a, s_prev):
        x = torch.cat((a, s_prev), dim=2)
        x = self.dense_1(x)
        # x -= self.time_decay * t
        w = F.softmax(x, dim=1)
        return torch.bmm(torch.transpose(a,1,2), w) # (batch, 64, 1)

    def user_attention(self, f_m, u):
        u = u.unsqueeze(-1)  # (batch, 5, 1)
        # bmm矩阵乘法
        w = torch.bmm(f_m, u)  # 通过内积求相似度
        w = F.softmax(w.squeeze(2), dim=1).unsqueeze(2)  # (batch, num_of_Channels, 1)
        return torch.bmm(torch.transpose(f_m, 1, 2), w).squeeze() # (batch, 5)

    def forward(self, U, C, cat, lens):
        tp, u = self.emb(U, C, cat)
        f_vec = tp[:,:,4:]

        packed_input = pack_padded_sequence(  # 压缩填充张量
            input=tp,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )

        lstm_output, (lstm_hidden, _) = self.lstm(packed_input) 

        a, output_lengths = pad_packed_sequence(lstm_output)  # 对压缩填充张量进行解压缩

        a = a.permute(1, 0, 2)  # 前两个数据列换位置

        # a = lstm_output[:, :, :self.cfgs['predictor_hidden_dim']] + lstm_output[:, :, self.cfgs['predictor_hidden_dim']:]

        c_m = self.seq_attention(a, tp).squeeze()
        f_m = self.dense_2(f_vec)
        u_m = self.dense_3(u) 
        uf_m = self.user_attention(f_m, u_m)

        h_f = uf_m + c_m
        return self.dense_final(h_f)






class Trainer(Base_Trainer):

    def __init__(self, cfgs, data_cfg):
        super().__init__(cfgs, data_cfg)
        if cfgs['pretrained']:
            self.model = ConvertionPredictor(cfgs, data_cfg).to(cfgs['device'])
            self.model.load_state_dict(torch.load(cfgs['pretrained_model_path']))
        else:
            self.model = ConvertionPredictor(cfgs, data_cfg).to(cfgs['device'])


    def train_eval(self):
        bceloss = nn.BCELoss(reduction="none")
        celoss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfgs['predictor_learning_rate'])
        optimizer.zero_grad()
        minloss = 1e5
        maxauc = 1e-5
        print('Start Training...')
        for i in range(self.epoches):
            print(f'Epoch: {i} ', end='| ')
            logloss = []
            if self.data_cfg['with_weight']:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, c_lens, y, w = b_data
                    pred = self.model(u, c, cat, c_lens)
                    loss_bce = bceloss(pred.flatten(), y) * w
                    loss_bce_mean = loss_bce.mean()
                    loss = loss_bce_mean
                    self.log_writer.add_scalar(f"loss/bce_loss".format(loss_bce_mean.item()), idx)
                    logloss.append(loss_bce_mean.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, c_lens, y = b_data
                    pred = self.model(u, c, cat, c_lens)
                    loss_bce = bceloss(pred.flatten(), y).mean()
                    loss = loss_bce
                    logloss.append(loss_bce.item())

                    self.log_writer.add_scalar(f"loss/bce_loss".format(loss_bce.item()), idx)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            logloss_mean = sum(logloss) / len(logloss)
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

        print('\nFinish')
        self.model.eval()
        return self.model
    
    def test(self, idx):
        self.model.eval()
        predictions = []
        labels = []
        for i, b_data in enumerate(self.datald_test):
            if self.data_cfg['with_weight']:
                u, c, cat, c_lens, y, _ = b_data
            else:
                u, c, cat, c_lens, y = b_data
            pred = self.model(u, c, cat, c_lens)
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
        datald_custom = DataLoader(Custom_Dataset(c, u, cat), batch_size=self.batch, shuffle=False, collate_fn=pred_collate)
        for i, b_data in enumerate(datald_custom):
            u, c, cat, c_lens = b_data
            pred = self.model(u, c, cat, c_lens)
            if len(pred) > 1:
                pred = pred.squeeze()
            converts.extend(pred.cpu().detach().numpy().tolist())
        return converts

