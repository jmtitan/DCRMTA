import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
from loader.data_loader import Custom_Dataset, pred_collate
from models.utils import Base_Trainer, Data_Emb
from torch.autograd import Function
from typing import Any, Optional, Tuple

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRLayer(nn.Module):
    def __init__(self):
        super(GRLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
    
class ConvertionPredictor(nn.Module):
    def __init__(self, cfgs, data_cfg):
        super(ConvertionPredictor, self).__init__()
        self.cfgs = cfgs
        self.data_cfg = data_cfg
        self.emb = Data_Emb(data_cfg)
        self.time_decay = cfgs['time_decay']
        fvec_in = sum(data_cfg['cat_embedding_dim_list'])
        lstm_in = data_cfg['C_embedding_dim'] + fvec_in
        self.num_features = cfgs['predictor_hidden_dim'] # 64
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
            nn.Linear(64,1)
            )
        self.final_activate = nn.Sigmoid()

        self.dense_rev = nn.Linear(cfgs['predictor_hidden_dim'], data_cfg['global_campaign_size']+1)  # nn.linear 可以兼容不同的维度输入Pytorch linear 多维输入的参数
        if cfgs['gradient_reversal_layer']:  # 是否梯度反转
            self.grl = GRLayer()
        
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
    
    def BAP(self, features, attentions):
        # Counterfactual Attention Learning
        if len(attentions.shape) < 2:
            attentions = attentions.unsqueeze(0)
        attn_map = attentions.unsqueeze(2)
        B,UF,M = features.size()
        # feature_matrix:(B,M,C) -> (B,M *C)
        feature_matrix = (torch.einsum('bum, bmi->bm',(features, attn_map)) / float(UF)).view(B,-1)
        #sign-sart
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix)+ 1e-6)#12 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)
        
        if self.training:
            fake_att = torch.zeros_like(attn_map).uniform_(0, 2)  # 零化 + 随机数填充
        else:
            fake_att = torch.ones_like(attn_map)
        counterfactual_feature = (torch.einsum('bum, bmi->bm',(features, fake_att)) / float(UF)).view(B,-1)
        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + 1e-6)
        counterfactual_feature = F.normalize(counterfactual_feature,dim=-1)

        return feature_matrix, counterfactual_feature
    
    def forward(self, U, C, cat, lens):
        tp, u = self.emb(U, C, cat)
        f_vec = tp[:,:,4:]
        c_rev = None

        packed_input = pack_padded_sequence(  # 压缩填充张量
            input=tp,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )

        # channel-touchpoint feature extraction
        lstm_out, (_, _) = self.lstm(packed_input) 
        a, _ = pad_packed_sequence(lstm_out)  # 对压缩填充张量进行解压缩
        a = a.permute(1, 0, 2)  # (batch, channel_seq, hidden_dim)
        if getattr(self, 'grl', None) is not None:
            c_rev = self.dense_rev(self.grl(a))  
        c_m = self.seq_attention(a, tp).squeeze()

        # user-touchpoint  feature extraction
        f_m = self.dense_2(f_vec)   #(B, seq, 64)
        u_m = self.dense_3(u)       #(B, 64)
        uf_m = torch.cat((u_m.unsqueeze(dim=1), f_m), dim=1) #(B, 1+seq, 64)
        uf_attn = self.user_attention(f_m, u_m) #(B, 64)
        uf_m, uf_m_hat = self.BAP(uf_m, uf_attn)    #(B, 64)
    
        h_f = uf_m + c_m
        h_f_hat = uf_m_hat + c_m

        p = self.dense_final(h_f)
        p_hat = self.dense_final(h_f_hat)
        
        return self.final_activate(p), self.final_activate(p-p_hat), c_rev






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
        nita = self.cfgs['bce_loss_nita']
        gamma = self.cfgs['ce_loss_gamma']
        delta = self.cfgs['cf_loss_delta']
        print('Start Training...')
        for i in range(self.epoches):
            print(f'Epoch: {i} ', end='| ')
            logloss = []
            for idx, b_data in enumerate(self.datald_train):
                u, c, cat, c_lens, y = b_data
                pred, cfeff, c_rev = self.model(u, c, cat, c_lens)
                c_rev = c_rev.permute(2, 0, 1).flatten(1).permute(1,0)
                loss_ce = celoss(c_rev, c.flatten())
                loss_bce = bceloss(pred.flatten(), y).mean()
                loss_bce_cf = bceloss(cfeff.flatten(), y).mean()
                loss = gamma*loss_ce + nita*loss_bce + delta * loss_bce_cf
                logloss.append(loss_bce.item())

                self.log_writer.add_scalar(f"loss/bce_loss".format(loss_bce.item()), idx)
                self.log_writer.add_scalar(f"loss/ce_loss".format(loss_ce.item()), idx)
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
            pred, _, _ = self.model(u, c, cat, c_lens)
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
        with torch.no_grad():
            converts = []
            datald_custom = DataLoader(Custom_Dataset(c, u, cat), batch_size=self.batch, shuffle=False, collate_fn=pred_collate)
            for i, b_data in enumerate(datald_custom):
                u, c, cat, c_lens = b_data
                pred, _, _ = self.model(u, c, cat, c_lens)
                if len(pred) > 1:
                    pred = pred.squeeze()
                else:
                    pred = pred[0]
                converts.extend(pred.cpu().detach().numpy().tolist())
            return converts
