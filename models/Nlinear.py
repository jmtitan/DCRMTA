import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn as nn   
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.utils import Data_Emb, Base_Trainer
from loader.data_loader import Custom_Dataset, pred_collate
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error

class ConvertionPredictor(nn.Module):
    def __init__(self, data_cfg) -> None:
        super(ConvertionPredictor, self).__init__()
        self.emb = Data_Emb(data_cfg)

        self.backbone = nn.Linear(49, 64)
        self.mlp = nn.Sequential(
            nn.Linear(69,128),
            nn.ReLU(),
            nn.Dropout(0.5),
 
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            )
        self.fc = nn.Linear(64, 1)
        self.activate = nn.Sigmoid()

        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
    
    def attention(self, x, f):
        mid = f
        mid = mid.unsqueeze(-1)  # merged_state(200, 64, 1)
        # bmm矩阵乘法
        w = torch.bmm(x, mid)  # 通过内积求相似度
        w = F.softmax(w.squeeze(2), dim=1).unsqueeze(2)  # weight (200, num_of_Channels, 1)就是注意力分配权重
        return torch.bmm(torch.transpose(x, 1, 2), w).squeeze(2)
    
    def forward(self, U, C, cat):
        tp, u = self.emb(U, C, cat)
        x = self.backbone(tp)
        fn = x[:, -1, :]
        x = self.attention(x, fn)
        h = torch.cat((x, u), 1)
        h = self.mlp(h)
        h = self.fc(h)
        output = self.activate(h)
        return output
    
class Trainer(Base_Trainer):
    def __init__(self, cfgs, data_cfg):
        super().__init__(cfgs, data_cfg)
        if cfgs['pretrained']:
            self.model = ConvertionPredictor(data_cfg).to(cfgs['device'])
            self.model.load_state_dict(torch.load(cfgs['pretrained_model_path']))
        else:
            self.model = ConvertionPredictor(data_cfg).to(cfgs['device'])
    
    

        