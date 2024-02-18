import sys
sys.path.append(".")
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from loader.txt_processor import load_config, load_data
import copy

data_cfg = load_config('./data/criteo_cfg.txt')
device = torch.device(data_cfg['device'])

class Datasets(Dataset):
    def __init__(self, isTrainSet) -> None:
        U, C, _, Y, _, _, cat1_9 = load_data(data_cfg, isTrainSet)
        if data_cfg['with_weight']:
            W = np.load(data_cfg['stored_weights_path']).tolist()
            self.x = list(zip(U, C, cat1_9, Y, W))
        else:
            self.x = list(zip(U, C, cat1_9, Y))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]
    
    def __len__(self):
        return len(self.x)
    
def criteo_collate(batch_data):
    u = []
    c = []
    cat = []
    y = []
    # user 
    u = torch.LongTensor([i[0] for i in batch_data]).to(device)
    # channel
    C_lens = []
    for jn in batch_data:
        C_lens.append(len(jn[1]))
        c.append(torch.LongTensor(jn[1]))
    c = pad_sequence(c, padding_value=data_cfg['global_campaign_size'], batch_first=True).to(device)
    # feature vector
    cat_padding_list = data_cfg['global_cat_num_list']  # cat1_9 代表 触点的特征数据f
    cat = copy.deepcopy([i[2] for i in batch_data])
    for i in range(len(batch_data)):
        while max(C_lens) > len(cat[i]):
            cat[i].append(cat_padding_list) # 有些广告的特征向量数与广告数不匹配，则padding
    cat = torch.LongTensor(cat).to(device)
    # converison label
    y = torch.Tensor([i[3] for i in batch_data]).to(device)

    if data_cfg['with_weight']:
        w = torch.Tensor([i[4] for i in batch_data]).to(device)
        return [u,c,cat,C_lens,y,w]
    
    return [u,c,cat,C_lens,y]







class Custom_Dataset(Dataset):
    # dataset for attribution
    def __init__(self, c, u, cat) -> None:
        self.x = list(zip(u, c, cat))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]
    
    def __len__(self):
        return len(self.x)

def pred_collate(batch_data):
    # dataloader fn for attribution
    u = []
    c = []
    cat = []
    # user 
    u = torch.LongTensor([i[0] for i in batch_data]).to(device)
    # channel
    C_lens = []
    for jn in batch_data:
        C_lens.append(len(jn[1]))
        c.append(torch.LongTensor(jn[1]))
    c = pad_sequence(c, padding_value=data_cfg['global_campaign_size'], batch_first=True).to(device)
    # feature vector
    cat_padding_list = data_cfg['global_cat_num_list']  # cat1_9 代表 触点的特征数据f
    cat = copy.deepcopy([i[2] for i in batch_data])
    for i in range(len(batch_data)):
        while max(C_lens) > len(cat[i]):
            cat[i].append(cat_padding_list) # 有些广告的特征向量数与广告数不匹配，则padding
    cat = torch.LongTensor(cat).to(device)
    return [u,c,cat,C_lens]
