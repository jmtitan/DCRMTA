import random
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torch import optim 
from torch.autograd import Function
from typing import Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
from loader.txt_processor import load_config, load_data
from loader.data_loader import Custom_Dataset, pred_collate
from models.utils import Base_Trainer, Data_Emb

torch.manual_seed(1)

# LSTM_VAE

class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        LSTM_hidden_layer_depth = 1,
        dropout_rate = 0.2
    ):
        super(LSTM_Encoder, self).__init__()
        self.model = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = LSTM_hidden_layer_depth,
            batch_first = True,
            dropout = dropout_rate
        )
        
    def forward(self, x):
        _, (h_end, c_end) = self.model(x)
        # h_end = h_end[:, -1, :]
        return h_end[-1]


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        batch_size,
        LSTM_hidden_layer_depth = 1,
        dropout_rate = 0.2,
        device = "cuda:0"
    ):
        super(LSTM_Decoder, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.LSTM_hidden_layer_depth = LSTM_hidden_layer_depth
        self.device = device

        self.model = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = LSTM_hidden_layer_depth,
            batch_first = True,
            dropout = dropout_rate
        )

    def forward(self, h_state, seq_len):
        decoder_inputs = torch.zeros(h_state.shape[0], seq_len, self.input_dim).to(self.device)
        c_0 = torch.zeros(self.LSTM_hidden_layer_depth, h_state.shape[0], self.hidden_dim).to(self.device)
        h_0 = h_state.repeat(self.LSTM_hidden_layer_depth, 1, 1)

        # print(decoder_inputs.shape)
        # print(c_0.shape)
        # print(h_0.shape)

        decoder_output, _ = self.model(decoder_inputs, (h_0, c_0))
        return decoder_output

class LSTM_OneHotVAE(nn.Module):
    def __init__(
        self,
        input_dim, 
        LSTM_hidden_dim,
        latent_variable_size,
        batch_size,
        device = "cuda:0"
    ):
        super(LSTM_OneHotVAE, self).__init__()
        self.latent_variable_size = latent_variable_size
        self.device = device
        
        self.encoder = LSTM_Encoder(input_dim, LSTM_hidden_dim)

        self.hidden_to_mean = nn.Linear(LSTM_hidden_dim, latent_variable_size)
        self.hidden_to_logvar = nn.Linear(LSTM_hidden_dim, latent_variable_size)

        self.latent_to_hidden = nn.Linear(latent_variable_size, LSTM_hidden_dim)

        self.decoder = LSTM_Decoder(input_dim, LSTM_hidden_dim, batch_size)

        self.hidden_to_output = nn.Linear(LSTM_hidden_dim, input_dim)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)   
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, x, lens, training = True):
        packed_x = pack_padded_sequence(
            input = x,
            lengths = lens,
            batch_first = True,
            enforce_sorted = False
        )
        self.training = training
        
        encoded_x = self.encoder(packed_x)
        mu = self.hidden_to_mean(encoded_x)
        logvar = self.hidden_to_logvar(encoded_x)
        z = mu

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            z = eps.mul(std).add_(mu)

        h_state = self.latent_to_hidden(z)

        decoder_output = self.decoder(h_state, max(lens))

        recon_x = self.hidden_to_output(decoder_output)

        return encoded_x, mu, logvar, z, h_state, decoder_output, recon_x
    
def train_vae(C, cfgs):
    campaign_size = cfgs['global_campaign_size']
    epochs = cfgs['vae_epoch_num']
    batch_size = cfgs['vae_batch_size']
    device = cfgs['device']

    model = LSTM_OneHotVAE(
        input_dim = campaign_size + 1,
        LSTM_hidden_dim = cfgs['vae_LSTM_hidden_dim'],
        latent_variable_size = cfgs['vae_latent_variable_size'],
        batch_size = batch_size,
        device = device
    )
    model = model.to(device)

    loss_crosse = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr =  cfgs['vae_learning_rate']
    )
  
    for e in range(epochs):
        random.seed(e)
        random.shuffle(C)

        optimizer.zero_grad()
        print("Start of epoch {}".format(e))
        all_diff = 0
        train_index = 0        

        while train_index < len(C):
            input_batch = C[train_index : train_index + batch_size
                                if train_index + batch_size < len(C)
                                else len(C)]
            batch_C, batch_lens = [], []
            for seq in input_batch:
                batch_lens.append(len(seq))
            for seq in input_batch:
                seq_list = []
                for i in range(max(batch_lens)):
                    cnt_1hot = [0] * (campaign_size + 1)
                    if i < len(seq):
                        cnt_1hot[seq[i]] = 1
                    else:
                        cnt_1hot[-1] = 1
                    seq_list.append(cnt_1hot)
                batch_C.append(seq_list)
            batch_C = torch.Tensor(batch_C).to(device)
            # print(batch_C.shape)

            encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = model(batch_C, batch_lens)

            cre_loss = 0
            for i, seq_len in enumerate(batch_lens):
                target = torch.LongTensor(input_batch[i][:seq_len]).to(device)
                cre_loss += loss_crosse(
                    recon_x[i, :seq_len, :],
                    target
                )
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 0)).mean().squeeze()
            diff = cre_loss + kld
            print("Batch {0}/{1}\t CrossEntropy_loss {2:.4f}\t KLD_loss {3:.4f}".format(
                train_index, len(C), cre_loss, kld))
            diff.backward()
            optimizer.step()
            all_diff += diff
            train_index += batch_size
        print("All loss: {}".format(all_diff))

    return model        



def judge_1hot_tag(out_list, tag):
    return (max(out_list) == out_list[tag])

def eval_vae(lstm_vae, C, cfgs):
    device = cfgs['device']
    campaign_size = cfgs['global_campaign_size']
    batch_size = cfgs['vae_batch_size']
    judge_index = 0
    right_slot, wrong_slot = 0, 0
    while judge_index < len(C):
        if right_slot + wrong_slot > 100000:
            break
        input_batch = C[judge_index : judge_index + batch_size
                        if judge_index + batch_size < len(C)
                        else len(C)]
        batch_C, batch_lens = [], []
        for seq in input_batch:
            batch_lens.append(len(seq))
        for seq in input_batch:
            seq_list = []
            for i in range(max(batch_lens)):
                cnt_1hot = [0] * (campaign_size + 1)
                if i < len(seq):
                    cnt_1hot[seq[i]] = 1
                else:
                    cnt_1hot[-1] = 1
                seq_list.append(cnt_1hot)
            batch_C.append(seq_list)
        batch_C = torch.Tensor(batch_C).to(device)        

        encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = lstm_vae(batch_C, batch_lens, False)
        

        for i in range(len(input_batch)):
            for j in range(batch_lens[i]):
                if judge_1hot_tag(recon_x[i,j], input_batch[i][j]):
                    right_slot += 1
                else: 
                    wrong_slot += 1

        judge_index += batch_size

    print("Right slot {}".format(right_slot))
    print("Wrong slot {}".format(wrong_slot))
    print("Acc is {}".format(right_slot / (right_slot + wrong_slot)))  















# Domain Classifier

class DomainClassifier(nn.Module):
    def __init__(
        self,
        n_hidden,
        dim_z, # the dim of latent vector from VAE
        num_embeddings, # can be seen as the number of users
        embedding_dim, 
        dim_hidden # the dim of hidden vector of DC
    ):
        super(DomainClassifier, self).__init__()
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim
        )
        self.input_net = nn.Linear(embedding_dim + dim_z, dim_hidden)
        self.hidden_net = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1) ]
        )
        self.output_net = nn.Linear(dim_hidden, 1)
    
    def forward(self, u, z):
        # print(u.shape)
        # u = u.squeeze()
        user_emb = self.embedding(u)    # 原始数据的嵌入向量的查找表
        zx = torch.cat((user_emb, z), 1)
        zy = F.elu(self.input_net(zx))
        for i in range(self.n_hidden - 1):
            zy = F.elu(self.hidden_net[i](zy))
        y = torch.sigmoid(self.output_net(zy))
        return y


def getDomainClassifier(lstm_vae, U, C, cfgs):
    user_num = cfgs['global_user_num']
    campaign_size = cfgs['global_campaign_size']
    epoch_num = cfgs['dc_epoch_num']
    device = cfgs['device']

    hidden_size = cfgs['dc_hidden_size']
    hidden_layer = cfgs['dc_hidden_layer']
    num_embeddings = user_num
    embedding_dim = cfgs['dc_user_embedding_dim']
    dc = DomainClassifier(
        hidden_layer,
        lstm_vae.latent_variable_size,
        num_embeddings,
        embedding_dim,
        hidden_size
    ).to(device)

    optimizer = optim.Adam(dc.parameters(), lr = cfgs['dc_learning_rate'])
    batch_size = cfgs['dc_batch_size']


    bceloss = nn.BCELoss(reduction = 'mean')

    for ep in range(epoch_num):
        random.seed(ep)
        random.shuffle(C)
        random.seed(ep)
        random.shuffle(U)

        cl_losses = []
        train_index = 0
        while train_index < len(C):
            input_batch = C[train_index : train_index + batch_size
                                if train_index + batch_size < len(C)
                                else len(C)]
            batch_C, batch_lens = [], []
            for seq in input_batch:
                batch_lens.append(len(seq))
            for seq in input_batch:
                seq_list = []
                for i in range(max(batch_lens)):
                    cnt_1hot = [0] * (campaign_size + 1)
                    if i < len(seq):
                        cnt_1hot[seq[i]] = 1
                    else:
                        cnt_1hot[-1] = 1 # pad one hot vector, manually
                    seq_list.append(cnt_1hot)
                batch_C.append(seq_list)
            batch_C = torch.Tensor(batch_C).to(device)

            encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = lstm_vae(batch_C, batch_lens, False)
            batch_z = mu + torch.exp(0.5 * logvar).to(device)  * torch.randn(size = mu.size()).to(device) # we can alse set true and use z
            batch_z_neg = torch.randn(size = batch_z.size()).to(device)

            batch_u = U[train_index : train_index + batch_size
                            if train_index + batch_size < len(U)
                            else len(U)]
            batch_u = torch.LongTensor(batch_u).to(device)
            batch_u = torch.cat((batch_u, batch_u), dim = 0).to(device)
            batch_z = torch.cat((batch_z, batch_z_neg), dim = 0).to(device)
            label_batch = torch.cat((torch.zeros(len(batch_C), 1), torch.ones(len(batch_C), 1)), dim = 0).to(device)

            pre_d = dc(batch_u, batch_z)

            loss = bceloss(pre_d, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cl_losses.append(loss.item())
            train_index += batch_size

        print("Epoch {}".format(ep))
        print("Avg loss {}".format(sum(cl_losses) / len(cl_losses)    ))
        # print(cl_losses)

    return dc 


















# Journey Reweighting

def calcSampleWeights(U, C, lstm_vae, dc, cfgs):
    campaign_size = cfgs['global_campaign_size']
    device = cfgs['device']
    batch_size = cfgs['vae_batch_size']   #dc_batch_size
    calc_weight_nums = cfgs['calcl_weight_nums']

    n = len(U)
    weight = np.zeros(n)
    train_index = 0
    while train_index < len(C):
        input_batch = C[train_index : train_index + batch_size
                            if train_index + batch_size < len(C)
                            else len(C)]        # C取一个batchsize的数据
        batch_C, batch_lens = [], []
        for seq in input_batch:
            batch_lens.append(len(seq))
        for seq in input_batch:        # 对C的数据进行1-HOT 编码
            seq_list = []
            for i in range(max(batch_lens)):
                cnt_1hot = [0] * (campaign_size + 1)
                if i < len(seq):
                    cnt_1hot[seq[i]] = 1
                else:
                    cnt_1hot[-1] = 1 # pad one hot vector, manually
                seq_list.append(cnt_1hot)
            batch_C.append(seq_list)

        batch_C = torch.Tensor(batch_C).to(device)
        encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = lstm_vae(batch_C, batch_lens, False)      # VAE输出

        batch_u = U[train_index : train_index + batch_size
                        if train_index + batch_size < len(U)
                        else len(U)]
        batch_u = torch.LongTensor(batch_u).to(device)
        nums = calc_weight_nums #50
        start = train_index
        end = min(train_index+batch_size, len(U))
        for j in range(nums):
            batch_z = mu + torch.exp(0.5 * logvar).to(device) * torch.randn(size = mu.size()).to(device) # 重参数化
            pre_d = dc(batch_u, batch_z)    # 领域分类器的输出：领域种类 * 各自概率
            pre_d = pre_d.detach().cpu().numpy().squeeze()
            weight[start:end] += ((1 - pre_d) / pre_d) / nums   # IPTW 逆概率加权 ?????为什么除以 nums
        weight[start:end] = 1 / weight[start:end]
        train_index += batch_size
    weight /= weight.mean()             # 归一化权重
    return weight

def Journey_Reweight():
    cfgs = load_config('../configs/causalMTA.txt')
    device = cfgs['device']

    # load_training_data
    U_train, C_train, T_train, Y_train, cost_train, CPO_train, cat1_9_train = load_data(cfgs)
    
    if cfgs['train_vae']:
        # train the VAE
        print("The VAE phase:")
        if (cfgs['pre_vae_path'] == 'None'):
            print("Start training the VAE...")
            C_train_input = copy.deepcopy(C_train)
            lstm_vae = train_vae(C_train_input, cfgs)
            torch.save(lstm_vae, cfgs['stored_vae_path'])
            print("End training the VAE.")
        else:
            print("Loading the VAE model...")
            lstm_vae = torch.load(cfgs['pre_vae_path']).to(device)
        # eval the VAE
        print("Start evaluating VAE.")
        C_train_input = copy.deepcopy(C_train)
        eval_vae(lstm_vae, C_train_input, cfgs)
    if cfgs['train_dc']:
        # get the domain classifier
        print("The domain classifier phase:")
        if (cfgs['pre_dc_path'] == 'None'):
            print("Start training the domain classifier...")
            U_train_input = copy.deepcopy(U_train)
            C_train_input = copy.deepcopy(C_train)
            dc = getDomainClassifier(lstm_vae, U_train_input, C_train_input, cfgs)
            torch.save(dc, cfgs['stored_dc_path'])
            print("End training the domain classifier.")
        else:
            print("Loading the dc model...")
            dc = torch.load(cfgs['pre_dc_path']).to(device)
    
    # # calculate the sample weights
    print("The weights calculation phase:")
    if (cfgs['pre_weights_path'] == 'None'):
        print("Statring calculating the weights...")
        U_train_input = copy.deepcopy(U_train)
        C_train_input = copy.deepcopy(C_train)
        weight = calcSampleWeights(U_train_input, C_train_input, lstm_vae, dc, cfgs)
        np.save(cfgs['stored_weights_path'], weight)
    else:
        print("Loading the weights...")
        weight = np.load(cfgs['pre_weights_path'])
    
    # skim at the weights
    print("Skimming at the weights.")
    print("The average of the weight is {}".format(weight.mean() ))
    print("The largest weight is {}".format(weight.max() ))
    print("The minimize of the weight is {}".format(weight.min()))
    print("The first 128 weights ars:")
    print(weight[:128])
    return weight
















# Causal Predictor

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

        self.emb = Data_Emb(data_cfg)
        
        self.bi_lstm = nn.LSTM(
            input_size=data_cfg['C_embedding_dim'] + sum(data_cfg['cat_embedding_dim_list']),   # 49
            hidden_size=cfgs['predictor_hidden_dim'],    # 64
            num_layers=cfgs['predictor_hidden_layer_depth'], # 2
            batch_first=True,
            dropout=cfgs['predictor_drop_rate'],
            bidirectional=True
        )
        self.lstm_h_dim = cfgs['predictor_hidden_dim']
        
        self.rev_mlp = nn.Linear(cfgs['predictor_hidden_dim'], data_cfg['global_campaign_size']+1)  # nn.linear 可以兼容不同的维度输入Pytorch linear 多维输入的参数

        self.final_input_net = nn.Linear(cfgs['predictor_hidden_dim'] + data_cfg['U_embedding_dim'], cfgs['predictor_fc_hidden_dim'])
        self.num_fc_hidden_layer = cfgs['predictor_fc_hidden_layer']
        self.final_hidden_net = nn.ModuleList(
            [nn.Linear(cfgs['predictor_fc_hidden_dim'], cfgs['predictor_fc_hidden_dim']) for i in range(cfgs['predictor_fc_hidden_layer'] - 1)]
        )
        self.final_output_net = nn.Linear(cfgs['predictor_fc_hidden_dim'], 1)
        self.sigmoid = nn.Sigmoid()
        if cfgs['gradient_reversal_layer']:  # 是否梯度反转
            self.grl = GRLayer()

    def attention(self, lstm_output, final_state):
        # final_state(200, 64)
        # lstm_output(200,num_of_Channels,64)
        merged_state = final_state
        merged_state = merged_state.unsqueeze(-1)  # merged_state(200, 64, 1)
        # bmm矩阵乘法
        weights = torch.bmm(lstm_output, merged_state)  # 通过内积求相似度
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)  # weight (200, num_of_Channels, 1)就是注意力分配权重
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, U, C, cat, lens):
        embeded_tp, embeded_u = self.emb(U, C, cat)
        packed_input = pack_padded_sequence(  # 压缩填充张量
            input=embeded_tp,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )

        lstm_output, (lstm_hidden, _) = self.bi_lstm(packed_input)

        lstm_output, output_lengths = pad_packed_sequence(lstm_output)  # 对压缩填充张量进行解压缩

        lstm_output = lstm_output.permute(1, 0, 2)  # 前两个数据列换位置

        lstm_output = lstm_output[:, :, :self.lstm_h_dim] + lstm_output[:, :, self.lstm_h_dim:]
        # 最后一维度数据分成前后两部分相加 (200, num_of_Channels, 128) - > (200, num_of_Channels, 64) + (200, num_of_Channels, 64)

        c_rev = None
        if getattr(self, 'grl', None) is not None:
            c_rev = self.rev_mlp(self.grl(lstm_output))  # crossentropyloss 不需要softmax

        querys = []
        for i, len in enumerate(output_lengths):  # 按滑动窗口获取lstm输出
            querys.append(lstm_output[i, len - 1, :])  # 只取lstm对于channel序列输出的最后一个state的编码
        querys = torch.stack(querys)  # querys (200, 64)

        # attention layer
        attn_output = self.attention(lstm_output, querys)  # attn_output (200, 64)

        final_input_tsr = torch.cat((attn_output, embeded_u), 1)
        final_middle = F.elu(self.final_input_net(final_input_tsr))
        for i in range(self.num_fc_hidden_layer - 1):
            final_middle = F.elu(self.final_hidden_net[i](final_middle))
        final_output = self.final_output_net(final_middle)

        return self.sigmoid(final_output), c_rev



















# model Trainer for causalMTA
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
        la = self.cfgs['var_loss_lambda']
        nita = self.cfgs['bce_loss_nita']
        gamma = self.cfgs['ce_loss_gamma']
        print('Start Training...')
        for i in range(self.epoches):
            print(f'Epoch: {i} ', end='| ')
            logloss = []
            if self.data_cfg['with_weight']:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, c_lens, y, w = b_data
                    pred, c_rev = self.model(u, c, cat, c_lens)
                    c_rev = c_rev.permute(2, 0, 1).flatten(1).permute(1,0)
                    loss_ce = celoss(c_rev, c.flatten())
                    loss_bce = bceloss(pred.flatten(), y) * w
                    loss_bce_mean = loss_bce.mean()
                    n_ele = loss_bce.numel()
                    var = torch.sum(torch.pow(torch.add(loss_bce, -loss_bce_mean), 2)) / (n_ele - 1)
                    loss_var = torch.pow(var / n_ele, 0.5)
                    loss = gamma*loss_ce + nita*loss_bce_mean + la*loss_var

                    self.log_writer.add_scalar(f"loss/bce_loss".format(loss_bce_mean.item()), idx)
                    self.log_writer.add_scalar(f"loss/ce_loss".format(loss_ce.item()), idx)
                    self.log_writer.add_scalar(f"loss/var_loss".format(loss_var.item()), idx)
                    logloss.append(loss_bce_mean.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                for idx, b_data in enumerate(self.datald_train):
                    u, c, cat, c_lens, y = b_data
                    pred, c_rev = self.model(u, c, cat, c_lens)
                    c_rev = c_rev.permute(2, 0, 1).flatten(1).permute(1,0)
                    loss_ce = celoss(c_rev, c.flatten())
                    loss_bce = bceloss(pred.flatten(), y).mean()
                    loss = gamma*loss_ce + nita*loss_bce
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
            pred, c_rev = self.model(u, c, cat, c_lens)
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
            pred, c_rev = self.model(u, c, cat, c_lens)
            if len(pred) > 1:
                pred = pred.squeeze()
            else:
                pred = pred[0]
            converts.extend(pred.cpu().detach().numpy().tolist())
        return converts


