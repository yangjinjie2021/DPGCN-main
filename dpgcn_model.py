# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_transformers import BertPreTrainedModel,BertModel

n_n = ['NN','NNP','NNPS','NNS','NFP']  #名词
jj = ['JJ','JJR','JJS','CC']  #形容词
rb = ['RB','RBR','RBS','RP']  #副词
vb = ['VB','VBD','VBG','VBN','VBP','VBZ']  #动词

class DPGraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(DPGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)
        hidden = torch.matmul(val_us, self.weight)
        output = hidden.transpose(1,2) * adj_us

        output = torch.sum(output, dim=2)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, emb_size):
        super(KeyValueMemoryNetwork, self).__init__()
        self.scale = np.power(emb_size, 0.5)

    def forward(self, key_seq, value_embed, hidden, mask_matrix, pos_emb):

        key = torch.bmm(hidden.float(), key_seq.transpose(1, 2))
        key = key / self.scale
        exp_key = torch.exp(key)
        delta_exp_key = torch.mul(exp_key.float(), mask_matrix.float())
        sum_delta_exp_key = torch.stack([torch.sum(delta_exp_key, 2)] * delta_exp_key.shape[2], 2)
        key_score = torch.div(delta_exp_key, sum_delta_exp_key + 1e-10)
        pos_emb = torch.matmul(key_score, pos_emb)
        embedding_val = value_embed.permute(3, 0, 1, 2)
        dep_emb = torch.mul(key_score.float(), embedding_val.float())
        dep_emb = dep_emb.permute(1, 2, 3, 0)
        dep_emb = torch.sum(dep_emb, 2)

        return dep_emb.type_as(hidden), pos_emb.type_as(hidden)

class DPGCN(BertPreTrainedModel):
    def __init__(self, config, opt):
        super(DPGCN, self).__init__(config, opt)
        self.config = config
        self.layer_number = 2
        self.num_labels = config.num_labels
        self.num_types = config.num_types
        self.num_pos_types = config.num_pos_types
        self.distance = config.distance
        self.device = opt.device

        self.bert = BertModel(config)
        self.TGCNLayers = nn.ModuleList(([DPGraphConvolution(config.hidden_size, config.hidden_size)
                                          for _ in range(self.layer_number)]))

        self.fc_single = nn.Linear(config.hidden_size , self.num_labels)
        self.cat_single = nn.Linear(config.hidden_size + config.pos_hidden + config.dep_hidden, config.hidden_size)
        self.dropout = nn.Dropout(config.bert_dropout)

        self.weight_nn =nn.Linear(config.hidden_size, config.hidden_size)
        self.weight_jj = nn.Linear(config.hidden_size, config.hidden_size)
        self.weight_rb = nn.Linear(config.hidden_size, config.hidden_size)
        self.weight_vb = nn.Linear(config.hidden_size, config.hidden_size)
        self.weight_other = nn.Linear(config.hidden_size,config.hidden_size)
        self.dep_gat = nn.Embedding(self.num_types, 2 * config.hidden_size, padding_idx=0)

        self.ensemble_linear = nn.Linear(1,2)
        self.ensemble = nn.Parameter(torch.FloatTensor(2, 1))

        self.dep_embedding = nn.Embedding(self.num_types, config.dep_hidden, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.num_pos_types,config.pos_hidden,padding_idx=0)
        # KVMN部分
        self.memory = KeyValueMemoryNetwork(emb_size=config.hidden_size)

    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        feat_dim = self.config.hidden_size
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, val_us.transpose(1,2)), -1).float()
        atten_expand = (dep_embed * val_cat)
        attention_score = torch.sum(atten_expand, dim=-1)
        #attention_score = F.relu(attention_score)
        #attention_score = attention_score / np.power(feat_dim, 0.1)
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score, adj.float()) # mask
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)

        if 'HalfTensor' in val_out.type():
            attention_score = attention_score.half()
        return attention_score

    def get_poslayer(self, hidden, pos_label):
        batch_size, max_len, feat_dim = hidden.shape
        hidden_pos = torch.zeros(batch_size, max_len, feat_dim, device=self.device).type_as(hidden)
        for i in range(max_len):
            for j in range(batch_size):
                if pos_label[i][j] != '0':
                    if pos_label[i][j] in n_n:
                        hidden_pos[j][i]  = self.weight_nn(hidden[j][i])
                    elif pos_label[i][j] in jj:
                        hidden_pos[j][i] = self.weight_jj(hidden[j][i])
                    elif pos_label[i][j] in rb:
                        hidden_pos[j][i] = self.weight_rb(hidden[j][i])
                    elif pos_label[i][j] in vb:
                        hidden_pos[j][i] = self.weight_vb(hidden[j][i])
                    else:
                        hidden_pos[j][i] = self.weight_other(hidden[j][i])
        return hidden_pos

    def get_avarage(self, aspect_indices, x):
        aspect_indices_us = torch.unsqueeze(aspect_indices, 2)
        x_mask = x * aspect_indices_us
        aspect_len = (aspect_indices_us != 0).sum(dim=1)
        x_sum = x_mask.sum(dim=1)
        x_av = torch.div(x_sum, aspect_len)

        return x_av

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        weight_mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                if aspect_double_idx[i, 0] - j > self.distance:
                    weight[i].append(0)
                    weight_mask[i].append(0)
                else:
                    weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
                    weight_mask[i].append(1)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
                weight_mask[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                if j - aspect_double_idx[i, 1] > self.distance:
                    weight[i].append(0)
                    weight_mask[i].append(0)
                else:
                    weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
                    weight_mask[i].append(1)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
                weight_mask[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.device)
        weight_mask = torch.tensor(weight_mask).unsqueeze(2).to(self.device)
        weight_mask = weight_mask.transpose(1, 2)
        return weight * x, weight_mask

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.device)
        return mask * x

    def forward(self, input_ids, segment_ids, valid_ids, mem_valid_ids, left_token_ids, key_list, \
                                             pos_ids, dep_adj_matrix, dep_value_matrix, pos_label):
        text_len = torch.sum(key_list != 0, dim=-1)
        aspect_len = torch.sum(mem_valid_ids != 0, dim=-1)
        left_len = left_token_ids
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        sequence_output, pooled_output = self.bert(input_ids, segment_ids)
        dep_embed = self.dep_embedding(dep_value_matrix)
        pos_embed = self.pos_embedding(pos_ids)
        dep_gat = self.dep_gat(dep_value_matrix)

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, device=input_ids.device).type_as(sequence_output)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        valid_output = self.dropout(valid_output)

        hidden = valid_output
        dep_output, pos_output= self.memory(valid_output, dep_embed, hidden, dep_adj_matrix, pos_embed)
        seq_cat = torch.cat([valid_output, dep_output, pos_output], dim=-1)
        seq_out = self.cat_single(seq_cat)

        for dpgcn in self.TGCNLayers:
            seq_gat = self.get_poslayer(seq_out, pos_label)
            attention_score = self.get_attention(seq_gat, dep_gat, dep_adj_matrix)
            seq_out = F.relu(dpgcn(seq_out, attention_score))

        pos_hidden = seq_out
        pos_hidden = self.position_weight(pos_hidden, aspect_double_idx, text_len, aspect_len)[0].to(torch.float32)
        weight = self.position_weight(pos_hidden, aspect_double_idx, text_len, aspect_len)[1]
        seq_out_mask = self.mask(seq_out , aspect_double_idx).to(torch.float32)
        alpha_mat = torch.matmul(seq_out_mask, pos_hidden.transpose(1, 2))
        alpha_sum = alpha_mat.sum(1, keepdim=True)
        alpha_score = alpha_sum / np.power(feat_dim, 0.8)
        alpha_score = torch.exp(alpha_score)
        alpha_score = torch.mul(alpha_score, weight)
        sum_alpha_score = torch.sum(alpha_score, dim=-1).unsqueeze(dim=-1).repeat(1, 1, max_len)
        alpha = torch.div(alpha_score, sum_alpha_score + 1e-20)
        alpha_out = torch.matmul(alpha, pos_hidden).squeeze(1)
        aspect_out = self.get_avarage(mem_valid_ids, seq_out)

        aspect_out = torch.unsqueeze(aspect_out, dim=-1)
        alpha_out = torch.unsqueeze(alpha_out, dim=-1)
        combine_out = torch.cat([aspect_out, alpha_out], dim=-1)
        combine_out = torch.matmul(combine_out, F.softmax(self.ensemble_linear.weight, dim=0))
        combine_out = combine_out.squeeze(dim=-1)
        combine_out = self.dropout(combine_out)
        output = self.fc_single(combine_out)

        return output
