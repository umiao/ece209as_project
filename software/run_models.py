import argparse
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import nni
from torch.utils.tensorboard import SummaryWriter
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import h5py
from torch.utils.data import Dataset, DataLoader
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
sys.path.append('..')

class Args(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            return AttributeError(name)

    def __setattr__(self, key, value):
        self[key] = value


def masked_mse_cal(inputs, target, mask):
    """ calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""
    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)


    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs['diagonal_attention_mask']
        self.device = kwargs['device']
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']

        if kwargs['param_sharing_strategy'] == 'between_group':
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_groups)
            ])

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        input_X = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = masks * X + (1 - masks) * learned_presentation  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)
        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(learned_presentation, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_MAE, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': reconstruction_MAE, 'imputation_MAE': imputation_MAE}


class SAITS(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']

        if kwargs['param_sharing_strategy'] == 'between_group':
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_groups)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_groups)
            ])

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely term e in math algo
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze()  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(self.weight_combine(torch.cat([masks, attn_weights], dim=2)))  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(X_tilde_3, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': final_reconstruction_MAE, 'imputation_MAE': imputation_MAE}



class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(RITS, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # other hyper parameters
        self.device = kwargs['device']
        self.MIT = kwargs['MIT']

        # create models
        self.rnn_cell = nn.LSTMCell(self.feature_num * 2, self.rnn_hidden_size)
        # # Temporal Decay here is used to decay the hidden state
        self.temp_decay_h = TemporalDecay(input_size=self.feature_num, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.feature_num, output_size=self.feature_num, diag=True)
        # # History regression and feature regression layer
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.feature_num)
        self.feat_reg = FeatureRegression(self.feature_num)
        # # weight-combine is used to combine history regression and feature regression
        self.weight_combine = nn.Linear(self.feature_num * 2, self.feature_num)

    def impute(self, data, direction):
        values = data[direction]['X']
        masks = data[direction]['missing_mask']
        deltas = data[direction]['deltas']

        # use device of input values
        hidden_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)
        cell_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)

        estimations = []
        reconstruction_loss = 0.0
        reconstruction_MAE = 0.0

        # imputation period
        for t in range(self.seq_len):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += masked_mae_cal(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += masked_mae_cal(z_h, x, m)

            alpha = F.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_MAE += masked_mae_cal(c_h, x, m)
            reconstruction_loss += reconstruction_MAE

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(inputs, (hidden_states, cell_states))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [reconstruction_MAE, reconstruction_loss]

    def forward(self, data, direction='forward'):
        imputed_data, [reconstruction_MAE, reconstruction_loss] = self.impute(data, direction)
        reconstruction_MAE /= self.seq_len
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= (self.seq_len * 3)

        ret_dict = {
            'consistency_loss': torch.tensor(0.0, device=self.device),  # single direction, has no consistency loss
            'reconstruction_loss': reconstruction_loss, 'reconstruction_MAE': reconstruction_MAE,
            'imputed_data': imputed_data,
        }
        if 'X_holdout' in data:
            ret_dict['X_holdout'] = data['X_holdout']
            ret_dict['indicating_mask'] = data['indicating_mask']
        return ret_dict


class BRITS(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(BRITS, self).__init__()
        self.MIT = kwargs['MIT']
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(seq_len, feature_num, rnn_hidden_size, **kwargs)
        self.rits_b = RITS(seq_len, feature_num, rnn_hidden_size, **kwargs)

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def merge_ret(self, ret_f, ret_b, stage):
        consistency_loss = self.get_consistency_loss(ret_f['imputed_data'], ret_b['imputed_data'])
        imputed_data = (ret_f['imputed_data'] + ret_b['imputed_data']) / 2
        reconstruction_loss = (ret_f['reconstruction_loss'] + ret_b['reconstruction_loss']) / 2
        reconstruction_MAE = (ret_f['reconstruction_MAE'] + ret_b['reconstruction_MAE']) / 2
        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(imputed_data, ret_f['X_holdout'], ret_f['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)
        imputation_loss = imputation_MAE

        ret_f['imputed_data'] = imputed_data
        ret_f['consistency_loss'] = consistency_loss
        ret_f['reconstruction_loss'] = reconstruction_loss
        ret_f['reconstruction_MAE'] = reconstruction_MAE
        ret_f['imputation_MAE'] = imputation_MAE
        ret_f['imputation_loss'] = imputation_loss
        return ret_f


    def impute(self, data):
        imputed_data_f, _ = self.rits_f.impute(data, 'forward')
        imputed_data_b, _ = self.rits_b.impute(data, 'backward')
        imputed_data_b = {'imputed_data_b': imputed_data_b}
        imputed_data_b = self.reverse(imputed_data_b)['imputed_data_b']
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data, [imputed_data_f, imputed_data_b]


    def forward(self, data, stage):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))
        ret = self.merge_ret(ret_f, ret_b, stage)
        return ret



class FCN_Regression(nn.Module):
    def __init__(self, feature_num, rnn_hid_size):
        super(FCN_Regression, self).__init__()
        self.feat_reg = FeatureRegression(rnn_hid_size * 2)
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))  # bias beta
        self.final_linear = nn.Linear(feature_num, feature_num)

        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x_t, m_t, target):
        h_t = F.tanh(
            F.linear(x_t, self.U * self.m) +
            F.linear(target, self.V1 * self.m) +
            F.linear(m_t, self.V2) +
            self.beta
        )
        x_hat_t = self.final_linear(h_t)
        return x_hat_t


class MRNN(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(MRNN, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = kwargs['device']

        self.f_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.rnn_cells = {'forward': self.f_rnn,
                          'backward': self.b_rnn}
        self.concated_hidden_project = nn.Linear(self.rnn_hidden_size * 2, self.feature_num)
        self.fcn_regression = FCN_Regression(feature_num, rnn_hidden_size)

    def gene_hidden_states(self, data, direction):
        values = data[direction]['X']
        masks = data[direction]['missing_mask']
        deltas = data[direction]['deltas']

        hidden_states_collector = []
        hidden_state = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            inputs = torch.cat([x, m, d], dim=1)
            hidden_state = self.rnn_cells[direction](inputs, hidden_state)
            hidden_states_collector.append(hidden_state)
        return hidden_states_collector

    def impute(self, data):
        hidden_states_f = self.gene_hidden_states(data, 'forward')
        hidden_states_b = self.gene_hidden_states(data, 'backward')[::-1]

        values = data['forward']['X']
        masks = data['forward']['missing_mask']

        reconstruction_loss = 0
        estimations = []
        for i in range(self.seq_len):  # calculating estimation loss for times can obtain better results than once
            x = values[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(x, m, RNN_imputed_data)  # FCN estimation is output extimation
            reconstruction_loss += masked_rmse_cal(FCN_estimation, x, m) + masked_rmse_cal(RNN_estimation, x, m)
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [estimations, reconstruction_loss]

    def forward(self, data, stage):
        values = data['forward']['X']
        masks = data['forward']['missing_mask']
        imputed_data, [estimations, reconstruction_loss] = self.impute(data)
        reconstruction_loss /= self.seq_len
        reconstruction_MAE = masked_mae_cal(estimations.detach(), values, masks)

        if stage == 'val':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(imputed_data, data['X_holdout'], data['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        ret_dict = {
            'reconstruction_loss': reconstruction_loss, 'reconstruction_MAE': reconstruction_MAE,
            'imputation_loss': imputation_MAE, 'imputation_MAE': imputation_MAE,
            'imputed_data': imputed_data,
        }
        if 'X_holdout' in data:
            ret_dict['X_holdout'] = data['X_holdout']
            ret_dict['indicating_mask'] = data['indicating_mask']
        return ret_dict


def parse_delta(masks, seq_len, feature_num):
    """generate deltas from masks, used in BRITS"""
    deltas = []
    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_num))
        else:
            deltas.append(np.ones(feature_num) + (1 - masks[h]) * deltas[-1])
    return np.asarray(deltas)


def fill_with_last_observation(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    out = np.nan_to_num(out)  # if nan still exists then fill with 0
    return out


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type


class LoadValTestDataset(LoadDataset):
    """Loading process of val or test set"""
    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
            self.X_hat = hf[set_name]['X_hat'][:]
            self.missing_mask = hf[set_name]['missing_mask'][:]
            self.indicating_mask = hf[set_name]['indicating_mask'][:]

        # fill missing values with 0
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ['Transformer', 'SAITS']:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_hat[idx].astype('float32')),
                torch.from_numpy(self.missing_mask[idx].astype('float32')),
                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.indicating_mask[idx].astype('float32')),
            )
        elif self.model_type in ['BRITS', 'MRNN']:
            forward = {'X_hat': self.X_hat[idx], 'missing_mask': self.missing_mask[idx],
                       'deltas': parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num)}
            backward = {'X_hat': np.flip(forward['X_hat'], axis=0).copy(),
                        'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
            backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward['X_hat'].astype('float32')),
                torch.from_numpy(forward['missing_mask'].astype('float32')),
                torch.from_numpy(forward['deltas'].astype('float32')),
                # for backward
                torch.from_numpy(backward['X_hat'].astype('float32')),
                torch.from_numpy(backward['missing_mask'].astype('float32')),
                torch.from_numpy(backward['deltas'].astype('float32')),

                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.indicating_mask[idx].astype('float32')),
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'
        return sample


class LoadTrainDataset(LoadDataset):
    """Loading process of train set"""
    def __init__(self, file_path, seq_len, feature_num, model_type, masked_imputation_task):
        super(LoadTrainDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.2
            assert 0 < self.artificial_missing_rate < 1, 'artificial_missing_rate should be greater than 0 and less than 1'

        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf['train']['X'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.masked_imputation_task:
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
            X_hat = np.copy(X)
            X_hat[indices] = np.nan  # mask values selected by indices
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
            X = np.nan_to_num(X)
            X_hat = np.nan_to_num(X_hat)
            # reshape into time series
            X = X.reshape(self.seq_len, self.feature_num)
            X_hat = X_hat.reshape(self.seq_len, self.feature_num)
            missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
            indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)

            if self.model_type in ['Transformer', 'SAITS']:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X_hat.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32')),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(indicating_mask.astype('float32')),
                )
            elif self.model_type in ['BRITS', 'MRNN']:
                forward = {'X_hat': X_hat, 'missing_mask': missing_mask,
                           'deltas': parse_delta(missing_mask, self.seq_len, self.feature_num)}

                backward = {'X_hat': np.flip(forward['X_hat'], axis=0).copy(),
                            'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
                backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward['X_hat'].astype('float32')),
                    torch.from_numpy(forward['missing_mask'].astype('float32')),
                    torch.from_numpy(forward['deltas'].astype('float32')),
                    # for backward
                    torch.from_numpy(backward['X_hat'].astype('float32')),
                    torch.from_numpy(backward['missing_mask'].astype('float32')),
                    torch.from_numpy(backward['deltas'].astype('float32')),

                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(indicating_mask.astype('float32')),
                )
            else:
                assert ValueError, f'Error model type: {self.model_type}'
        else:
            # if training without masked imputation task, then there is no need to artificially mask out observed values
            missing_mask = (~np.isnan(X)).astype(np.float32)
            X = np.nan_to_num(X)
            if self.model_type in ['Transformer', 'SAITS']:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32')),
                )
            elif self.model_type in ['BRITS', 'MRNN']:
                forward = {'X': X, 'missing_mask': missing_mask,
                           'deltas': parse_delta(missing_mask, self.seq_len, self.feature_num)}
                backward = {'X': np.flip(forward['X'], axis=0).copy(),
                            'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
                backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward['X'].astype('float32')),
                    torch.from_numpy(forward['missing_mask'].astype('float32')),
                    torch.from_numpy(forward['deltas'].astype('float32')),
                    # for backward
                    torch.from_numpy(backward['X'].astype('float32')),
                    torch.from_numpy(backward['missing_mask'].astype('float32')),
                    torch.from_numpy(backward['deltas'].astype('float32')),
                )
            else:
                assert ValueError, f'Error model type: {self.model_type}'
        return sample


class LoadDataForImputation(LoadDataset):
    """Load all data for imputation, we don't need do any artificial mask here,
    just input original data into models and let them impute missing values"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadDataForImputation, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
        self.missing_mask = (~np.isnan(self.X)).astype(np.float32)
        self.X = np.nan_to_num(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ['Transformer', 'SAITS']:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.missing_mask[idx].astype('float32'))
            )
        elif self.model_type in ['BRITS', 'MRNN']:
            forward = {'X': self.X[idx], 'missing_mask': self.missing_mask[idx],
                       'deltas': parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num)}

            backward = {'X': np.flip(forward['X'], axis=0).copy(),
                        'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
            backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward['X'].astype('float32')),
                torch.from_numpy(forward['missing_mask'].astype('float32')),
                torch.from_numpy(forward['deltas'].astype('float32')),
                # for backward
                torch.from_numpy(backward['X'].astype('float32')),
                torch.from_numpy(backward['missing_mask'].astype('float32')),
                torch.from_numpy(backward['deltas'].astype('float32')),
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'
        return sample


class UnifiedDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, model_type, batch_size=1024, num_workers=4,
                 masked_imputation_task=False):
        """
        dataset_path: path of directory storing h5 dataset;
        seq_len: sequence length, i.e. time steps;
        feature_num: num of features, i.e. feature dimensionality;
        batch_size: size of mini batch;
        num_workers: num of subprocesses for data loading;
        model_type: model type, determine returned values;
        masked_imputation_task: whether to return data for masked imputation task, only for training/validation sets;
        """
        self.dataset_path = os.path.join(dataset_path, 'datasets.h5')
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num, self.model_type,
                                              self.masked_imputation_task)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num, self.model_type)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(self.dataset_path, 'test', self.seq_len, self.feature_num,
                                               self.model_type)
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        data_for_imputation = LoadDataForImputation(self.dataset_path, set_name, self.seq_len, self.feature_num,
                                                    self.model_type)
        dataloader_for_imputation = DataLoader(data_for_imputation, self.batch_size, shuffle=False)
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        train_set_for_imputation = self.prepare_dataloader_for_imputation('train')
        val_set_for_imputation = self.prepare_dataloader_for_imputation('val')
        test_set_for_imputation = self.prepare_dataloader_for_imputation('test')
        return train_set_for_imputation, val_set_for_imputation, test_set_for_imputation

plt.rcParams['savefig.dpi'] = 300  # pixel
plt.rcParams['figure.dpi'] = 300  # resolution
plt.rcParams["figure.figsize"] = [8, 4]  # figure size


def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """ calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """ calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area


def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    """
    pos_label: The label of the positive class.
    """
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert 'args.class_num>2, class need to be specified for precision_recall_fscore_support'
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, class_predictions,
                                                                       pos_label=pos_label, warn_for=())
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(labels, probabilities[:, -1], pos_label=pos_label)
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        'classification_predictions': class_predictions,
        'acc_score': acc_score, 'precision': precision, 'recall': recall, 'f1': f1,
        'precisions': precisions, 'recalls': recalls, 'fprs': fprs, 'tprs': tprs,
        'ROC_AUC': ROC_AUC, 'PR_AUC': PR_AUC,
    }
    return classification_metrics


def plot_AUCs(pdf_file, x_values, y_values, auc_value, title, x_name, y_name, dataset_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, '.', label=f'{dataset_name}, AUC={auc_value:.3f}', rasterized=True)
    l = ax.legend(fontsize=10, loc='lower left')
    l.set_zorder(20)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title, fontsize=12)
    pdf_file.savefig(fig)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


class Controller:
    def __init__(self, early_stop_patience):
        self.original_early_stop_patience_value = early_stop_patience
        self.early_stop_patience = early_stop_patience
        self.state_dict = {
            # `step` is for training stage
            'train_step': 0,
            # below are for validation stage
            'val_step': 0,
            'epoch': 0,
            'best_imputation_MAE': 1e9,
            'should_stop': False,
            'save_model': False
        }

    def epoch_num_plus_1(self):
        self.state_dict['epoch'] += 1

    def __call__(self, stage, info=None):
        if stage == 'train':
            self.state_dict['train_step'] += 1
        else:
            self.state_dict['val_step'] += 1
            self.state_dict['save_model'] = False
            current_imputation_MAE = info['imputation_MAE']
            imputation_MAE_dropped = False  # flags to decrease early stopping patience

            # update best_loss
            if current_imputation_MAE < self.state_dict['best_imputation_MAE']:
                self.state_dict['best_imputation_MAE'] = current_imputation_MAE
                imputation_MAE_dropped = True
            if imputation_MAE_dropped:
                self.state_dict['save_model'] = True

            if self.state_dict['save_model']:
                self.early_stop_patience = self.original_early_stop_patience_value
            else:
                # if use early_stopping, then update its patience
                if self.early_stop_patience > 0:
                    self.early_stop_patience -= 1
                elif self.early_stop_patience == 0:
                    self.state_dict['should_stop'] = True  # to stop training process
                else:
                    pass  # which means early_stop_patience_value is set as -1, not work
        return self.state_dict


def check_saving_dir_for_model(args, time_now):
    saving_path = os.path.join(args.result_saving_base_dir, args.model_name)
    if not args.test_mode:
        log_saving = os.path.join(saving_path, 'logs')
        model_saving = os.path.join(saving_path, 'models')
        sub_model_saving = os.path.join(model_saving, time_now)
        [os.makedirs(dir_) for dir_ in [model_saving, log_saving, sub_model_saving] if not os.path.exists(dir_)]
        return sub_model_saving, log_saving
    else:
        log_saving = os.path.join(saving_path, 'test_log')
        return None, log_saving


def save_model(model, optimizer, model_state_info, args, saving_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(), # don't save optimizer, considering GANs have 2 optimizers
        'training_step': model_state_info['train_step'],
        'epoch': model_state_info['epoch'],
        'model_state_info': model_state_info,
        'args': args
    }
    torch.save(checkpoint, saving_path)


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_model_saved_with_module(model, checkpoint_path):
    """
    To load models those are trained in parallel and saved with module (need to remove 'module.'
    """
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = dict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model







RANDOM_SEED = 26

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings('ignore')  # if to ignore warnings

MODEL_DICT = {
    # Self-Attention (SA) based
    'Transformer': TransformerEncoder,
    'SAITS': SAITS,
    # RNN based
    'BRITS': BRITS, 'MRNN': MRNN,
}
OPTIMIZER = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}


def summary_write_into_tb(summary_writer, info_dict, step, stage):
    """write summary into tensorboard file"""
    summary_writer.add_scalar(f'total_loss/{stage}', info_dict['total_loss'], step)
    summary_writer.add_scalar(f'imputation_loss/{stage}', info_dict['imputation_loss'], step)
    summary_writer.add_scalar(f'imputation_MAE/{stage}', info_dict['imputation_MAE'], step)
    summary_writer.add_scalar(f'reconstruction_loss/{stage}', info_dict['reconstruction_loss'], step)
    summary_writer.add_scalar(f'reconstruction_MAE/{stage}', info_dict['reconstruction_MAE'], step)


def result_processing(results):
    """process results and losses for each training step"""
    results['total_loss'] = torch.tensor(0.0, device=args.device)
    if args.model_type == 'BRITS':
        results['total_loss'] = results['consistency_loss'] * args.consistency_loss_weight
    results['reconstruction_loss'] = results['reconstruction_loss'] * args.reconstruction_loss_weight
    results['imputation_loss'] = results['imputation_loss'] * args.imputation_loss_weight
    if args.MIT:
        results['total_loss'] += results['imputation_loss']
    if args.ORT:
        results['total_loss'] += results['reconstruction_loss']
    return results


def process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer):
    """process each training step and return whether to early stop"""
    state_dict = training_controller(stage='train')
    # apply gradient clipping if args.max_norm != 0
    if args.max_norm != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    results['total_loss'].backward()
    optimizer.step()

    summary_write_into_tb(summary_writer, results, state_dict['train_step'], 'train')
    if state_dict['train_step'] % args.eval_every_n_steps == 0:
        state_dict_from_val = validate(model, val_dataloader, summary_writer, training_controller)
        if state_dict_from_val['should_stop']:
            return True
    return False


def model_processing(data, model, stage,
                     # following arguments are only required in the training stage
                     optimizer=None, val_dataloader=None, summary_writer=None, training_controller=None):
    if stage == 'train':
        optimizer.zero_grad()
        if not args.MIT:
            if args.model_type in ['BRITS', 'MRNN']:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                    map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            else:  # then for self-attention based models, i.e. Transformer/SAITS
                indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask}
            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller,
                                                        summary_writer)
        else:
            if args.model_type in ['BRITS', 'MRNN']:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
                indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                          'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            else:
                indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                          'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(results, optimizer, val_dataloader,
                                                        training_controller, summary_writer)
        return early_stopping

    else:  # in val/test stage
        if args.model_type in ['BRITS', 'MRNN']:
            indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
            indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                      'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                      'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            inputs['missing_mask'] = inputs['forward']['missing_mask']  # for error calculation in validation stage
        else:
            indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                      'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
        results = model(inputs, stage)
        results = result_processing(results)
        return inputs, results


def train(model, optimizer, train_dataloader, test_dataloader, summary_writer, training_controller):
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False
        for idx, data in enumerate(train_dataloader):
            model.train()
            early_stopping = model_processing(data, model, 'train', optimizer, test_dataloader, summary_writer,
                                              training_controller)
            if early_stopping:
                break
        if early_stopping:
            break
        training_controller.epoch_num_plus_1()
    return


def validate(model, val_iter, summary_writer, training_controller):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    total_loss_collector, imputation_loss_collector, reconstruction_loss_collector, reconstruction_MAE_collector = [], [], [], []

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, 'val')
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

            total_loss_collector.append(results['total_loss'].data.cpu().numpy())
            reconstruction_MAE_collector.append(results['reconstruction_MAE'].data.cpu().numpy())
            reconstruction_loss_collector.append(results['reconstruction_loss'].data.cpu().numpy())
            imputation_loss_collector.append(results['imputation_loss'].data.cpu().numpy())

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
    info_dict = {'total_loss': np.asarray(total_loss_collector).mean(),
                 'reconstruction_loss': np.asarray(reconstruction_loss_collector).mean(),
                 'imputation_loss': np.asarray(imputation_loss_collector).mean(),
                 'reconstruction_MAE': np.asarray(reconstruction_MAE_collector).mean(),
                 'imputation_MAE': imputation_MAE.cpu().numpy().mean()}
    state_dict = training_controller('val', info_dict)
    summary_write_into_tb(summary_writer, info_dict, state_dict['val_step'], 'val')
    if args.param_searching_mode:
        nni.report_intermediate_result(info_dict['imputation_MAE'])
        if args.final_epoch or state_dict['should_stop']:
            nni.report_final_result(state_dict['best_imputation_MAE'])

    if (state_dict['save_model'] and args.model_saving_strategy) or args.model_saving_strategy == 'all':
        saving_path = os.path.join(
            args.model_saving, 'model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}'.
                format(state_dict['train_step'], state_dict['val_step'], info_dict['imputation_MAE']))
        save_model(model, optimizer, state_dict, args, saving_path)
    return state_dict


def test_trained_model(model, test_dataloader):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            inputs, results = model_processing(data, model, 'test')
            # collect X_holdout, indicating_mask and imputed data
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)

    assessment_metrics = {'imputation_MAE on the test set': imputation_MAE,
                          'imputation_RMSE on the test set': imputation_RMSE,
                          'imputation_MRE on the test set': imputation_MRE,
                          'trainable parameter num': args.total_params}
    with open(os.path.join(args.result_saving_path, 'overall_performance_metrics.out'), 'w') as f:
        for k, v in assessment_metrics.items():
            f.write(k + ':' + str(v))
            f.write('\n')


def impute_all_missing_data(model, train_data, val_data, test_data):
    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                if args.model_type in ['BRITS', 'MRNN']:
                    indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                        map(lambda x: x.to(args.device), data)
                    inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                              'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
                else:  # then for self-attention based models, i.e. Transformer/SAITS
                    indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                    inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask}
                imputed_data, _ = model.impute(inputs)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)

            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputations_collector = torch.cat(imputations_collector)
            imputations = imputations_collector.data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]  # to ensure the order of samples
            imputed_data_dict[set_name] = ordered

    imputation_saving_path = os.path.join(args.result_saving_path, 'imputations.h5')
    with h5py.File(imputation_saving_path, 'w') as hf:
        hf.create_dataset('imputed_train_set', data=imputed_data_dict['train'])
        hf.create_dataset('imputed_val_set', data=imputed_data_dict['val'])
        hf.create_dataset('imputed_test_set', data=imputed_data_dict['test'])
    return


def read_arguments(arg_parser, cfg_parser):
    # file path
    arg_parser.dataset_base_dir = cfg_parser.get('file_path', 'dataset_base_dir')
    arg_parser.result_saving_base_dir = cfg_parser.get('file_path', 'result_saving_base_dir')
    # dataset info
    arg_parser.seq_len = cfg_parser.getint('dataset', 'seq_len')
    arg_parser.batch_size = cfg_parser.getint('dataset', 'batch_size')
    arg_parser.num_workers = cfg_parser.getint('dataset', 'num_workers')
    arg_parser.feature_num = cfg_parser.getint('dataset', 'feature_num')
    arg_parser.dataset_name = cfg_parser.get('dataset', 'dataset_name')
    arg_parser.dataset_path = os.path.join(arg_parser.dataset_base_dir, arg_parser.dataset_name)
    arg_parser.eval_every_n_steps = cfg_parser.getint('dataset', 'eval_every_n_steps')
    # training settings
    arg_parser.MIT = cfg_parser.getboolean('training', 'MIT')
    arg_parser.ORT = cfg_parser.getboolean('training', 'ORT')
    arg_parser.lr = cfg_parser.getfloat('training', 'lr')
    arg_parser.optimizer_type = cfg_parser.get('training', 'optimizer_type')
    arg_parser.weight_decay = cfg_parser.getfloat('training', 'weight_decay')
    arg_parser.device = cfg_parser.get('training', 'device')
    arg_parser.epochs = cfg_parser.getint('training', 'epochs')
    arg_parser.early_stop_patience = cfg_parser.getint('training', 'early_stop_patience')
    arg_parser.model_saving_strategy = cfg_parser.get('training', 'model_saving_strategy')
    arg_parser.max_norm = cfg_parser.getfloat('training', 'max_norm')
    arg_parser.imputation_loss_weight = cfg_parser.getfloat('training', 'imputation_loss_weight')
    arg_parser.reconstruction_loss_weight = cfg_parser.getfloat('training', 'reconstruction_loss_weight')
    # model settings
    arg_parser.model_name = cfg_parser.get('model', 'model_name')
    arg_parser.model_type = cfg_parser.get('model', 'model_type')
    return arg_parser


if __name__ == '__main__':
    config_path = r'/content/drive/MyDrive/SAITS-master/configs/PhysioNet2012_MRNN_best.ini'
    test_mode = False
    param_searching_mode = True

    args = Args({'config_path': config_path, 'test_mode': test_mode,
                 'param_searching_mode': param_searching_mode})

    # load settings from config file
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)

    if args.model_type in ['Transformer', 'SAITS']:  # if SA-based model
        args.input_with_mask = cfg.getboolean('model', 'input_with_mask')
        args.n_groups = cfg.getint('model', 'n_groups')
        args.n_group_inner_layers = cfg.getint('model', 'n_group_inner_layers')
        args.param_sharing_strategy = cfg.get('model', 'param_sharing_strategy')
        assert args.param_sharing_strategy in ['inner_group', 'between_group'], \
            'only "inner_group"/"between_group" sharing'
        args.d_model = cfg.getint('model', 'd_model')
        args.d_inner = cfg.getint('model', 'd_inner')
        args.n_head = cfg.getint('model', 'n_head')
        args.d_k = cfg.getint('model', 'd_k')
        args.d_v = cfg.getint('model', 'd_v')
        args.dropout = cfg.getfloat('model', 'dropout')
        args.diagonal_attention_mask = cfg.getboolean('model', 'diagonal_attention_mask')

        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f'{args.model_name}/{experiment_id}/{trial_id}'
            dict_args['d_k'] = dict_args['d_model'] // dict_args['n_head']
        model_args = {
            'device': args.device, 'MIT': args.MIT,
            # imputer args
            'n_groups': dict_args['n_groups'], 'n_group_inner_layers': args.n_group_inner_layers,
            'd_time': args.seq_len, 'd_feature': args.feature_num, 'dropout': dict_args['dropout'],
            'd_model': dict_args['d_model'], 'd_inner': dict_args['d_inner'], 'n_head': dict_args['n_head'],
            'd_k': dict_args['d_k'], 'd_v': dict_args['d_v'],
            'input_with_mask': args.input_with_mask,
            'diagonal_attention_mask': args.diagonal_attention_mask,
            'param_sharing_strategy': args.param_sharing_strategy,
        }
    elif args.model_type in ['BRITS', 'MRNN']:  # if RNN-based model
        if args.model_type == 'BRITS':
            args.consistency_loss_weight = cfg.getfloat('training', 'consistency_loss_weight')
        args.rnn_hidden_size = cfg.getint('model', 'rnn_hidden_size')

        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f'{args.model_name}/{experiment_id}/{trial_id}'
        model_args = {
            'device': args.device, 'MIT': args.MIT,
            # imputer args
            'seq_len': args.seq_len, 'feature_num': args.feature_num,
            'rnn_hidden_size': dict_args['rnn_hidden_size']
        }
    else:
        assert ValueError, f'Given model_type {args.model_type} is not in {MODEL_DICT.keys()}'

    # parameter insurance
    assert args.model_saving_strategy.lower() in ['all', 'best', 'none'], 'model saving strategy must be all/best/none'
    if args.model_saving_strategy.lower() == 'none':
        args.model_saving_strategy = False
    assert args.optimizer_type in OPTIMIZER.keys(), \
        f'optimizer type should be in {OPTIMIZER.keys()}, but get{args.optimizer_type}'
    assert args.device in ['cpu', 'cuda'], 'device should be cpu or cuda'

    time_now = datetime.now().__format__('%Y-%m-%d_T%H:%M:%S')
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now)

    unified_dataloader = UnifiedDataLoader(args.dataset_path, args.seq_len, args.feature_num, args.model_type,
                                           args.batch_size, args.num_workers, args.MIT)
    model = MODEL_DICT[args.model_type](**model_args)
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # if utilize GPU and GPU available, then move
    if 'cuda' in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if args.test_mode:
        args.model_path = cfg.get('test', 'model_path')
        args.save_imputations = cfg.getboolean('test', 'save_imputations')
        args.result_saving_path = cfg.get('test', 'result_saving_path')
        os.makedirs(args.result_saving_path) if not os.path.exists(args.result_saving_path) else None
        model = load_model(model, args.model_path)
        test_dataloader = unified_dataloader.get_test_dataloader()
        test_trained_model(model, test_dataloader)
        if args.save_imputations:
            train_data, val_data, test_data = unified_dataloader.prepare_all_data_for_imputation()
            impute_all_missing_data(model, train_data, val_data, test_data)
    else:  # in the training mode
        optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=dict_args['lr'],
                                                   weight_decay=args.weight_decay)
        train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
        training_controller = Controller(args.early_stop_patience)

        train_set_size = unified_dataloader.train_set_size

        tb_summary_writer = SummaryWriter(os.path.join(args.log_saving, 'tensorboard_' + time_now))
        train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller)

