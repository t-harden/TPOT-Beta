# coding = utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import copy
import itertools

class SurrogateNet(nn.Module):  #数据集元特征与每个operator先元素积后再输入AttentionNet得到数据集embedding，再与3个operator的原始embedding拼接
    def __init__(self, da_dim, op_dim, embed_size, DaAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(SurrogateNet, self).__init__()

        self.embed_size = embed_size
        "数据集向量embedding部分（Attention）"
        self.reduction_layer_da = nn.Sequential(nn.Linear(da_dim, embed_size), nn.BatchNorm1d(embed_size))
        # self.reduction_layer_da = nn.Linear(da_dim, embed_size)

        self.DaAttention_factor = DaAttention_factor
        self.attention_W = nn.Parameter(torch.Tensor(embed_size, self.DaAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.DaAttention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.DaAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        "三个operator向量embedding部分"
        self.reduction_layer_op1 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op2 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op3 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))

        "数据集向量和三个operator向量连接后的预测部分-加Batch Normalization版"
        in_dim = embed_size * 4
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))
        "不加Batch Normalization版"
        # self.layer1 = nn.Linear(in_dim, n_hidden_1)
        # self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        # self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        # self.layer4 = nn.Linear(n_hidden_3, n_hidden_4)
        # self.layer5 = nn.Linear(n_hidden_4, out_dim)

        "参数初始化"
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, input_da, input_ops):  # input_da是[batch-size,da_dim](256,111)的张量
                                             # input_ops是[batch-size,3,op_dim](256,3,768)的张量
        # print(input_da)
        # print(input_ops)
        "三个operator向量embedding部分"
        op_num = 3
        batch_size = input_ops.shape[0]
        op1_embed = F.relu(self.reduction_layer_op1(input_ops[:, 0, :])) # [256,768]->[256,100]
        op2_embed = F.relu(self.reduction_layer_op2(input_ops[:, 1, :]))  # [256,768]->[256,100]
        op3_embed = F.relu(self.reduction_layer_op3(input_ops[:, 2, :]))  # [256,768]->[256,100]
        # print("input_ops[:, 2, :]: ", input_ops[:, 2, :].shape)
        # print("op3_embed: ", op3_embed.shape)

        "数据集向量embedding部分（Attention）"
        #print("da_embed前:", input_da, input_da.shape)
        #print("da_embed中:", self.reduction_layer_da(input_da))
        #print(self.reduction_layer_da.named_parameters())
        # for name, params in self.reduction_layer_da.named_parameters():
        #     print(name, params, params.grad)
        #print(self.reduction_layer_da, self.reduction_layer_da.weight.grad)
        #print(self.reduction_layer_da.bias, self.reduction_layer_da.bias.grad)
        da_embed = F.relu(self.reduction_layer_da(input_da)) # [256,111]->[256,100]
        #print("da_embed后:", da_embed, da_embed.shape)
        # if torch.isnan(da_embed).any():
        #    print("da_embed后:", da_embed, da_embed.shape)
        #    exit()
        op_embeds = torch.cat((op1_embed.unsqueeze(0), op2_embed.unsqueeze(0), op3_embed.unsqueeze(0)), dim=0)  # [256,100]->[1,256,100]->[3,256,100]
        # if torch.isnan(op_embeds).any():
        #    print("op_embeds:", op_embeds)
        #    exit()
        attention_input = op_embeds * da_embed  # [3,256,100] * [256,100]= [3,256,100]
        # if torch.isnan(attention_input).any():
        #     print("attention_input:", attention_input)
        #     exit()
        # print("attention_input: ", attention_input.shape)  # [3,batch_size,100]
        attention_input = attention_input.reshape(batch_size, op_num, self.embed_size)  # 把注意力网络的输入重新变成[batch-size,operator数,embed_size]的形式
                                                                                        # [3,256,100]->[256,3,100]
        attention_temp = F.relu(torch.tensordot(attention_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(self.normalized_att_score * attention_input, dim=1) # [256,100]
        # if torch.isnan(attention_output).any():
        #     print("attention_output:", attention_output)
        #     exit()

        "数据集向量和三个operator向量连接后的预测部分"
        pred_input = torch.cat((attention_output, op1_embed, op2_embed, op3_embed),
                                dim=1)  # [batch-size,embed_size*4] [batch-size,400]
        # print("demand_embed：", demand_embed.shape)
        # print("pred_input：", pred_input.shape)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        'sigmoid形式'
        out = torch.sigmoid(self.layer5(hidden_4_out))
        #print("---", out, '---')
        # out = self.layer5(hidden_4_out)
        # print('---',out.shape)
        return out

class SurrogateNet_interaction(nn.Module):  # 数据集元特征与每个operator先元素积后再输入AttentionNet得到数据集embedding，
                                            # 交互方式一：3个operator里1与2交互、2与3交互，然后把数据集+前两者+3拼接
                                            # 交互方式二：3个operator两两交互后再与数据集拼接
    def __init__(self, da_dim, op_dim, embed_size, DaAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim, BiInteraction): #BiInteraction指示3个operator是否两两交互
        super(SurrogateNet_interaction, self).__init__()

        self.embed_size = embed_size
        self.BiInteraction = BiInteraction
        "数据集向量embedding部分（Attention）"
        self.reduction_layer_da = nn.Sequential(nn.Linear(da_dim, embed_size), nn.BatchNorm1d(embed_size))
        # self.reduction_layer_da = nn.Linear(da_dim, embed_size)

        self.DaAttention_factor = DaAttention_factor
        self.attention_W = nn.Parameter(torch.Tensor(embed_size, self.DaAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.DaAttention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.DaAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        "三个operator向量embedding和交互部分"
        self.reduction_layer_op1 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op2 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op3 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.interaction1 = nn.Sequential(nn.Linear(embed_size * 2, embed_size), nn.BatchNorm1d(embed_size))
        self.interaction2 = nn.Sequential(nn.Linear(embed_size * 2, embed_size), nn.BatchNorm1d(embed_size))
        if self.BiInteraction:
            self.interaction3 = nn.Sequential(nn.Linear(embed_size * 2, embed_size), nn.BatchNorm1d(embed_size))

        "数据集向量和三个交互向量连接后的预测部分-加Batch Normalization版"
        in_dim = embed_size * 4
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))

    def forward(self, input_da, input_ops):  # input_da是[batch-size,da_dim](256,111)的张量
                                             # input_ops是[batch-size,3,op_dim](256,3,768)的张量

        "三个operator向量embedding和交互部分"
        op_num = 3
        batch_size = input_ops.shape[0]
        op1_embed = F.relu(self.reduction_layer_op1(input_ops[:, 0, :])) # [256,768]->[256,100]
        op2_embed = F.relu(self.reduction_layer_op2(input_ops[:, 1, :]))  # [256,768]->[256,100]
        op3_embed = F.relu(self.reduction_layer_op3(input_ops[:, 2, :]))  # [256,768]->[256,100]

        # Operator 1 和 Operator 2 的交互
        interaction1_output = F.relu(self.interaction1(torch.cat((op1_embed, op2_embed), dim=1)))
        # Operator 2 和 Operator 3 的交互
        interaction2_output = F.relu(self.interaction2(torch.cat((op2_embed, op3_embed), dim=1)))
        # Operator 1 和 Operator 3 的交互
        if self.BiInteraction:
            interaction3_output = F.relu(self.interaction3(torch.cat((op1_embed, op3_embed), dim=1)))

        "数据集向量embedding部分（Attention）"
        da_embed = F.relu(self.reduction_layer_da(input_da)) # [256,111]->[256,100]
        #print("da_embed后:", da_embed, da_embed.shape)
        op_embeds = torch.cat((op1_embed.unsqueeze(0), op2_embed.unsqueeze(0), op3_embed.unsqueeze(0)), dim=0)  # [256,100]->[1,256,100]->[3,256,100]
        attention_input = op_embeds * da_embed  # [3,256,100] * [256,100]= [3,256,100]
        # print("attention_input: ", attention_input.shape)  # [3,batch_size,100]
        attention_input = attention_input.reshape(batch_size, op_num, self.embed_size)  # 把注意力网络的输入重新变成[batch-size,operator数,embed_size]的形式
                                                                                        # [3,256,100]->[256,3,100]
        attention_temp = F.relu(torch.tensordot(attention_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(self.normalized_att_score * attention_input, dim=1) # [256,100]


        "数据集向量和三个operator向量连接后的预测部分"
        if self.BiInteraction:
            pred_input = torch.cat((attention_output, interaction1_output, interaction2_output, interaction3_output),
                                   dim=1)  # [batch-size,embed_size*4] [batch-size,400]
        else:
            pred_input = torch.cat((attention_output, interaction1_output, interaction2_output, op3_embed),
                                dim=1)  # [batch-size,embed_size*4] [batch-size,400]
        # print("demand_embed：", demand_embed.shape)
        # print("pred_input：", pred_input.shape)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        'sigmoid形式'
        out = torch.sigmoid(self.layer5(hidden_4_out))
        #print("---", out, '---')
        # out = self.layer5(hidden_4_out)
        # print('---',out.shape)
        return out

class SurrogateNet_GRU(nn.Module):  # 数据集元特征与每个operator先元素积后再输入AttentionNet得到数据集embedding，
                                    # 3个operator经过一个GRU网络后得到pipeline的embedding
                                    # 最后把数据集和pipeline的embedding拼接起来送入MLP
    def __init__(self, da_dim, op_dim, embed_size, DaAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim, BiDirection):
        super(SurrogateNet_GRU, self).__init__()

        self.embed_size = embed_size
        "数据集向量embedding部分（Attention）"
        self.reduction_layer_da = nn.Sequential(nn.Linear(da_dim, embed_size), nn.BatchNorm1d(embed_size))
        # self.reduction_layer_da = nn.Linear(da_dim, embed_size)

        self.DaAttention_factor = DaAttention_factor
        self.attention_W = nn.Parameter(torch.Tensor(embed_size, self.DaAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.DaAttention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.DaAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        "三个operator向量embedding和GRU部分"
        self.reduction_layer_op1 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op2 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op3 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        #定义GRU网络
        if BiDirection:
            self.gru = nn.GRU(input_size=embed_size, hidden_size=int(embed_size/2), num_layers=2, bias=True,
            batch_first=True, dropout=0, bidirectional=True) #双向双层普通GRU
        else:
            self.gru = nn.GRU(input_size=embed_size, hidden_size=embed_size, num_layers=1, bias=True,
                            batch_first=True, dropout=0, bidirectional=False) #单向单层普通GRU

        for name, para in self.gru.named_parameters(): #orthogonal初始化
            if name.startswith("weight"):
                nn.init.orthogonal_(para)
            elif name.startswith("bias"):
                nn.init.constant_(para, 0)

        "数据集向量和三个operator的GRU向量连接后的预测部分-加Batch Normalization版"
        in_dim = embed_size * 2
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))


    def forward(self, input_da, input_ops):  # input_da是[batch-size,da_dim](256,111)的张量
                                             # input_ops是[batch-size,3,op_dim](256,3,768)的张量

        "三个operator向量embedding和GRU部分"
        op_num = 3
        batch_size = input_ops.shape[0]
        op1_embed = F.relu(self.reduction_layer_op1(input_ops[:, 0, :])) # [256,768]->[256,100]
        op2_embed = F.relu(self.reduction_layer_op2(input_ops[:, 1, :]))  # [256,768]->[256,100]
        op3_embed = F.relu(self.reduction_layer_op3(input_ops[:, 2, :]))  # [256,768]->[256,100]
        ops_embed = torch.cat((op1_embed.unsqueeze(1), op2_embed.unsqueeze(1), op3_embed.unsqueeze(1)), dim=1) # [256,100]->[256,1,100]->[256,3,100]
        gru_out, _ = self.gru(ops_embed) # 当batch_first是True时gru_out是[batch-size, seq_len, num_directions * hidden_size]，即[256,3,1*100]

        "数据集向量embedding部分（Attention）"
        da_embed = F.relu(self.reduction_layer_da(input_da)) # [256,111]->[256,100]
        #print("da_embed后:", da_embed, da_embed.shape)
        op_embeds = torch.cat((op1_embed.unsqueeze(0), op2_embed.unsqueeze(0), op3_embed.unsqueeze(0)), dim=0)  # [256,100]->[1,256,100]->[3,256,100]
        attention_input = op_embeds * da_embed  # [3,256,100] * [256,100]= [3,256,100]
        # print("attention_input: ", attention_input.shape)  # [3,batch_size,100]
        attention_input = attention_input.reshape(batch_size, op_num, self.embed_size)  # 把注意力网络的输入重新变成[batch-size,operator数,embed_size]的形式
                                                                                        # [3,256,100]->[256,3,100]
        attention_temp = F.relu(torch.tensordot(attention_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(self.normalized_att_score * attention_input, dim=1) # [256,100]

        "数据集向量和三个operator的GRU向量连接后的预测部分"
        pred_input = torch.cat((attention_output, gru_out[:, -1, :]),
                                dim=1)  # [batch-size,embed_size*2] [batch-size,200]
        # print("demand_embed：", demand_embed.shape)
        # print("pred_input：", pred_input.shape)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        'sigmoid形式'
        out = torch.sigmoid(self.layer5(hidden_4_out))
        return out

class SurrogateNet_Transformer(nn.Module):  # 数据集元特征与每个operator先元素积后再输入AttentionNet得到数据集embedding，
                                            # 3个operator经过一个transformer的Encoder网络后得到pipeline的embedding
                                            # 最后把数据集和pipeline的embedding拼接起来送入MLP


    def __init__(self, da_dim, op_dim, embed_size, DaAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                 out_dim, transformerout_avg):
        super(SurrogateNet_Transformer, self).__init__()

        self.embed_size = embed_size
        self.transformerout_avg = transformerout_avg #表示transformer的encoder输出是取序列内所有operator的平均还是只取最后一个
        "数据集向量embedding部分（Attention）"
        self.reduction_layer_da = nn.Sequential(nn.Linear(da_dim, embed_size), nn.BatchNorm1d(embed_size))
        # self.reduction_layer_da = nn.Linear(da_dim, embed_size)

        self.DaAttention_factor = DaAttention_factor
        self.attention_W = nn.Parameter(torch.Tensor(embed_size, self.DaAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.DaAttention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.DaAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        "三个operator向量embedding和Transformer部分"
        self.reduction_layer_op1 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op2 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.reduction_layer_op3 = nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size))
        # 定义Transformer的Encoder网络
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=2) # d_model：输入特征数量，nhead：多头注意力模型中的头数
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4) # num_layers：编码器中子编码器层数

        "数据集向量和三个operator的Transformer向量连接后的预测部分-加Batch Normalization版"
        in_dim = embed_size * 2
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))


    def forward(self, input_da, input_ops):  # input_da是[batch-size,da_dim](256,111)的张量
        # input_ops是[batch-size,3,op_dim](256,3,768)的张量

        "三个operator向量embedding和Transformer部分"
        op_num = 3
        batch_size = input_ops.shape[0]
        op1_embed = F.relu(self.reduction_layer_op1(input_ops[:, 0, :]))  # [256,768]->[256,100]
        op2_embed = F.relu(self.reduction_layer_op2(input_ops[:, 1, :]))  # [256,768]->[256,100]
        op3_embed = F.relu(self.reduction_layer_op3(input_ops[:, 2, :]))  # [256,768]->[256,100]
        ops_embed = torch.cat((op1_embed.unsqueeze(0), op2_embed.unsqueeze(0), op3_embed.unsqueeze(0)),
                              dim=0)  # [256,100]->[1,256,100]->[3,256,100]
        transformer_out = self.transformer_encoder(ops_embed) #out的形状和输入相同[S为源序列长度，N为batch size，E为特征]，即[3,256,100]

        "数据集向量embedding部分（Attention）"
        da_embed = F.relu(self.reduction_layer_da(input_da))  # [256,111]->[256,100]
        # print("da_embed后:", da_embed, da_embed.shape)
        op_embeds = torch.cat((op1_embed.unsqueeze(0), op2_embed.unsqueeze(0), op3_embed.unsqueeze(0)),
                              dim=0)  # [256,100]->[1,256,100]->[3,256,100]
        attention_input = op_embeds * da_embed  # [3,256,100] * [256,100]= [3,256,100]
        # print("attention_input: ", attention_input.shape)  # [3,batch_size,100]
        attention_input = attention_input.reshape(batch_size, op_num,
                                                  self.embed_size)  # 把注意力网络的输入重新变成[batch-size,operator数,embed_size]的形式
        # [3,256,100]->[256,3,100]
        attention_temp = F.relu(torch.tensordot(attention_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(self.normalized_att_score * attention_input, dim=1)  # [256,100]

        "数据集向量和三个operator的Transformer向量连接后的预测部分"
        if self.transformerout_avg:
            pip_embed = torch.mean(transformer_out, dim=0, keepdim=True).squeeze(0) #[3,256,100]->[1,256,100]->[256,100]
        else:
            pip_embed = transformer_out[-1, :, :]
        pred_input = torch.cat((attention_output, pip_embed), dim=1)  # [batch-size,embed_size*2] [batch-size,200]
        # print("demand_embed：", demand_embed.shape)
        # print("pred_input：", pred_input.shape)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        'sigmoid形式'
        out = torch.sigmoid(self.layer5(hidden_4_out))
        return out