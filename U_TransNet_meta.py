import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_model)  # [max_len,d_model]
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))  # [1,d_model/2]
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # [1,max_len,d_model]
    def forward(self, x):  # x = [1,wordnum,d_model]

        x = x + Variable(self.pe[:, :x.size(1)].to(device),
                         requires_grad=False)
        return self.dropout(x)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, size, dropout, d_model, d_ff, h, pos_embed):
        super(Encoder, self).__init__()
        #embedding
        self.pos_embed = pos_embed
        #layernorm
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.layernorm4 = nn.LayerNorm(d_model)
        #multihead_attention
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h


        self.linear11 = nn.Linear(d_model, d_model)
        self.linear12 = nn.Linear(d_model, d_model)
        self.linear13 = nn.Linear(d_model, d_model)
        self.linear14 = nn.Linear(d_model, d_model)

        self.linear21 = nn.Linear(d_model, d_model)
        self.linear22 = nn.Linear(d_model, d_model)
        self.linear23 = nn.Linear(d_model, d_model)
        self.linear24 = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


        self.N = N

        # self.feed_forward = feed_forward
        self.w_11 = nn.Linear(d_model, d_ff) #encoder 1
        self.w_21 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.w_12 = nn.Linear(d_model, d_ff) #encoder 2
        self.w_22 = nn.Linear(d_ff, d_model)

        # self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # self.size = size

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        x = self.pos_embed(x)
        temp = x
        query, key, value = x, x, x
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.linear11(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear12(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear13(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = self.linear14(x)
        x = x + temp
        x = self.layernorm1(x)
        temp = x
        x = self.w_21(self.dropout(F.relu(self.w_11(x))))
        x = x + temp
        x = self.layernorm2(x)

        #encoder 2
        temp = x
        query, key, value = x, x, x
        if mask is not None:
            # Same mask applied to module h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)


        query = self.linear21(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear22(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear23(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)


        x = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = self.linear24(x)
        x = x + temp
        x = self.layernorm3(x)
        temp = x
        x = self.w_22(self.dropout(F.relu(self.w_12(x))))
        x = x + temp
        x = self.layernorm4(x)
        return x


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)#!!



def make_model(NTrans, d_model, d_ff, h, dropout=0.1):
    c = copy.deepcopy
    position = PositionalEncoding(1, dropout)
    model = Encoder(NTrans, d_model,  dropout, d_model, d_ff, h , c(position))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)

class U_TransNet(nn.Module):
    def __init__(self, motiflen=8):
        super(U_TransNet, self).__init__()
        self.trans1 = make_model(1,64,128,8)
        self.trans2 = make_model(1,64,128,8)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.averagepooling = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend_conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=3 // 2,  groups=1)
        self.blend_conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=3 // 2,  groups=1)
        self.blend_conv2 = nn.Conv1d(64, 4, kernel_size=3, stride=1, padding=3 // 2,  groups=1)
        self.blend_conv1 =  nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=3 // 2,  groups=1)
        self.batchnorm64_4 = nn.BatchNorm1d(64)
        self.batchnorm64_3 = nn.BatchNorm1d(64)
        self.batchnorm64_2 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(4)
        # general functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        res1 = x
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        res2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        res3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)
        out1_1 = self.trans1(out1)
        out1_2 = self.trans2(torch.flip(out1, [1]))
        out1 = out1_1 + out1_2
        out1 = self.dropout(out1)
        res4 = out1.permute(0, 2, 1)
        up5 = self.averagepooling(res4)
        # decoding
        up4 = upsample(up5, res4.size()[-1])
        up4 = up4 + res4
        up4 = self.batchnorm64_4(up4)
        up4 = self.relu(up4)
        up4 = self.blend_conv4(up4)
        up3 = upsample(up4, res3.size()[-1])
        up3 = up3 + res3
        up3 = self.batchnorm64_3(up3)
        up3 = self.relu(up3)
        up3 = self.blend_conv3(up3)
        up2 = upsample(up3, res2.size()[-1])
        up2 = up2 + res2

        up2 = self.batchnorm64_2(up2)
        up2 = self.relu(up2)
        up2 = self.blend_conv2(up2)

        up1 = upsample(up2, res1.size()[-1])

        up1 = self.batchnorm4(up1)
        up1 = self.relu(up1)
        out_dense = self.blend_conv1(up1)

        return out_dense

