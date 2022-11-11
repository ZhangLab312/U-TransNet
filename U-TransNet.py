import  torch
from    torch import nn
from torch.nn import functional as F

def upsample(x, out_channel):
    return F.interpolate(x, size=out_channel, mode='linear', align_corners=False)

class chromatin_encoding(nn.Module):
    def __init__(self,motiflen=8):
        super(chromatin_encoding,self).__init__()
        self.trans1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128), num_layers=2)
        self.trans2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128), num_layers=2)
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=motiflen)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=motiflen)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()


    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # x = torch.transpose(x, 1, 2)
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
        return  res2, res3, res4

class U_TransNet_multi_DNA(nn.Module):
    def __init__(self, motiflen=8):
        super(U_TransNet_multi_DNA, self).__init__()
        # encoding
        self.trans1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128), num_layers=2)
        self.trans2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128), num_layers=2)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=motiflen)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=motiflen)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.averagepooling = nn.AdaptiveAvgPool1d(1)
        # decoding
        self.blend_conv4 = nn.Conv1d(64, 64, kernel_size=5, padding=5 // 2)
        self.blend_conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=5 // 2)
        self.blend_conv2 = nn.Conv1d(64, 4, kernel_size=5, padding=5 // 2)
        self.blend_conv1 =  nn.Conv1d(4, 1, kernel_size=5, padding=5 // 2)
        self.batchnorm64_4 = nn.BatchNorm1d(64)
        self.batchnorm64_3 = nn.BatchNorm1d(64)
        self.batchnorm64_2 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(4)
        # general functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, res22, res23, res24):

        x = torch.transpose(x, 1, 2)
        b, _, _ = x.size()
        # encoding
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

        res4 = res4+res24

        up5 = self.averagepooling(res4)

        up4 = upsample(up5, res4.size()[-1])
        up4 = up4 + res4 + res24
        up4 = self.batchnorm64_4(up4)
        up4 = self.relu(up4)
        up4 = self.blend_conv4(up4)
        up3 = upsample(up4, res3.size()[-1])
        up3 = up3 + res3 + res23
        up3 = self.batchnorm64_3(up3)
        up3 = self.relu(up3)
        up3 = self.blend_conv3(up3)
        up2 = upsample(up3, res2.size()[-1])
        up2 = up2 + res2 + res22

        up2 = self.batchnorm64_2(up2)
        up2 = self.relu(up2)
        up2 = self.blend_conv2(up2)

        up1 = upsample(up2, res1.size()[-1])

        up1 = self.batchnorm4(up1)
        up1 = self.relu(up1)
        out_dense = self.blend_conv1(up1)

        return out_dense

class U_TransNet_single(nn.Module):
   
    def __init__(self, motiflen=8):
        super(U_TransNet_single, self).__init__()
        # encoding
        self.trans1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128), num_layers=2)
        self.trans2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128), num_layers=2)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=motiflen)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=motiflen)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.averagepooling = nn.AdaptiveAvgPool1d(1)
        # decoding
        self.blend_conv4 = nn.Conv1d(64, 64, kernel_size=5, padding=5 // 2)
        self.blend_conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=5 // 2)
        self.blend_conv2 = nn.Conv1d(64, 4, kernel_size=5, padding=5 // 2)
        self.blend_conv1 =  nn.Conv1d(4, 1, kernel_size=5, padding=5 // 2)
        self.batchnorm64_4 = nn.BatchNorm1d(64)
        self.batchnorm64_3 = nn.BatchNorm1d(64)
        self.batchnorm64_2 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(4)
        # general functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()


    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

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