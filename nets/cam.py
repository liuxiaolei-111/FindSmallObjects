import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.model import  Conv_Bn_Activation, Upsample ,ASPP
from process.attheat import attheat, att

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # self.mlp = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Linear(gate_channels // reduction_ratio, gate_channels)
        #     )
        self.fc = nn.Sequential(nn.Conv2d(gate_channels, gate_channels, 1))

        self.conv = nn.Conv2d(gate_channels, gate_channels, 1) #
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.fc( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.fc( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw =self.fc( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.fc( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # att(attheat(channel_att_sum, 'cam*'), 'cam*_att2')
        scale = torch.sigmoid( channel_att_sum ).expand_as(x)
        return self.conv(x * scale)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )



class CAMmy(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max']):
        super(CAMmy, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, pool_types)

    def forward(self, x):

        return self.ChannelGate(x)


class CAM(nn.Module):
    def __init__(self, channels= [128,256, 512, 1024]):
        super(CAM, self).__init__()
        self.ASPP1 = ASPP(512,256)
        self.upsample = Upsample()
        self.conv = nn.ModuleList()
        self.conv.append(Conv_Bn_Activation(channels[3], channels[0], 1, 1, activation='mish'))
        self.conv.append(Conv_Bn_Activation(channels[2], channels[0], 1, 1, activation='mish'))
        self.conv.append(Conv_Bn_Activation(channels[1], channels[0], 1, 1, activation='mish'))


    def forward(self, input):
        cat = []
        for i, conv in enumerate(self.conv):
            cat.append(self.upsample(conv(input[i]), input[3].size()))
        cat.append(input[3])
        cat = torch.cat(cat, dim=1)

        return self.ASPP1(cat)



class Context_AM(nn.Module):
    def __init__(self, channels= [128,256, 512, 1024]):
        super(Context_AM, self).__init__()

        self.ASPP = ASPP(256, 128)
        self.upsample = Upsample()
        self.c = Conv_Bn_Activation(channels[3], channels[2], 1, 1, activation='mish')
        self.c1 = Conv_Bn_Activation(channels[3], channels[1], 1, 1, activation='mish')
        self.c2 = Conv_Bn_Activation(channels[2], channels[0], 1, 1, activation='mish')
    def forward(self, input):
        #  input = [d5,d4,d3,d2]

        xt = self.c(input[0])
        x = self.upsample(xt, input[1].size())
        xt = torch.cat([x,input[1]],dim=1)
        xt = self.c1(xt)
        x = self.upsample(xt, input[2].size())
        xt = torch.cat([x, input[2]], dim=1)
        xt = self.c2(xt)
        x = self.upsample(xt, input[3].size())
        xt = torch.cat([x, input[3]], dim=1)

        del x

        return self.ASPP(xt)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features
# class ESEAttn(nn.Module):
#     def __init__(self, feat_channels):
#         super(ESEAttn, self).__init__()
#         self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
#         self.conv = Conv_Bn_Activation(feat_channels, feat_channels, 1)
#
#     def forward(self, feat, avg_feat):
#         weight = F.sigmoid(self.fc(avg_feat))
#         return self.conv(feat * weight)
class connect(nn.Module):
    def __init__(self,inputchannel):
        super(connect, self).__init__()
        self.channel = inputchannel
        self.conv = Conv_Bn_Activation(self.channel*2, self.channel, 1, 1, activation='mish')
    def forward(self,x, att):
        return self.conv(torch.cat([x,x*torch.sigmoid(att)],dim=1))