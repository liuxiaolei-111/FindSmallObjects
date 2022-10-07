import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F
import sys
from mmcv.cnn import constant_init, kaiming_init, xavier_init

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size


        return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation = 'mish', bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03))
            # self.conv.append(nn.GroupNorm(int(out_channels//16) if int(out_channels//16)>0 else 1,out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="mish",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # Conv = Conv_Bn_Activation()
        #--------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        #--------------------------------------------------#
        self.conv1 = Conv_Bn_Activation(in_channels, hidden_channels, 1, stride=1, activation=act)
        #--------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        #--------------------------------------------------#
        self.conv2 = Conv_Bn_Activation(hidden_channels, out_channels, 3, stride=1, activation=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=1, act="mish"):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        #--------------------------------------------------#
        #   主干部分的初次卷积
        #--------------------------------------------------#
        self.conv1  = Conv_Bn_Activation(in_channels, hidden_channels, 1, stride=1, activation=act)
        #--------------------------------------------------#
        #   大的残差边部分的初次卷积
        #--------------------------------------------------#
        self.conv2  = Conv_Bn_Activation(in_channels, hidden_channels, 1, stride=1, activation=act)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        self.conv3  = Conv_Bn_Activation(2 * hidden_channels, out_channels, 1, stride=1, activation=act)

        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        #--------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0,  act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)
        # self.cbam = CBAM(out_channels)
    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)
'''
写在开头：
卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存。
'''

#Mish激活函数，自定义
class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#---------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        # self.bn = nn.GroupNorm(out_channels//16,out_channels)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

        self.block2 = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )
    def forward(self, x):
        return x + self.block(x) + self.block2(x)


class Bottleneck1(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        hidden_channels = hidden_channels

        self.conv1 = BasicConv(in_channels, hidden_channels, 1, stride=1 )
        self.conv2 = BasicConv(in_channels, hidden_channels, 1, stride=1)
        self.conv3 = BasicConv(in_channels, hidden_channels, 1, stride=1)
        self.conv4 = BasicConv(hidden_channels, hidden_channels//2, 1, stride=1)
        self.conv5 = BasicConv(hidden_channels, hidden_channels//2, 1, stride=1)
        self.conv6 = nn.Sequential(
            BasicConv(hidden_channels // 2, hidden_channels // 2, 3, stride=1),
            BasicConv(hidden_channels // 2, hidden_channels // 2, 3, stride=1)
        )
        self.conv7 = nn.Sequential(
            BasicConv(hidden_channels // 2, hidden_channels // 2, 3, stride=1),
            BasicConv(hidden_channels // 2, hidden_channels // 2, 3, stride=1)
        )
        self.conv8 = BasicConv(in_channels*4, in_channels, 1, stride=1)


    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(y2)
        y5 = self.conv5(y3)
        y4 = self.conv6(y4)
        y5 = self.conv7(y5)
        y = self.conv8(torch.cat([y1, y2, y3, y4, y5],dim=1))
        return y


class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:

            self.split_conv0 = BasicConv(out_channels, out_channels, 1)


            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:

            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2,out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = nn.Sequential(
            BasicConv(out_channels, out_channels, 1))


    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x

class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)



    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        out2 = self.stages[1](x)
        out2 = self.cbam1(out2)
        out3 = self.stages[2](out2)
        out3 = self.cbam2(out3)
        out4 = self.stages[3](out3)
        out4 = self.cbam3(out4)
        out5 = self.stages[4](out4)
        out5 = self.cbam4(out5)

        return [out5, out4, out3, out2]

class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 3, 6, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out



class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 平铺
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # shared MLP 多层感知机
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # avg_pool size: [bs, gate_channels, 1, 1]
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
            	# avg_pool + max_pool
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    # [bs, channel, w, h] to [bs, channel, w*h]
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
    	# 沿着原feature的channel分别做max_pool和avg_pool，然后将两者concat
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
