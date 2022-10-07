import numpy as np
import cv2
from torch import nn
import torch
from nets.model import CSPLayer, Conv_Bn_Activation ,Upsample,  ASPP

from .cam import CAMmy, CAM, Context_AM, connect
from .cam import SpatialPyramidPooling as spp
from nets.model import CSPDarkNet


class Head(nn.Module):
    def __init__(self,classes):
        super().__init__()
        # self.num_classes = cfg.NUM_CLASSES
        ###########
        # my head
        in_channels = [128, 256, 512]
        ###########
        # Contrasted head
        # in_channels = [256, 512,1024]
        ############
        self.upsample = Upsample()
        self.classes = classes
        # self.stems = nn.ModuleList()
        self.stems_cls = nn.ModuleList()
        self.stems_reg = nn.ModuleList()
        self.cls_conv = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.bbox_conv = nn.ModuleList()
        self.bbox_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()


        for i in range(len(in_channels)):


            self.stems_cls.append(CAMmy(in_channels[i]))
            self.stems_reg.append(CAMmy(in_channels[i]))
            self.cls_conv.append( Conv_Bn_Activation(in_channels[i] , in_channels[i], 1, 1, 'mish'))
            self.cls_preds.append(nn.Conv2d(in_channels[i], self.classes, 3, 1,padding = 1))
            self.bbox_conv.append( Conv_Bn_Activation(in_channels[i] , in_channels[i], 3, 1, 'mish'))
            self.bbox_preds.append( nn.Conv2d(in_channels[i] , 4, 1, 1))
            self.obj_preds.append(nn.Conv2d(in_channels = in_channels[i], out_channels = 1, kernel_size=1, stride = 1, padding = 0))
    def forward(self,features):

        output = []
        for l, feature in enumerate(features):
            # stems = self.stems[l](feature)

            cls_feat = self.cls_conv[l](self.stems_cls[l](feature))
            reg_feat = self.bbox_conv[l](self.stems_reg[l](feature))
            cls_output = self.cls_preds[l](cls_feat+feature)
            reg_output = self.bbox_preds[l](reg_feat)
            obj_output = self.obj_preds[l](reg_feat)

            out = torch.cat([reg_output, obj_output, cls_output], 1)
            output.append(out)
            del out
        return output


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = backbone()
        in_channels = [256, 512, 1024]
        #--------------

        self.cam_tex = Context_AM()
        # --------------

        self.SPP = spp()
        # -------------------------------------------#
        # 产生通道注意力
        # -------------------------------------------#
        self.cam1 = CAMmy(128)
        self.cam2 = CAMmy(256)
        self.cam3 = CAMmy(512)

        self.upsample = Upsample()

        # --------------------------------------------

        # --------------------------------------------
        self.aspp_d2 = Conv_Bn_Activation(512, 128, 1, 1, activation='mish')
        self.as_d2_cat = Conv_Bn_Activation(256, 128, 1, 1, activation='mish')
        self.aspp_d3 = Conv_Bn_Activation(512, 256, 3, 2, activation='mish')
        self.as_d3_cat = Conv_Bn_Activation(512, 256, 1, 1, activation='mish')
        self.aspp_d4 = Conv_Bn_Activation(512, 512, 3, 4, activation='mish')
        self.as_d4_cat = Conv_Bn_Activation(1024, 512, 1, 1, activation='mish')

        # --------------------------------------------

        # --------------------------------------------

        self.d5d4conv = Conv_Bn_Activation(4096, 512, 1, 1, activation='mish')

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        self.lateral_conv0 =Conv_Bn_Activation(1024, 512, 1, 1, activation='mish')

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[1] ),
            3,
            False)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = Conv_Bn_Activation(512, 256 , 1, 1, activation='mish')
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            2 * 256,
            256,
            3,
            False)

        self.reduce_conv0 = Conv_Bn_Activation(256, 128, 1, 1, activation='mish')
        # -------------------------------------------#
        #   160, 160, 256 -> 160, 160, 128
        # -------------------------------------------#
        self.C2_p2 = CSPLayer(
            256,
            128,
            3,
            False)

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv_Bn_Activation(128, 128, 3, 2, activation='mish')
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
           256,
           256 ,
            3,
            False)

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        self.bu_conv1 = Conv_Bn_Activation(256 , 256 , 3, 2, activation='mish')
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        self.C3_n4 = CSPLayer(
            2 * 256,
            512,
            3,
            False)

    def forward(self, input):

        # [d5, d4, d3, d2] = input
     #--------------------------------------------
        aspp = self.cam_tex(input)

        cam1 = self.aspp_d2(aspp)
        cam1 = self.cam1(input[3])*torch.sigmoid(cam1)
        cam1 = torch.cat([cam1,input[3]],dim=1)
        cam1 = self.as_d2_cat(cam1)
        # d3

        cam2 = self.aspp_d3(aspp)
        cam2 = self.cam2(input[2]) * torch.sigmoid(cam2)
        cam2 = torch.cat([ cam2, input[2]], dim=1)
        cam2 = self.as_d3_cat(cam2)

        # d4
        cam3 = self.aspp_d4(aspp)
        cam3 = self.cam3(input[1]) * torch.sigmoid(cam3)
        cam3 = torch.cat([cam3, input[1]], dim=1)
        cam3 = self.as_d4_cat(cam3)

        # c=512
        d5_d4 = self.d5d4conv(self.SPP(input[0]) )
        d5_d4 = self.upsample(d5_d4, input[1].size())

        P5_upsample1 = torch.cat([d5_d4, cam3],dim=1)
        P5_upsample = self.C3_p4(P5_upsample1)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        P4 = self.reduce_conv1(P5_upsample)
        # -------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        # -------------------------------------------#
        P4_upsample = self.upsample(P4,input[2].size())
        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        # -------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, cam2], 1)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        P3_upsample = self.C3_p3(P4_upsample)
        # -------------------------------------------#
        #   160, 160, 256 -> 160, 160, 128
        # -------------------------------------------#
        P3 = self.reduce_conv0(P3_upsample)
        P3_upsample = self.upsample(P3, input[3].size())
        # -------------------------------------------#
        #   160, 160, 128 + 80, 80, 256 -> 160, 160, 256
        # -------------------------------------------#
        P3_upsample =torch.cat([P3_upsample,cam1],dim=1)
        # -------------------------------------------#
        #   d2 out  -> 160, 160, 128
        # -------------------------------------------#
        P2_out = self.C2_p2(P3_upsample)

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        P3_downsample = self.bu_conv2(P2_out)
        # -------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        P3_downsample = torch.cat([P3_downsample, P3], 1)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        P4_out = self.C3_n3(P3_downsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        P4_downsample = self.bu_conv1(P4_out)
        # -------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        # -------------------------------------------#
        P4_downsample = torch.cat([P4_downsample, P4], 1)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        P5_out = self.C3_n4(P4_downsample)

        return (P2_out, P4_out, P5_out)


#my_backbone+EC-PAFPN+Decoupled Attention Head  （我的模型）
class Body(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.classes = classes
        self.backbone = CSPDarkNet([1, 2, 6, 6, 3])
        self.neck = Neck()
        self.head     = Head(self.classes)

    def forward(self, x):

        return self.head.forward(self.neck.forward(self.backbone.forward(x)))

