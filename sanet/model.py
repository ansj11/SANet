import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models.resnet import resnet101
from IPython import embed

from .blocks import ContextBlock, SAB


def conv(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=True),
        nn.ReLU(inplace=True))

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes*upscale_factor**2),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU(inplace=True)
    )

class SANet(nn.Module):
    def __init__(self, pretrained=True, output_channel=1, rate=1, fixed_feature_weights=False):
        super(SANet, self).__init__()
        self.output_channel = output_channel
       
        resnet = resnet101(pretrained=pretrained)
        
        # Freeze resnet weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False
        
        c = [i//rate for i in [64, 256, 512, 1024, 2048]]

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        
        # previous layers
        self.mid1 = conv_bn(c[0], c[0], 1)
        self.mid2 = conv_bn(c[1], c[1], 1)
        self.mid3 = conv_bn(c[2], c[2], 1)
        self.mid4 = conv_bn(c[3], c[3], 1)

        # Smooth layers
        self.smooth1 = conv_bn(c[4], c[3], 1)
        self.smooth2 = conv_bn(c[3], c[2], 1)
        self.smooth3 = conv_bn(c[2], c[1], 1)
        self.smooth4 = conv_bn(c[0]*2, c[0], 1)
        self.smooth5 = conv(c[0]*4, c[0]*2, 1)

        self.upconv1 = conv_bn(c[4], c[3], 1)
        self.upconv2 = conv_bn(c[3], c[2], 1)
        self.upconv3 = conv_bn(c[2], c[1], 1)
        self.upconv4 = conv_bn(c[1], c[0], 1)
        self.upconv5 = conv(c[0]*2, c[0], 1)

        self.att1 = SAB(c[3], c[3])
        self.att2 = SAB(c[2], c[2])
        self.att3 = SAB(c[1], c[1])
        self.att4 = SAB(c[0], c[0])

        # upshuffle layers
        self.up1 = upshuffle(c[3], c[0], 8)
        self.up2 = upshuffle(c[2], c[0], 4)
        self.up3 = upshuffle(c[1], c[0], 2)

        # Context block
        self.gc2 = ContextBlock(c[1], 1/4)
        self.gc3 = ContextBlock(c[2], 1/4)
        self.gc4 = ContextBlock(c[3], 1/4)
        self.gc5 = ContextBlock(c[4], 1/4)

        # Depth prediction
        self.predict = predict(c[0], self.output_channel)

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c2 = self.gc2(c2)
        c3 = self.layer2(c2)
        c3 = self.gc3(c3)
        c4 = self.layer3(c3)
        c4 = self.gc4(c4)
        c5 = self.layer4(c4)
        p5 = self.gc5(c5)
        
        # Top-down
        _, _, H, W = c4.size()
        p4 = F.upsample(p5, size=(H, W), mode='bilinear')
        p4  = self.upconv1(p4)
        b  = self.att1(p4, self.mid4(c4))
        p4 = torch.cat([p4, b], dim=1)
        p4 = self.smooth1(p4)
        
        _, _, H, W = c3.size()
        p3 = F.upsample(p4, size=(H, W), mode='bilinear')
        p3  = self.upconv2(p3)
        b  = self.att2(p3, self.mid3(c3))
        p3 = torch.cat([p3, b], dim=1)
        p3 = self.smooth2(p3)

        _, _, H, W = c2.size()
        p2 = F.upsample(p3, size=(H, W), mode='bilinear')
        p2  = self.upconv3(p2)
        b  = self.att3(p2, self.mid2(c2))
        p2 = torch.cat([p2, b], dim=1)
        p2 = self.smooth3(p2)

        _, _, H, W = c1.size()
        p1 = F.upsample(p2, size=(H, W), mode='bilinear')
        p1  = self.upconv4(p1)
        b  = self.att4(p1, self.mid1(c1))
        p1 = torch.cat([p1, b], dim=1)
        p1 = self.smooth4(p1)

        # concatenate all branch
        d4, d3, d2, d1 = self.up1(p4), self.up2(p3), self.up3(p2), p1
        d = torch.cat([F.upsample(p, size=(H, W), mode='bilinear') for p in [d1, d2, d3, d4]], dim=1)
        d = self.smooth5(d)

        _, _, H, W = x.size()
        p = F.upsample(d, size=(H, W), mode='bilinear')
        p = self.upconv5(p)
        return self.predict(p)


def load_state_dict(own_state,  new_state_dict):
    for name, param in new_state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                continue

