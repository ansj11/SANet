import torch
from torch import nn
from IPython import embed


# Spatial attention block
class SAB(nn.Module):
    def __init__(self, inp, oup, mode='spatial', fuse='mul'):
        super(SAB, self).__init__()
        if mode == 'channel':
            self.conv_mask = nn.Conv2d(inp+oup, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
            self.att = nn.Sequential(
                                     nn.Conv2d(inp+oup, oup, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.ReLU(inplace=True)
                                    )
        elif mode == 'spatial':
            self.att = nn.Sequential(nn.Conv2d(inp+oup, inp//2, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(inp//2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.ReLU(inplace=True)
                                    )
        else:
            self.att = nn.Sequential(nn.Conv2d(inp+oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.ReLU(inplace=True)
                                    )
        self.fuse = fuse
        self.mode = mode

    def forward(self, g, x):
        g = torch.cat([g, x], dim=1)
        if self.mode == 'channel':
            N, C, H, W = g.size()
            input = g
            input = input.view(N, C, H*W)
            input = input.unsqueeze(1)
            context_mask = self.conv_mask(g)
            context_mask = context_mask.view(N, 1, H*W)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input, context_mask)
            g = context.view(N, C, 1, 1)
        att = self.att(g)
        if self.fuse == 'mul':
            out = att * x
        else:
            out = x + att
        return out


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.conv_mask.bias, 0)
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            nn.init.constant_(self.channel_add_conv[-1].weight, 0)
            nn.init.constant_(self.channel_add_conv[-1].bias, 0)
        if self.channel_mul_conv is not None:
            nn.init.constant_(self.channel_mul_conv[-1].weight, 0)
            nn.init.constant_(self.channel_mul_conv[-1].bias, 0)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, C, 1, H * W]
            input_x = input_x.unsqueeze(2)
            # [N, C, H, W]
            context_mask = self.conv_mask(x)
            # [N, C, H * W]
            context_mask = context_mask.view(batch, channel, height * width)
            # [N, C, H * W]
            context_mask = self.softmax(context_mask)
            # [N, C, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, C, 1, 1]
            context = torch.matmul(input_x, context_mask)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
 
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
