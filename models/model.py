from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 with_bn=True,
                 with_act=True
                 ):
        super(ConvBlock, self).__init__()
        bias = not with_bn
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
        self.with_act = with_act
        if with_act:
            self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels//2, kernel_size=1)
        self.conv2 = ConvBlock(in_channels//2, out_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            out = out + x
        return out


class YOLOBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOBlock, self).__init__()
        double_out_channels = out_channels * 2
        self.conv1 = ConvBlock(in_channels, out_channels, 1, 1)
        self.conv2 = ConvBlock(out_channels, double_out_channels, 3, padding=1)
        self.conv3 = ConvBlock(double_out_channels, out_channels, 1)
        self.conv4 = ConvBlock(out_channels, double_out_channels, 3, padding=1)
        self.conv5 = ConvBlock(double_out_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.stage0 = ConvBlock(3, 32, 3, 1)
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, 3, 2, 1),
            BasicBlock(64, 64)    
        ) # 1/2
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, 3, 2, 1),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        ) # 1/4
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, 3, 2, 1),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        ) # 1/8
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, 3, 2, 1),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        ) # 1/16
        self.stage5 = nn.Sequential(
            ConvBlock(512, 1024, 3, 2, 1),
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024)
        ) # 1/32

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        out3 = self.stage3(x)
        out4 = self.stage4(out3)
        out5 = self.stage5(out4)
        return out3, out4, out5


class YOLONeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLONeck, self).__init__()
        assert len(in_channels) == len(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = len(self.in_channels)

        for i in range(self.num_scales):
            yolo__block_in_channels = in_channels[i] + out_channels[i]
            if i == self.num_scales - 1:
                yolo__block_in_channels = in_channels[i]
            self.add_module(f'yolo_block{i}', YOLOBlock(yolo__block_in_channels, out_channels[i]))

            if i != 0:
                self.add_module(f'conv{i}', ConvBlock(out_channels[i], out_channels[i-1], 1))
                self.add_module(f'upscale{i}', nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        outs = []
        out = getattr(self, f'yolo_block{self.num_scales - 1}')(feats[-1]) # highest level
        outs.append(out)

        for i in range(1, self.num_scales):
            # high-level to low-level, i.e. 1, 0
            level_id = self.num_scales - 1 - i
            x = feats[level_id]

            bottom = getattr(self, f'conv{level_id + 1}')(out)
            bottom = getattr(self, f'upscale{level_id + 1}')(bottom)
            x = torch.cat((bottom, x), 1)

            out = getattr(self, f'yolo_block{level_id}')(x)
            outs.insert(0, out)

        return outs


class YOLOHead(nn.Module):
    def __init__(self, in_channels, stride, anchors, num_classes):
        super(YOLOHead, self).__init__()
        self.stride = stride
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_per_anchor = 5 + num_classes

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2) # (n_anchors, 2)
        anchors_grid = anchors.clone().view(1, -1, 1, 1, 2) # (1, num_anchors, 1, 1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors_grid)
        self.grid = None

        mid_channels = 2 * in_channels
        out_channels = self.num_anchors * self.num_per_anchor
        self.conv = ConvBlock(in_channels, mid_channels, 3, padding=1)
        self.head = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (b, A*(4 + 1 + c), h, w)
        """
        x = self.head(self.conv(x))
        print(x.shape)
        # (B, C, H, W) => (B, n_anchors, H, W, n_classes + 5)
        B, _, H, W = x.shape
        x = x.view(B, self.num_anchors, self.num_per_anchor, H, W).permute(0, 1, 3, 4, 2).contiguous()

        # decode
        if self.grid is None:
            self.grid = self.make_grid(W, H).to(x.device) # (1, 1, H, W, 2)

        x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * self.stride
        x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid
        x[..., 4:] = x[..., 4:].sigmoid() # (B, n_anchors, H, W, num_per_anchor)

        x = x.view(B, -1, self.num_per_anchor) # (B, b_boxes, num_per_anchor)
        return x

    @staticmethod
    def make_grid(w, h):
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, h, w, 2)).float()


class YOLO(nn.Module):
    def __init__(self, anchors, num_classes):
        super(YOLO, self).__init__()
        self.backbone = Darknet53()
        self.neck = YOLONeck(
            in_channels=(256, 512, 1024),
            out_channels=(128, 256, 512)
        )
        self.head0 = YOLOHead(128, 8, anchors[0], num_classes)
        self.head1 = YOLOHead(256, 16, anchors[1], num_classes)
        self.head2 = YOLOHead(512, 32, anchors[2], num_classes)

    def forward(self, x):
        outs = []
        feats = self.backbone(x)
        feats = self.neck(feats)
        outs.append(self.head0(feats[0]))
        outs.append(self.head1(feats[1]))
        outs.append(self.head2(feats[2]))
        return outs


def build_model(anchors, num_classes, device, checkpoint=None):
    model = YOLO(anchors, num_classes)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    model = model.to(device)

    return model


if __name__ == "__main__":
    # x = torch.randn((1, 3, 416, 416))
    # model = Darknet53()
    # out3, out4, out5 = model(x)
    # print(out3.shape)
    # print(out4.shape)
    # print(out5.shape)

    # x = torch.randn((1, 1024, 8, 8))
    # model = YOLOBlock(1024, 512)
    # print(model(x).shape)

    # feats = [torch.randn(1, 1024, 8, 8),
    #          torch.randn(1, 512, 16, 16),
    #          torch.randn(1, 256, 32, 32)]
    # feats = list(reversed(feats))

    # model = YOLONeck([256, 512, 1024], [128, 256, 512])

    # outs = model(feats)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    # anchors = [(10, 13), (16, 30), (33, 23)]
    # model = YOLOHead(32, anchors, 80)
    # x = torch.randn((1, 255, 8, 8))
    # print(model(x).shape)

    anchors = [[(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]]
    num_classes = 20
    model = YOLO(anchors, num_classes)
    x = torch.randn((1, 3, 416, 416))
    outs = model(x)
    print(outs[0].shape)
    print(outs[1].shape)
    print(outs[2].shape)