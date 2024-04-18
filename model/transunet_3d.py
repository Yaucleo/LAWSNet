import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.ndimage import morphology
from vit_3d import ViT
from scipy.ndimage import sobel

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm3d(width)

        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm3d(width)

        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x

# 编码部分
class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        # CNN
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        # self.vit_img_dim = img_dim // patch_dim
        self.vit_img_dim = [int(x // patch_dim) for x in img_dim]
        # print(111)
        # print(self.vit_img_dim)
        # self.vit_img_dim = img_dim // patch_dim
        # self.vit_img_dim = (16, 16, 16)

        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.conv2 = nn.Conv3d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        x5 = self.vit(x4)
        x6 = rearrange(x5, 'b (x y z) d -> b d x y z',
                                           x=self.vit_img_dim[0], y=self.vit_img_dim[1], z=self.vit_img_dim[2])
        # x6 = rearrange(x5, "b (x y z) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x7 = self.conv2(x6)
        x8 = self.norm2(x7)
        x9 = self.relu(x8)

        return x9, x1, x2, x3

# 解码
class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv3d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x

class DC(nn.Module):
    def __init__(self, class_num, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(class_num, out_channels, kernel_size=1)


    def forward(self, input):
        input = nn.Sigmoid(input)
        input[input > 0.5] = 1
        input[input < 1] = 0
        outputs = input.cpu().detach().numpy()
        outputs = outputs.squeeze(0)
        outputs = outputs.squeeze(0)
        edges_x = sobel(outputs, axis=0)
        edges_y = sobel(outputs, axis=1)
        edges_z = sobel(outputs, axis=2)
        total_edges = np.sqrt(edges_x ** 2 + edges_y ** 2 + edges_z ** 2)
        total_edges[total_edges != 0] = 1
        structure_7 = np.ones((7, 7, 7), dtype=bool)
        result_area = morphology.binary_dilation(total_edges, structure=structure_7)
        total_edges = torch.Tensor(result_area).unsqueeze(0)
        # 维度转换
        res = self.conv(total_edges)
        return res



class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder1 = Decoder(out_channels, class_num)

        self.decoder2 = Decoder(out_channels, class_num)

        self.DC = DC(class_num, out_channels * 2)


    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x_endo = self.decoder1(x, x1, x2, x3)
        endo = x_endo.clone().detach()
        endo_wall = self.DC(endo, x)
        x = x * endo_wall
        x_wall = self.decoder2(x, x1, x2, x3)

        return x_endo, x_wall


if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=(80, 128, 128),
                          in_channels=16,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)
    res = transunet(torch.randn(1, 16, 144, 144, 20))
    print(res)

    # print(transunet(torch.randn(1, 1, 80, 128, 128)))
