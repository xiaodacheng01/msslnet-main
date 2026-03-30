"""

   时间: 2025/3/20 9:47
   内容:

                           """


import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile


# class RPReLU(nn.Module):
#     def __init__(self, num_parameters=1, init_alpha=1.0, init_beta=0.25):
#         super(RPReLU, self).__init__()
#         # 正区间缩放参数 alpha（默认恒等映射）
#         self.alpha = nn.Parameter(torch.full((num_parameters,), init_alpha))
#         # 负区间斜率参数 beta（类似PReLU）
#         self.beta = nn.Parameter(torch.full((num_parameters,), init_beta))

#     def forward(self, x):
#         print('x.shape',x.shape)
#         # 正区间：alpha * x，负区间：beta * x
#         pos = (x >= 0).to(x.dtype) * self.alpha * x
#         neg = (x < 0).to(x.dtype) * self.beta * x
#         return pos + neg


# rprelu = RPReLU(num_parameters=64)  # 输入特征图通道数为64

class SLCB(nn.Module):
    def __init__(self, dim, kernel1, k2, dilation1, dilation2, drop_out=0.1):
        super(SLCB, self).__init__()

        self.sparse_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel1, stride=1, padding=kernel1 // 2, groups=dim, dilation=1),
            nn.Conv2d(dim, dim, k2, stride=1, padding=k2 // 2, groups=dim, dilation=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(drop_out)
        )

        self.dense_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, dilation=1),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.25)
        )

        self.out_pwconv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.se = GMS(dim)

        # self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        sparse_branch = self.sparse_conv(x) ** 2

        dense_branch = self.dense_conv(x)

        # weights = F.softmax(self.w, dim=0)
        # out = sparse_branch * weights[0] + dense_branch * weights[1]
        out = self.out_pwconv(torch.cat([sparse_branch, dense_branch], dim=1))

        out = self.se(out)
        return out

class SLCB1(nn.Module):
    def __init__(self, dim, kernel1, k2, dilation1, dilation2, drop_out=0.1):
        super(SLCB1, self).__init__()

        self.sparse_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel1, stride=1, padding=kernel1 // 2, groups=dim, dilation=1),
            nn.Conv2d(dim, dim, k2, stride=1, padding=k2 // 2, groups=dim, dilation=1),
            nn.ReLU(),
            nn.Dropout(drop_out)
        )
        self.norm = nn.GroupNorm(1, dim)

        self.dense_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, dilation=1),
            nn.PReLU(num_parameters=dim, init=0.25),
            nn.GroupNorm(1, dim),
        )

        self.out_pwconv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GroupNorm(1, dim),
            nn.ReLU()
        )

        self.gms = GMS(dim)

        # self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        sparse_branch = self.sparse_conv(x) ** 2
        sparse_branch = self.norm(sparse_branch)

        dense_branch = self.dense_conv(x)

        # weights = F.softmax(self.w, dim=0)

        # out = sparse_branch * weights[0] + dense_branch * weights[1]
        out = self.out_pwconv(torch.cat([sparse_branch, dense_branch], dim=1))

        out = self.gms(out)
        return out

class GMS(nn.Module):
    def __init__(self, dim, height=2, reduction=4):
        super(GMS, self).__init__()

        self.height = height
        d = max(dim // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x))
        attn = self.softmax(attn)
        return x * attn


class MSSLB(nn.Module):
    def __init__(self, dim):
        super(MSSLB, self).__init__()

        self.splited_dim = dim // 4

        self.branch13 = nn.Sequential(
            SLCB(self.splited_dim, kernel1=13, k2=5, dilation1=1, dilation2=1),
        )

        self.branch11 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            SLCB(self.splited_dim, kernel1=11, k2=5, dilation1=1, dilation2=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.branch9 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            SLCB(self.splited_dim, kernel1=9, k2=5, dilation1=1, dilation2=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.branch7 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            SLCB(self.splited_dim, kernel1=7, k2=5, dilation1=1, dilation2=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, x):
        b13, b11, b9, b7 = torch.split(x, [self.splited_dim, self.splited_dim, self.splited_dim, self.splited_dim], 1)

        # b13 = self.branch13(b13) * self.conv13(b13)
        # b11 = self.branch11(b11) * self.conv11(b11)
        # b9 = self.branch9(b9) * self.conv9(b9)
        # b7 = self.branch7(b7) * self.conv7(b7)
        b13 = self.branch13(b13)
        b11 = self.branch11(b11)
        b9 = self.branch9(b9)
        b7 = self.branch7(b7)

        out = torch.cat([b13, b11, b9, b7], dim=1)
        out = self.out_conv(out)

        return out


class CGM(nn.Module):
    def __init__(self, dim, global_branch_kernel, m2l_ks):
        super(CGM, self).__init__()

        # self.m2l_ks = global_branch_kernel//2+1
        # self.gms = GMS(dim)

        self.lk_branch = nn.Sequential(
            # down
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, groups=dim,
                      dilation=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            LKBranch(dim, global_branch_kernel=global_branch_kernel),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.MSSLB_branch = MSSLB(dim)

        self.sa = SpatialAttention()

        self.m2l1 = nn.Sequential(
            nn.Conv2d(dim, dim, m2l_ks, 1, m2l_ks // 2 * 2, groups=dim, dilation=2),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.25),
        )
        self.m2l2 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lk_branch = self.lk_branch(x)
        MSSLB_branch = self.MSSLB_branch(x)
        MSSLB_branch_sa = self.sa(MSSLB_branch)

        intersect1 = MSSLB_branch_sa * lk_branch

        intersect2 = self.m2l1(intersect1)
        intersect2 = self.m2l2(torch.cat([intersect2, intersect1], dim=1))

        return x + intersect2 * MSSLB_branch


class LKBranch(nn.Module):
    def __init__(self, dim, global_branch_kernel, drop_path=0.1):
        super(LKBranch, self).__init__()

        self.gms = GMS(dim)

        self.lk = nn.Sequential(
            nn.Conv2d(dim, dim, global_branch_kernel, stride=1, padding=global_branch_kernel // 2, groups=dim,
                      dilation=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, global_branch_kernel, stride=1, padding=global_branch_kernel // 2, groups=dim,
                      dilation=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Dropout(drop_path),
        )

    def forward(self, x):
        x = self.lk(x)
        x = self.gms(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x


class GatedPFN(nn.Module):
    def __init__(self, dim, out_dim):
        super(GatedPFN, self).__init__()

        self.dim_conv = dim
        self.out_dim = out_dim
        self.splited_dim = dim // 4

        self.out_act = nn.Sequential(
            nn.Conv2d(dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            # nn.PReLU(num_parameters=out_dim, init=0.25),
            nn.ReLU()
        )

    def forward(self, x):

        out = self.out_act(x)

        return out


class BGF(nn.Module):
    def __init__(self, dim, bottleneck_dim, out_dim):
        super(BGF, self).__init__()

        self.dim = dim
        self.bottleneck_dim = bottleneck_dim

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, self.bottleneck_dim, 1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(True),

            nn.Conv2d(bottleneck_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

        self.dwdconv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2,
                      dilation=1),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 4, kernel_size=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU()
        )
        self.dwdconv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=2, groups=dim // 2,
                      dilation=2),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 4, kernel_size=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU()
        )
        self.dwdconv3 = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=3, groups=dim // 2,
                      dilation=3),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 4, kernel_size=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU()
        )
        self.dwdconv4 = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=4, groups=dim // 2,
                      dilation=4),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 4, kernel_size=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU()
        )
        self.fu_pwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            # nn.PReLU(num_parameters=out_dim, init=0.25),
            nn.ReLU()
        )
        self.x_l_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, dilation=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            # nn.PReLU(num_parameters=out_dim, init=0.25),
            nn.PReLU(num_parameters=dim, init=0.25),
        )

        # self.rprelu = RPReLU(num_parameters=dim)
        # self.prelu = nn.PReLU(num_parameters=dim, init=0.25)
        self.w = nn.Parameter(torch.ones(2))

        self.out_pwconv = nn.Sequential(
            nn.Conv2d(2 * dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            # nn.PReLU(num_parameters=out_dim, init=0.25),
            nn.ReLU()
        )

    def forward(self, x_l, x_h):
        x_h_ = self.topdown(x_h)
        x_l = x_l * x_h_

        x_l4, x_l3, x_l2, x_l1 = torch.split(x_l, [self.dim // 4, self.dim // 4, self.dim // 4, self.dim // 4], 1)
        x_h4, x_h3, x_h2, x_h1 = torch.split(x_h, [self.dim // 4, self.dim // 4, self.dim // 4, self.dim // 4], 1)

        fusion1 = self.dwdconv1(torch.cat([x_l1, x_h1], dim=1))
        fusion2 = self.dwdconv2(torch.cat([x_l2, x_h2], dim=1))
        fusion3 = self.dwdconv3(torch.cat([x_l3, x_h3], dim=1))
        fusion4 = self.dwdconv4(torch.cat([x_l4, x_h4], dim=1))

        fusion = torch.cat([fusion4, fusion3, fusion2, fusion1], dim=1)
        fusion = self.fu_pwconv(fusion)

        senti = fusion * x_h

        weights = F.softmax(self.w, dim=0)

        senti = senti * weights[0] - self.x_l_conv(x_l) * weights[1]

        out = self.out_pwconv(torch.cat([senti, fusion], dim=1))

        return out


class Stem(nn.Module):

    def __init__(self, dim, mid_dim, out_dim):
        super(Stem, self).__init__()

        # self.rprelu = RPReLU(out_dim)
        self.stem_convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=mid_dim, kernel_size=1),
            nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=2, padding=1, groups=mid_dim,
                      dilation=1),
            nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            # nn.PReLU(num_parameters=out_dim, init=0.25),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.stem_convs(x)
        return x


class Head(nn.Module):

    def __init__(self, dim, mid_dim, final_dim):
        super(Head, self).__init__()

        # self.rprelu = RPReLU(mid_dim)
        # self.prelu =
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=mid_dim, kernel_size=1),
            nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, groups=mid_dim,
                      dilation=1),
            nn.BatchNorm2d(mid_dim),
            # self.rprelu(),
            # nn.PReLU(num_parameters=mid_dim, init=0.25),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=mid_dim, out_channels=final_dim, kernel_size=1),
        )


    def forward(self, x):
        # print('head x.shape', x.shape)
        x = _upsample_like(x)
        # print('after head x.shape', x.shape)
        return self.head(x)


def _upsample_like(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear')


class MSSLNet(nn.Module):
    def __init__(self, init_dim):
        super(MSSLNet, self).__init__()

        self.stem = Stem(3, mid_dim=init_dim * 2, out_dim=init_dim)

        self.encode1 = nn.Sequential(
            CGM(init_dim, global_branch_kernel=13, m2l_ks=7),
            GatedPFN(init_dim, init_dim * 2)
        )
        self.encode2 = nn.Sequential(
            CGM(init_dim * 2, global_branch_kernel=11, m2l_ks=5),
            GatedPFN(init_dim * 2, (init_dim * 2) * 2)
        )
        self.encode3 = nn.Sequential(
            CGM(init_dim * 4, global_branch_kernel=9, m2l_ks=5),
            GatedPFN(init_dim * 4, (init_dim * 4))
        )
        self.encode4 = nn.Sequential(
            CGM(init_dim * 4, global_branch_kernel=7, m2l_ks=3),
            GatedPFN(init_dim * 4, init_dim * 4)
        )

        self.pool_layer1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool_layer2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # self.pool_layer3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.decode3 = nn.Sequential(
            CGM(init_dim * 4, global_branch_kernel=9, m2l_ks=5),
            GatedPFN(init_dim * 4, (init_dim * 8) // 2)
        )
        self.decode2 = nn.Sequential(
            CGM(init_dim * 4, global_branch_kernel=11, m2l_ks=5),
            GatedPFN(init_dim * 4, (init_dim * 4) // 2)
        )
        self.decode1 = nn.Sequential(
            CGM(init_dim * 2, global_branch_kernel=13, m2l_ks=7),
            GatedPFN(init_dim * 2, (init_dim * 2) // 2)
        )

        self.fusion1 = BGF(dim=init_dim * 2, bottleneck_dim=(init_dim * 2) // 4, out_dim=init_dim * 2)
        self.fusion2 = BGF(dim=init_dim * 4, bottleneck_dim=(init_dim * 4) // 4, out_dim=init_dim * 4)
        self.fusion3 = BGF(dim=init_dim * 4, bottleneck_dim=(init_dim * 4) // 4, out_dim=init_dim * 4)

        self.head = Head(init_dim, mid_dim=init_dim // 2, final_dim=1)

    def forward(self, x):
        x_stem = self.stem(x)

        x1 = self.encode1(x_stem)
        x_temp = self.pool_layer1(x1)

        x2 = self.encode2(x_temp)
        x_temp = self.pool_layer2(x2)

        x3 = self.encode3(x_temp)
        x4 = self.encode4(x_temp)

        x4_up = x4
        fusion_43 = self.fusion3(x3, x4_up)
        out3 = self.decode3(fusion_43)

        out3_up = _upsample_like(out3)
        fusion_32 = self.fusion2(x2, out3_up)
        out2 = self.decode2(fusion_32)

        out2_up = _upsample_like(out2)
        fusion_21 = self.fusion1(x1, out2_up)
        out1 = self.decode1(fusion_21)

        out = self.head(out1)

        # return F.sigmoid(out)
        return out


if __name__ == '__main__':
    model = MSSLNet(16)
    input = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9} GFLOPs")
    print(params)
