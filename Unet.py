import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv   # 假定 common.conv 已定义并行为正确


class UNetConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, need_norm_layer=True, need_bias=True, pad='zero'):
        super(UNetConv2d, self).__init__()
        if need_norm_layer:
            self.block = nn.Sequential(
                conv(in_ch, out_ch, kernel_size, bias=need_bias, pad=pad),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                conv(out_ch, out_ch, kernel_size, bias=need_bias, pad=pad),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                conv(in_ch, out_ch, kernel_size, bias=need_bias, pad=pad),
                nn.LeakyReLU(0.2, inplace=True),
                conv(out_ch, out_ch, kernel_size, bias=need_bias, pad=pad),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        return self.block(x)



# 上采样块：上采样

class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, upsample_mode='nearest', need_norm_layer=True, need_bias=True, pad='zero'):
        super(UnetUp, self).__init__()
        assert upsample_mode in ['nearest', 'bilinear']
        self.upsample_mode = upsample_mode
        self.conv = UNetConv2d(in_ch, out_ch, kernel_size, need_norm_layer, need_bias, pad)
    '''
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode,
                          align_corners=(self.upsample_mode == 'bilinear'))
        if skip is not None:
            # 若 skip 与 x 空间尺寸不一致（可能由奇数下采样导致），可以在此处裁剪 skip：
            if skip.shape[-2:] != x.shape[-2:]:
                # center-crop skip to match x
                sh, sw = skip.shape[-2:]
                th, tw = x.shape[-2:]
                dh = (sh - th) // 2
                dw = (sw - tw) // 2
                skip = skip[..., dh:dh+th, dw:dw+tw]
            x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
    '''

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],  # 🔴 关键：直接对齐 skip
                mode=self.upsample_mode,
                align_corners=(self.upsample_mode == 'bilinear')
            )
            x = torch.cat([skip, x], dim=1)
        else:
            x = F.interpolate(
                x, scale_factor=2,
                mode=self.upsample_mode,
                align_corners=(self.upsample_mode == 'bilinear')
            )
        x = self.conv(x)

        return x




class UNet(nn.Module):
    def __init__(self,
                 num_input_channels=3,
                 num_output_channels=1,
                 num_channels_down=[16, 32, 64, 128, 128],
                 num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4 ,4],
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_skip_size=1,
                 pad='zero',
                 upsample_mode='nearest',
                 need_norm_layers=True,
                 need_bias=True,
                 need_sigmoid=True):
        super(UNet, self).__init__()

        assert len(num_channels_down) == len(num_channels_up), "down 和 up 列表长度必须相同 (depth)"
        self.depth = len(num_channels_down)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample_mode = upsample_mode


        if num_channels_skip is None:
            num_channels_skip = [0] * (self.depth - 1)

        if len(num_channels_skip) >= self.depth:
            num_channels_skip = list(num_channels_skip[:self.depth - 1])

        assert len(num_channels_skip) == self.depth - 1, "num_channels_skip 长度应为 depth - 1"


        self.enc_convs = nn.ModuleList()
        for i in range(self.depth):
            in_ch = num_input_channels if i == 0 else num_channels_down[i - 1]
            out_ch = num_channels_down[i]
            self.enc_convs.append(
                UNetConv2d(in_ch, out_ch, kernel_size=filter_size_down,
                           need_norm_layer=need_norm_layers, need_bias=need_bias, pad=pad)
            )


        self.skip_convs = nn.ModuleList()
        for i in range(self.depth - 1):
            ch_in = num_channels_down[i]
            ch_skip = num_channels_skip[i]
            if ch_skip > 0:
                self.skip_convs.append(
                    nn.Sequential(
                        conv(ch_in, ch_skip, kernel_size=filter_skip_size, bias=need_bias, pad=pad),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            else:
                self.skip_convs.append(nn.Identity())


        self.ups = nn.ModuleList()
        prev_ch = num_channels_down[-1]  # bottleneck channels
        for i in range(self.depth - 2, -1, -1):
            skip_ch = num_channels_skip[i]
            in_ch = prev_ch + skip_ch
            out_ch = num_channels_up[i]
            self.ups.append(
                UnetUp(in_ch, out_ch, kernel_size=filter_size_up, upsample_mode=upsample_mode,
                       need_norm_layer=need_norm_layers, need_bias=need_bias, pad=pad)
            )
            prev_ch = out_ch

        # output head
        self.out_conv = nn.Conv2d(prev_ch, num_output_channels, kernel_size=1, bias=True)
        self.out_act = nn.Sigmoid() if need_sigmoid else nn.Identity()

    def forward(self, x):
        skips = []

        for i, enc in enumerate(self.enc_convs):
            x = enc(x)
            if i < self.depth - 1:

                skip_feat = self.skip_convs[i](x)
                #print(skip_feat.shape)
                skips.append(skip_feat)
                x = self.pool(x)
            else:
                # 最底层（bottleneck）
                bottleneck = x

        x = bottleneck
        #print(x.shape)
        for j, up in enumerate(self.ups):
            skip_i = self.depth - 2 - j   # depth-2 ... 0
            skip_feat = skips[skip_i]
            #print(skip_feat.shape)
            x = up(x, skip_feat)
            #print(x.shape)

        y = self.out_conv(x)
        y = self.out_act(y)
        return y



if __name__ == "__main__":
    x = torch.randn(1, 2, 256, 256)
    net = UNet(num_input_channels=2,
                num_output_channels=1,
                num_channels_down=[16, 32, 64],
                num_channels_up=[16, 32, 64],
                num_channels_skip=[4, 4],
                upsample_mode='bilinear',
                need_sigmoid=True)
    y = net(x)
    print("Output:", y.shape)
