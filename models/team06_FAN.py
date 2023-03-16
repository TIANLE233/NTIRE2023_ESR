import math

import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    r"""

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))

        rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)


class MLP(nn.Module):
    def __init__(self, n_feats, act_layer: nn.Module) -> None:
        super().__init__()
        i_feats = n_feats * 2
        self.fc1 = Conv2d1x1(n_feats, i_feats, bias=True)
        self.act = act_layer()
        self.fc2 = Conv2d1x1(i_feats, n_feats, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FocalModulation(nn.Module):
    r"""Focal Modulation.

    Modified from https://github.com/microsoft/FocalNet.

    Args:
        dim (int): Number of input channels.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int): Step to increase the focal window
        act_layer (nn.Module):

    """

    def __init__(self, dim: int, act_layer: nn.Module, focal_level: int = 4,
                 focal_window: int = 3, focal_factor: int = 2) -> None:
        super().__init__()

        self.dim = dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.act = act_layer()

        self.f = Conv2d1x1(dim, 2 * dim)

        self.focal_layers = nn.ModuleList()
        for i in range(self.focal_level):
            kernel_size = self.focal_factor * i + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    self.act))

        self.h = Conv2d1x1(dim, dim)
        self.proj = Conv2d1x1(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        q, ctx = torch.split(x, [self.dim, self.dim], 1)

        ctx_all = 0
        for i in range(self.focal_level):
            ctx = self.focal_layers[i](ctx)
            ctx_all = ctx_all + ctx
        x_out = q * self.sigmoid(self.h(ctx_all))
        return self.proj(x_out)


class TransformerGroup(nn.Module):
    r"""

    Args:
        sa_list:
        mlp_list:

    """

    def __init__(self, sa_list: list, mlp_list: list, dim: int) -> None:
        super(TransformerGroup, self).__init__()

        assert len(sa_list) == len(mlp_list)

        self.sa_list = nn.ModuleList(sa_list)
        self.mlp_list = nn.ModuleList(mlp_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)
        return x


class FocalTG(TransformerGroup):
    def __init__(self, n_t: int, dim: int, act_layer: nn.Module) -> None:
        sa_list = [FocalModulation(dim=dim, focal_level=4, focal_window=3, focal_factor=2, act_layer=act_layer)
                   for _ in range(n_t)]

        mlp_list = [MLP(dim, act_layer=act_layer)
                    for _ in range(n_t)]

        super(FocalTG, self). \
            __init__(sa_list=sa_list, mlp_list=mlp_list, dim=dim)


class FAN(nn.Module):
    def __init__(self, upscale: int = 4, num_in_ch: int = 3, num_out_ch: int = 3, task: str = 'lsr',
                 n_t: int = 12, n_g: int = 1, dim: int = 32, act_layer: nn.Module = nn.GELU) -> None:
        super(FAN, self).__init__()

        self.sub_mean = MeanShift(255, sign=-1, data_type='DF2K')
        self.add_mean = MeanShift(255, sign=1, data_type='DF2K')

        self.head = Conv2d3x3(num_in_ch, dim)

        modules_body = [FocalTG(n_t=n_t, dim=dim, act_layer=act_layer)
                        for _ in range(n_g)]
        self.body = nn.Sequential(*modules_body)

        self.tail = Upsampler(upscale=upscale, in_channels=dim,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x
