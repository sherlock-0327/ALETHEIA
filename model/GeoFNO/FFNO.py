"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import copy
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm


# class WNLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
#         super().__init__(in_features=in_features,
#                          out_features=out_features,
#                          bias=bias,
#                          device=device,
#                          dtype=dtype)
#         if wnorm:
#             weight_norm(self)
#
#         self._fix_weight_norm_deepcopy()
#
#     def _fix_weight_norm_deepcopy(self):
#         # Fix bug where deepcopy doesn't work with weightnorm.
#         # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
#         orig_deepcopy = getattr(self, '__deepcopy__', None)
#
#         def __deepcopy__(self, memo):
#             # save and delete all weightnorm weights on self
#             weights = {}
#             for hook in self._forward_pre_hooks.values():
#                 if isinstance(hook, WeightNorm):
#                     weights[hook.name] = getattr(self, hook.name)
#                     delattr(self, hook.name)
#             # remove this deepcopy method, restoring the object's original one if necessary
#             __deepcopy__ = self.__deepcopy__
#             if orig_deepcopy:
#                 self.__deepcopy__ = orig_deepcopy
#             else:
#                 del self.__deepcopy__
#             # actually do the copy
#             result = copy.deepcopy(self)
#             # restore weights and method on self
#             for name, value in weights.items():
#                 setattr(self, name, value)
#             self.__deepcopy__ = __deepcopy__
#             return result
#         # bind __deepcopy__ to the weightnorm'd layer
#         self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)

class WNLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        self.wnorm = wnorm
        if wnorm:
            weight_norm(self)

    def __deepcopy__(self, memo):
        # Create a new instance without triggering weight_norm twice
        new_layer = WNLinear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
            wnorm=False  # Temporarily skip to avoid duplication
        )

        # Manually copy parameters and buffers
        for attr in ['weight', 'bias']:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(new_layer, attr, copy.deepcopy(val, memo))

        # Copy other attributes
        new_layer.training = self.training

        # Reapply weight_norm if used
        if self.wnorm:
            weight_norm(new_layer)

        return new_layer

class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, S1, S2, S3 = x.shape

        # # # Dimesion Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))

        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + xz

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class FNOFactorizedMesh3D(nn.Module):
    def __init__(self, modes_x, modes_y, modes_z, width, input_dim, output_dim,
                 n_layers, share_weight, factor, ff_weight_norm, n_ff_layers,
                 layer_norm):
        super().__init__()
        self.padding = 8  # pad the domain if input is non-periodic
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = WNLinear(input_dim + 3, self.width, wnorm=ff_weight_norm)
        self.n_layers = n_layers

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       modes_x=modes_x,
                                                       modes_y=modes_y,
                                                       modes_z=modes_z,
                                                       forecast_ff=None,
                                                       backcast_ff=None,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       use_fork=False,
                                                       dropout=0.0))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))

    def forward(self, x, pos):
        x = x.reshape(1, 20, 20, 20, x.shape[-1])  # (Batch, size_x, size_y, size_z, channel)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # [B, X, Y, Z, channel + 3]
        x = self.in_proj(x)  # [B, X, Y, Z, H]
        x = x.permute(0, 4, 1, 2, 3)  # [B, H, X, Y, Z]
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
        x = x.permute(0, 2, 3, 4, 1)  # [B, X, Y, Z, H]

        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, _ = layer(x)
            x = x + b

        b = b[..., :-self.padding, :-self.padding, :-self.padding, :]
        output = self.out(b)

        return output

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
