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

def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

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
                weight = torch.FloatTensor(in_dim, out_dim, n_modes)
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
        x_ftz = dct(x, norm='ortho')

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            self.fourier_weight[2])

        xz = idct(out_ft, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_2 = x.transpose(-1, -2)
        x_fty = dct(x_2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S3, S2)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :, :self.modes_y] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_fty[:, :, :, :, :self.modes_y],
            self.fourier_weight[1])

        xy = idct(out_ft, norm='ortho')
        xy = xy.transpose(-1, -2)
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_3 = x.transpose(-1, -3)
        x_ftx = dct(x_3, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S3, S2, S1)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :, :self.modes_x] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftx[:, :, :, :, :self.modes_x],
            self.fourier_weight[0])

        xx = idct(out_ft, norm='ortho')
        xx = xx.transpose(-1, -3)
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + xz

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class CNOFactorizedMesh3D(nn.Module):
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
        x = torch.cat((x, grid), dim=-1)  # [B, X, Y, Z,  channel + 3]
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
