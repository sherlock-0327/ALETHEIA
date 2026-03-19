import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


################################################################
# 3D Gabor Layer
################################################################
class GaborConv3d(nn.Module):
    def __init__(self, kernel_num=1):
        super().__init__()
        self.kernel_num = kernel_num

        # 3D Gabor parameters
        # freq: center frequency magnitude
        self.freq = nn.Parameter(torch.tensor([1.1107]), requires_grad=True)

        # Orientation angles in 3D: theta (azimuth), phi (elevation)
        self.theta = nn.Parameter(torch.tensor([0.39]), requires_grad=True)
        self.phi = nn.Parameter(torch.tensor([0.39]), requires_grad=True)

        # Gaussian envelope widths
        self.sigma = nn.Parameter(torch.tensor([2.82]), requires_grad=True)
        # Gamma parameters to control aspect ratio in the two perpendicular directions
        self.gamma1 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, size_x, size_y, size_z, device):
        # Create 3D grid centered at 0
        x = torch.linspace(-0.5, 0.5, size_x, device=device)
        y = torch.linspace(-0.5, 0.5, size_y, device=device)
        z = torch.linspace(-0.5, 0.5, size_z, device=device)

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        # Rotation logic:
        # Construct a rotated coordinate system (u, v, w) based on theta and phi.
        # u points in the direction of the frequency vector.
        # v, w are orthogonal.

        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)

        # Coordinate transformation (Spherical-like rotation)
        # u is the longitudinal direction (propagation direction)
        u = grid_x * cos_theta * sin_phi + grid_y * sin_theta * sin_phi + grid_z * cos_phi
        # v is one lateral direction
        v = grid_x * cos_theta * cos_phi + grid_y * sin_theta * cos_phi - grid_z * sin_phi
        # w is the other lateral direction
        w = -grid_x * sin_theta + grid_y * cos_theta

        # Calculate Gabor envelope
        # We assume the filter is a Gaussian centered at 'freq' along u-axis, and 0 along v, w.
        f = self.freq
        sigma_x = self.sigma
        sigma_y = sigma_x / (self.gamma1 + 1e-5)
        sigma_z = sigma_x / (self.gamma2 + 1e-5)

        test1 = sigma_x ** 2 * (u - (f / np.pi)) ** 2
        test2 = sigma_y ** 2 * v ** 2
        test3 = sigma_z ** 2 * w ** 2

        weight = torch.exp(-2 * np.pi ** 2 * (test1 + test2 + test3))

        return weight  # shape: (size_x, size_y, size_z)


################################################################
# 3D Spectral Layer with Gabor Filter
################################################################
class SpectralConv3d_Gabor(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_Gabor, self).__init__()

        """
        3D Fourier layer with Gabor Filtering.
        It does FFT, Gabor weighting, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))

        # We need 4 weights for 3D RFFT because:
        # dim 1 (x) and dim 2 (y) are fully wrapped (positive & negative freq).
        # dim 3 (z) is RFFT (only positive freq).
        # So we have 4 corners: (+,+), (-,+), (+,-), (-,-) in the (x,y) plane.
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

        # Gabor Filter
        self.g0 = GaborConv3d(kernel_num=1)

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,z ), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # 1. Compute FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # 2. Compute Gabor Mask
        # Note: We generate the mask on the full spatial grid size, then FFT-shift it to align with frequency modes
        size_x, size_y, size_z = x.size(-3), x.size(-2), x.size(-1)

        gabor = self.g0(size_x, size_y, size_z, x.device)
        gabor = torch.fft.ifftshift(gabor)  # Shift zero freq to corner

        # Add batch and channel dimensions for broadcasting: (1, 1, x, y, z)
        gabor = gabor.unsqueeze(0).unsqueeze(0)

        # 3. Slice Gabor mask for the 4 corners of the Fourier Spectrum
        # RFFT structure:
        # axis -3 (x): 0 to modes (pos), -modes to end (neg)
        # axis -2 (y): 0 to modes (pos), -modes to end (neg)
        # axis -1 (z): 0 to modes (pos only)

        # Corner 1: (Pos X, Pos Y)
        gabor1 = gabor[..., :self.modes1, :self.modes2, :self.modes3]
        # Corner 2: (Neg X, Pos Y)
        gabor2 = gabor[..., -self.modes1:, :self.modes2, :self.modes3]
        # Corner 3: (Pos X, Neg Y)
        gabor3 = gabor[..., :self.modes1, -self.modes2:, :self.modes3]
        # Corner 4: (Neg X, Neg Y)
        gabor4 = gabor[..., -self.modes1:, -self.modes2:, :self.modes3]

        # 4. Apply Weights and Gabor Filters
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Apply weights * gabor_slice
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1 * gabor1)

        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2 * gabor2)

        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3 * gabor3)

        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4 * gabor4)

        # 5. Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


################################################################
# 3D GFNO Model
################################################################
class GFNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels, H, W, D):
        super(GFNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv (SpectralConv3d_Gabor).
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.H = H
        self.W = W
        self.D = D
        self.width = width
        # Padding is optional depending on boundary conditions, keeping it small as in reference
        self.padding = 5

        self.fc0 = nn.Linear(in_channels + 3, self.width)

        # Replace standard SpectralConv3d with Gabor version
        self.conv0 = SpectralConv3d_Gabor(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_Gabor(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_Gabor(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_Gabor(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        # Optional: BatchNorms were in FNO3d reference, included here
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, pos, surf_pos=None):
        # x shape input: (Batch, Size_x, Size_y, Size_z, In_Channels)
        # Note: If input includes grid (x,y,z,t), ensure in_channels matches.
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C) # (Batch, size_x, size_y, size_z, channel)
        grid = pos.reshape(B, self.H, self.W, self.D, -1)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, X, Y, Z)

        # Pad if non-periodic
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Crop padding
        x = x[..., :-self.padding, :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 4, 1)  # (B, X, Y, Z, C)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(B, N, -1)

        return x