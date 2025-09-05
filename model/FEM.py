import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import torch
import argparse
import train
from dataset import load_all_data
import os


class FEMHeatSolver(nn.Module):
    def __init__(self, mode='Q2T', num_time_steps=13, dt=0.01, alpha=1.0, rho=1.0, c=1.0, num_points=8000):
        super(FEMHeatSolver, self).__init__()
        self.mode = mode  # Q2T, T2T, T2Q
        self.num_time_steps = num_time_steps if mode == 'T2T' else 13
        self.dt = dt
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.rho_c = nn.Parameter(torch.tensor(rho * c, dtype=torch.float32))
        self.num_points = num_points

        # Projection layers
        self.time_projection = nn.Linear(13 - num_time_steps, 1)
        # self.input_projection = nn.Linear(in_channels, 128) if in_channels > 1 else None
        # self.output_projection = nn.Linear(128, out_channels) if out_channels > 1 else None

        # FEM Matrices
        self.stiffness_matrix, self.mass_matrix = self._build_fem_matrices()

    def _build_fem_matrices(self):
        stiffness = np.eye(self.num_points)
        mass = np.eye(self.num_points)
        return csr_matrix(stiffness), csr_matrix(mass)

    def forward(self, x, grid=None):
        """
        x: (batch_size, num_points, in_channels)
        """
        batch_size = x.shape[0]
        device = x.device

        if self.mode == 'Q2T':
            Q = x[:, :, 0]
            T = torch.zeros(batch_size, self.num_points, self.num_time_steps, device=device)
            T_current = torch.zeros(batch_size, self.num_points, device=device)

            for t in range(self.num_time_steps):
                laplacian_T = torch.stack([
                    torch.from_numpy(self.stiffness_matrix.dot(T_current[b].cpu().detach().numpy()))
                    for b in range(T_current.shape[0])
                ], dim=0).to(device)
                dT_dt = (Q / self.rho_c) + (self.alpha * laplacian_T)
                T_current = T_current + self.dt * dT_dt
                T[:, :, t] = T_current

            return T

        elif self.mode == 'T2T':
            # x: (B, N, Tin)
            Tin = x.shape[-1]
            Tout = self.num_time_steps  # = 2

            T_pred = torch.zeros(batch_size, self.num_points, Tout, device=device)
            T_current = self.time_projection(x).squeeze(-1)

            for t in range(Tout):
                laplacian_T = torch.stack([
                    torch.from_numpy(self.stiffness_matrix.dot(T_current[b].cpu().detach().numpy()))
                    for b in range(batch_size)
                ], dim=0).to(device)

                dT_dt = self.alpha * laplacian_T
                T_current = T_current + self.dt * dT_dt
                T_pred[:, :, t] = T_current

            return T_pred  # shape: (B, N, Tout)


        elif self.mode == 'T2Q':
            T_observed = x
            Q_estimated = torch.zeros(batch_size, self.num_points, 1, device=device)
            T_current = torch.zeros(batch_size, self.num_points, device=device)

            for t in range(self.num_time_steps):
                T_next = T_observed[:, :, t]
                laplacian_T = torch.stack([
                    torch.from_numpy(self.stiffness_matrix.dot(T_current[b].cpu().detach().numpy()))
                    for b in range(T_current.shape[0])
                ], dim=0).to(device)
                if t == 0:
                    dT_dt = (T_next - T_current) / self.dt
                else:
                    dT_dt = (T_next - T_observed[:, :, t - 1]) / self.dt

                Q_t = (dT_dt - self.alpha * laplacian_T) * self.rho_c
                Q_estimated += Q_t.unsqueeze(-1)
                T_current = T_next

            return Q_estimated

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")