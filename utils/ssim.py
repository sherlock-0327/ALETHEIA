import torch
import numpy as np
from scipy.interpolate import griddata

@torch.no_grad()
def voxelize_pointcloud_batch(pos, values, grid_size=(64, 64, 64),
                              method='nearest', bounds=None):
    """
    pos:    [B, N, 3]
    values: [B, N, C]
    grid_size: (Dx, Dy, Dz)
    method: 'linear' | 'nearest' | 'cubic'
    bounds: ((x_min,y_min,z_min),(x_max,y_max,z_max))
    """
    assert pos.ndim == 3 and values.ndim == 3
    B, N, _ = pos.shape
    C = values.shape[2]
    Dx, Dy, Dz = grid_size

    pos_np = pos.detach().cpu().numpy()
    val_np = values.detach().cpu().numpy()
    grids = []

    for b in range(B):
        p = pos_np[b]      # [N,3]
        v = val_np[b]      # [N,C]

        if bounds is None:
            (x_min, y_min, z_min) = p.min(axis=0)
            (x_max, y_max, z_max) = p.max(axis=0)
        else:
            (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds

        gx = np.linspace(x_min, x_max, Dx)
        gy = np.linspace(y_min, y_max, Dy)
        gz = np.linspace(z_min, z_max, Dz)
        grid_x, grid_y, grid_z = np.meshgrid(gx, gy, gz, indexing='ij')

        grid = np.zeros((C, Dx, Dy, Dz), dtype=np.float32)
        for c in range(C):
            grid[c] = griddata(points=p, values=v[:, c],
                               xi=(grid_x, grid_y, grid_z),
                               method=method, fill_value=0.0)
        grids.append(torch.from_numpy(grid))

    return torch.stack(grids, dim=0)  # [B, C, Dx, Dy, Dz]

@torch.no_grad()
def ssim3d_pointcloud_monai(pos, pred, target, grid_size=(64,64,64),
                            method='nearest', max_val=None, bounds=None):
    """
    pos:    [B, N, 3]
    pred:   [B, N, C]
    target: [B, N, C]
    """
    vol_pred = voxelize_pointcloud_batch(pos, pred,   grid_size, method, bounds)
    vol_tgt  = voxelize_pointcloud_batch(pos, target, grid_size, method, bounds)

    if max_val is None:
        max_val = (vol_tgt.max() - vol_tgt.min())
        max_val = float(max_val.item()) if max_val > 0 else 1.0

    ssim3d = SSIM3D(window_size=11, max_val=max_val, padding='same')
    return ssim3d(vol_pred, vol_tgt).mean(dim=(1, 2, 3, 4)).mean().numpy()

@torch.no_grad()
def point_ssim(pred, target, pos, k=20, c1=1e-4, c2=9e-4):
    """
    pred, target: (B, N, C)
    pos: (B, N, 3)
    """
    B, N, C = pred.shape

    dist = torch.cdist(pos, pos)  # (B, N, N)
    _, idx = dist.topk(k, largest=False)  # (B, N, k)

    def get_neighbor_feat(x, idx):
        # x: (B, N, C), idx: (B, N, k)
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        return x[batch_idx, idx, :]

    pred_knn = get_neighbor_feat(pred, idx)
    target_knn = get_neighbor_feat(target, idx)

    mu_p = pred_knn.mean(dim=2)  # (B, N, C)
    mu_t = target_knn.mean(dim=2)  # (B, N, C)

    sigma_p = pred_knn.var(dim=2)  # (B, N, C)
    sigma_t = target_knn.var(dim=2)

    sigma_pt = ((pred_knn - mu_p.unsqueeze(2)) * (target_knn - mu_t.unsqueeze(2))).mean(dim=2)

    #SSIM Algorithm
    l = (2 * mu_p * mu_t + c1) / (mu_p ** 2 + mu_t ** 2 + c1)
    cs = (2 * sigma_pt + c2) / (sigma_p + sigma_t + c2)

    ssim_map = l * cs
    return ssim_map.mean()
