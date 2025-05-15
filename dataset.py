import os
import glob
import numpy as np
import pyvista as pv
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_vtu_point_data(vtu_path):
    mesh = pv.read(vtu_path)
    return mesh.points, {k: mesh.point_data[k] for k in mesh.point_data}


def load_pair(key, q_path, t_path):
    q_pts, q_data = load_vtu_point_data(q_path)
    t_pts, t_data = load_vtu_point_data(t_path)
    if not np.allclose(q_pts, t_pts):
        raise ValueError(f"Point mismatch at {key}")
    pos = torch.tensor(q_pts, dtype=torch.float)
    x = torch.tensor(np.vstack(list(t_data.values())).T, dtype=torch.float)
    y = torch.tensor(next(iter(q_data.values())), dtype=torch.float)
    return Data(pos=pos, x=x, y=y)


def downsample_data(data, keep_count=None, ratio=None):
    """
    Reduces the number of points in Data evenly, either by specifying the number of points to keep or the percentage to keep.
    """
    num_points = data.pos.size(0)

    # Prioritize the use of specified points
    if keep_count is not None:
        if keep_count < 1:
            raise ValueError("keep_count must be >= 1")
        keep = min(num_points, keep_count)
    elif ratio is not None:
        if not (0 < ratio <= 1):
            raise ValueError("ratio must be between 0 and 1")
        keep = max(1, int(num_points * ratio))
    else:
        return data

    idx = np.linspace(0, num_points - 1, keep, dtype=int)
    return Data(
        pos=data.pos[idx],
        x=data.x[idx],
        y=(data.y[idx] if data.y.dim() > 0 else data.y)
    )


def load_all_data(root_dir,
                  max_workers=8,
                  data_num=None,
                  downsample_count=None,
                  downsample_ratio=None,
                  data_type='unstructured_data',
                  normalize=True):
    """
    VTU data are loaded and optionally Z-score standardized.
    The standardization process uses mean, unbiased variance calculations.
    """
    # Collect all (Q, T) file pairs
    tasks = []
    sim_dirs = [os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))]
    for sim in sim_dirs:
        data_dir = os.path.join(sim, data_type)
        q_files = sorted(glob.glob(os.path.join(data_dir, '*_Q.vtu')))
        t_files = sorted(glob.glob(os.path.join(data_dir, '*_T.vtu')))
        if len(q_files) != len(t_files):
            print(f"Skipping {sim}: `{data_type}` Mismatch in number of documents")
            continue
        q_map = {os.path.basename(p).replace('_Q.vtu', ''): p for p in q_files}
        t_map = {os.path.basename(p).replace('_T.vtu', ''): p for p in t_files}
        for key in sorted(q_map):
            if key in t_map:
                tasks.append((key, q_map[key], t_map[key], sim))

    if data_num is not None:
        tasks = tasks[:data_num]
    if not tasks:
        raise RuntimeError("No data tasks found")

    data_list = []
    # Concurrent loading and downsampling
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_pair, key, q, t): (key, sim)
                   for key, q, t, sim in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading all files"):
            key, sim = futures[fut]
            try:
                data = fut.result()
                data = downsample_data(data,
                                       keep_count=downsample_count,
                                       ratio=downsample_ratio)
                data_list.append(data)
            except Exception as e:
                print(f"Skipping {sim}/{key}: {e}")
    if not data_list:
        raise RuntimeError("No data loaded")


    stats = {}
    if normalize:
        # 1. Calculate the global mean (online algorithm to avoid running out of memory)
        total_points = 0
        mean_x = None
        mean_pos = None
        mean_y = None
        for d in data_list:
            x_np = d.x.numpy()
            pos_np = d.pos.numpy()
            y_np = d.y.numpy().reshape(-1, 1) if d.y.dim() == 1 else d.y.numpy()
            n = x_np.shape[0]
            if mean_x is None:
                mean_x = x_np.mean(axis=0)
                mean_pos = pos_np.mean(axis=0)
                mean_y = y_np.mean(axis=0)
                total_points = n
            else:
                # Incremental calculation of mean value
                new_total = total_points + n
                mean_x = (mean_x * total_points + x_np.sum(axis=0)) / new_total
                mean_pos = (mean_pos * total_points + pos_np.sum(axis=0)) / new_total
                mean_y = (mean_y * total_points + y_np.sum(axis=0)) / new_total
                total_points = new_total

        # 2. Calculate overall variance (divided by n)
        total_points = 0
        var_x = np.zeros_like(mean_x)
        var_pos = np.zeros_like(mean_pos)
        var_y = np.zeros_like(mean_y)
        for d in data_list:
            x_np = d.x.numpy()
            pos_np = d.pos.numpy()
            y_np = d.y.numpy().reshape(-1, 1) if d.y.dim() == 1 else d.y.numpy()
            n = x_np.shape[0]
            diff_x = x_np - mean_x
            diff_pos = pos_np - mean_pos
            diff_y = y_np - mean_y
            var_x += (diff_x ** 2).sum(axis=0)
            var_pos += (diff_pos ** 2).sum(axis=0)
            var_y += (diff_y ** 2).sum(axis=0)
            total_points += n

        std_x = np.sqrt(var_x / total_points)  # Overall std
        std_pos = np.sqrt(var_pos / total_points)
        std_y = np.sqrt(var_y / total_points)

        # 3. Application standardization
        for d in data_list:
            d.x = ((d.x.numpy() - mean_x) / (std_x + 1e-8)).astype(np.float32)
            d.x = torch.tensor(d.x)
            d.pos = ((d.pos.numpy() - mean_pos) / (std_pos + 1e-8)).astype(np.float32)
            d.pos = torch.tensor(d.pos)
            if d.y.dim() == 1:
                y_norm = (d.y.numpy().reshape(-1, 1) - mean_y) / (std_y + 1e-8)
                d.y = torch.tensor(y_norm.reshape(-1), dtype=torch.float)
            else:
                y_norm = (d.y.numpy() - mean_y) / (std_y + 1e-8)
                d.y = torch.tensor(y_norm, dtype=torch.float)

        stats = {'mean_x': mean_x, 'std_x': std_x, 'mean_pos': mean_pos, 'std_pos': std_pos, 'mean_y': mean_y, 'std_y': std_y}

    return data_list, stats


if __name__ == '__main__':
    ds2 = load_all_data(
        "/dataset path/",
        max_workers=8,
        data_num=100,
        downsample_count=8000,
        data_type='structured_data'
    )
    print(f"Structured total: {len(ds2)}")

    ds1 = load_all_data(
        "/dataset path/",
        max_workers=8,
        data_num=100,
        downsample_count=8000,
        data_type='unstructured_data'
    )
    print(f"Unstructured total: {len(ds1)}")


