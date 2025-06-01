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


def load_combined_vtu(key, q_path, t_path, surf_path=None):
    q_pts, q_data = load_vtu_point_data(q_path)
    t_pts, t_data = load_vtu_point_data(t_path)
    if not np.allclose(q_pts, t_pts):
        raise ValueError(f"Point mismatch at {key}")
    pos = torch.tensor(q_pts, dtype=torch.float)
    t = torch.tensor(np.vstack(list(t_data.values())).T, dtype=torch.float)
    q_raw = next(iter(q_data.values()))
    q = torch.tensor(q_raw[:, None] if q_raw.ndim == 1 else q_raw, dtype=torch.float)

    data = Data(pos=pos, t=t, q=q)

    if surf_path is not None:
        surf_pts, surf_data = load_vtu_point_data(surf_path)
        surf_array = np.vstack(list(surf_data.values())).T
        data.surf = torch.tensor(surf_array, dtype=torch.float)

    return data


def downsample_data(data, keep_count=None, ratio=None, surf_keep_count=None, surf_ratio=None):
    num_points = data.pos.size(0)
    if keep_count is not None:
        keep = min(num_points, keep_count)
    elif ratio is not None:
        if not (0 < ratio <= 1):
            raise ValueError("ratio must be between 0 and 1")
        keep = max(1, int(num_points * ratio))
    else:
        keep = num_points

    idx = np.random.choice(num_points, keep, replace=False)
    downsampled = Data(
        pos=data.pos[idx],
        t=data.t[idx],
        q=(data.q[idx] if data.q.dim() > 0 else torch.tensor(data.q, dtype=torch.float)[idx])
    )

    if hasattr(data, 'surf'):
        surf_points = data.surf.size(0)
        if surf_keep_count is not None:
            surf_keep = min(surf_points, surf_keep_count)
        elif surf_ratio is not None:
            if not (0 < surf_ratio <= 1):
                raise ValueError("surf_ratio must be between 0 and 1")
            surf_keep = max(1, int(surf_points * surf_ratio))
        else:
            surf_keep = surf_points

        surf_idx = np.random.choice(surf_points, surf_keep, replace=False)
        downsampled.surf = data.surf[surf_idx]

    return downsampled


def load_all_data(root_dir,
                  max_workers=8,
                  data_num=None,
                  downsample_count=None,
                  downsample_ratio=None,
                  surf_downsample_ratio=None,
                  surf_downsample_count=None,
                  data_type='unstructured_data',
                  normalize=True,
                  use_surf=True):
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
        surf_map = {}

        if use_surf:
            surf_dir = os.path.join(sim, 'surf_input')
            surf_files = sorted(glob.glob(os.path.join(surf_dir, '*_Tsurf.vtu')))
            surf_map = {os.path.basename(p).replace('_Tsurf.vtu', ''): p for p in surf_files}

        for key in sorted(q_map):
            if key in t_map and (not use_surf or key in surf_map):
                tasks.append((key, q_map[key], t_map[key], surf_map.get(key)))
            else:
                print(f"Missing pair or surface for key: {key} in {sim}")

    if data_num is not None:
        tasks = tasks[:data_num]
    if not tasks:
        raise RuntimeError("No data tasks found")

    data_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_combined_vtu, key, q, t, surf): (key, surf)
            for key, q, t, surf in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading all files"):
            key, surf = futures[fut]
            try:
                data = fut.result()
                data = downsample_data(data,
                                       keep_count=downsample_count,
                                       ratio=downsample_ratio,
                                       surf_keep_count=surf_downsample_count,
                                       surf_ratio=surf_downsample_ratio)
                if data.t.numel() > 0 and data.pos.numel() > 0 and data.q.numel() > 0 and (not use_surf or data.surf.numel() > 0):
                    data_list.append(data)
            except Exception as e:
                print(f"Skipping {key}: {e}")

    if not data_list:
        raise RuntimeError("No data loaded")

    stats = {}
    if normalize:
        total_points = 0
        mean_t, mean_pos, mean_q, mean_surf = None, None, None, None
        for d in data_list:
            t_np = d.t.numpy()
            pos_np = d.pos.numpy()
            q_np = d.q.numpy()
            surf_np = d.surf.numpy() if use_surf and hasattr(d, 'surf') else None
            n = t_np.shape[0]
            if n == 0:
                continue
            if mean_t is None:
                mean_t, mean_pos, mean_q = t_np.mean(0), pos_np.mean(0), q_np.mean(0)
                if use_surf and surf_np is not None:
                    mean_surf = surf_np.mean(0)
            else:
                new_total = total_points + n
                mean_t = (mean_t * total_points + t_np.sum(0)) / new_total
                mean_pos = (mean_pos * total_points + pos_np.sum(0)) / new_total
                mean_q = (mean_q * total_points + q_np.sum(0)) / new_total
                if use_surf and surf_np is not None:
                    mean_surf = (mean_surf * total_points + surf_np.sum(0)) / new_total
            total_points += n

        var_t = np.zeros_like(mean_t)
        var_pos = np.zeros_like(mean_pos)
        var_q = np.zeros_like(mean_q)
        var_surf = np.zeros_like(mean_surf) if use_surf else None
        for d in data_list:
            t_np = d.t.numpy()
            pos_np = d.pos.numpy()
            q_np = d.q.numpy()
            surf_np = d.surf.numpy() if use_surf and hasattr(d, 'surf') else None
            diff_t = t_np - mean_t
            diff_pos = pos_np - mean_pos
            diff_q = q_np - mean_q
            var_t += (diff_t ** 2).sum(0)
            var_pos += (diff_pos ** 2).sum(0)
            var_q += (diff_q ** 2).sum(0)
            if use_surf and surf_np is not None:
                diff_surf = surf_np - mean_surf
                var_surf += (diff_surf ** 2).sum(0)

        std_t = np.sqrt(np.clip(var_t / total_points, a_min=0, a_max=None))
        std_pos = np.sqrt(np.clip(var_pos / total_points, a_min=0, a_max=None))
        std_q = np.sqrt(np.clip(var_q / total_points, a_min=0, a_max=None))
        std_surf = np.sqrt(np.clip(var_surf / total_points, a_min=0, a_max=None)) if use_surf else None

        for d in data_list:
            d.t = torch.tensor(((d.t.numpy() - mean_t) / (std_t + 1e-8)).astype(np.float32))
            d.pos = torch.tensor(((d.pos.numpy() - mean_pos) / (std_pos + 1e-8)).astype(np.float32))
            d.q = torch.tensor(((d.q.numpy() - mean_q) / (std_q + 1e-8)).astype(np.float32))
            if use_surf and hasattr(d, 'surf'):
                d.surf = torch.tensor(((d.surf.numpy() - mean_surf) / (std_surf + 1e-8)).astype(np.float32))

        stats = {'mean_t': mean_t, 'std_t': std_t, 'mean_pos': mean_pos,
                 'std_pos': std_pos, 'mean_q': mean_q, 'std_q': std_q}
        if use_surf:
            stats.update({'mean_surf': mean_surf, 'std_surf': std_surf})

    return data_list, stats


if __name__ == '__main__':
    ds_structured, stats_structured = load_all_data(
        "/dataset path/",
        max_workers=8,
        data_num=200,
        downsample_count=8000,
        surf_downsample_count=8000,
        data_type='structured_data'
    )
    print(f"Structured total: {len(ds_structured)}")

    ds_unstructured, stats_unstructured = load_all_data(
        "/dataset path/",
        max_workers=8,
        data_num=100,
        downsample_count=8000,
        surf_downsample_ratio=0.5,
        data_type='unstructured_data'
    )
    print(f"Unstructured total: {len(ds_unstructured)}")
