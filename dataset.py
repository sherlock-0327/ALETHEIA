import os
import glob
import numpy as np
import pyvista as pv
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# initial temperature setting
init_temperature = 293.15

def load_vtu_point_data(vtu_path):
    mesh = pv.read(vtu_path)
    return mesh.points, {k: mesh.point_data[k] for k in mesh.point_data}


def load_combined_vtu(key, q_path, t_path, surf_path=None):
    q_pts, q_data = load_vtu_point_data(q_path)
    t_pts, t_data = load_vtu_point_data(t_path)

    if not np.allclose(q_pts, t_pts, rtol=1e-5, atol=1e-8):
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
        data.surf_pos = torch.tensor(surf_pts, dtype=torch.float)

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

    # idx = np.random.choice(num_points, keep, replace=False)
    idx = np.linspace(0, num_points - 1, num=keep, dtype=int)

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
        downsampled.surf_pos = data.surf_pos[surf_idx]

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
                  unit_normalize=True,
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
        tasks = random.sample(tasks, data_num)
    if not tasks:
        raise RuntimeError("No data tasks found")

    data_list = []
    skipped = 0
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
                else:
                    skipped += 1
            except Exception as e:
                print(f"Skipping {key}: {e}")
                skipped += 1

    if not data_list:
        raise RuntimeError("No data loaded")

    print(f"Loaded {len(data_list)} samples, Skipped {skipped}")

    stats = {}
    if normalize:
        if unit_normalize:
            norm_pos, norm_surf_pos = None, None
            for d in data_list:
                d.t = d.t - init_temperature
                mean_t = d.t.mean(0, keepdim=True)
                std_t = d.t.std(0, unbiased=False, keepdim=True) + 1e-8
                d.t = (d.t - mean_t) / std_t

                if norm_pos is None:
                    mean_pos = d.pos.mean(0, keepdim=True)
                    std_pos = d.pos.std(0, unbiased=False, keepdim=True) + 1e-8
                    norm_pos = (d.pos - mean_pos) / std_pos
                d.pos = norm_pos

                mean_q = d.q.mean(0, keepdim=True)
                std_q = d.q.std(0, unbiased=False, keepdim=True) + 1e-8
                d.q = (d.q - mean_q) / std_q

                if use_surf and hasattr(d, 'surf'):
                    d.surf = d.surf - init_temperature
                    mean_surf = d.surf.mean(0, keepdim=True)
                    std_surf = d.surf.std(0, unbiased=False, keepdim=True) + 1e-8
                    d.surf = (d.surf - mean_surf) / std_surf

                    if norm_surf_pos is None:
                        mean_surf_pos = d.surf_pos.mean(0, keepdim=True)
                        std_surf_pos = d.surf_pos.std(0, unbiased=False, keepdim=True) + 1e-8
                        norm_surf_pos = (d.surf_pos - mean_surf_pos) / std_surf_pos
                    d.surf_pos = norm_surf_pos

            stats = {}
        else:
            total_main_points = 0
            total_surf_points = 0

            mean_t, mean_pos, mean_q = None, None, None
            mean_surf, mean_surf_pos = None, None
            for d in data_list:
                t_np = d.t.numpy()
                pos_np = d.pos.numpy()
                q_np = d.q.numpy()

                n = t_np.shape[0]

                if mean_t is None:
                    mean_t, mean_pos, mean_q = t_np.mean(0), pos_np.mean(0), q_np.mean(0)
                else:
                    new_total = total_main_points + n
                    mean_t = (mean_t * total_main_points + t_np.sum(0)) / new_total
                    mean_pos = (mean_pos * total_main_points + pos_np.sum(0)) / new_total
                    mean_q = (mean_q * total_main_points + q_np.sum(0)) / new_total
                total_main_points += n

                if use_surf and hasattr(d, 'surf'):
                    surf_np = d.surf.numpy()
                    surf_pos_np = d.surf_pos.numpy()
                    m = surf_np.shape[0]
                    if mean_surf is None:
                        mean_surf = surf_np.mean(0)
                        mean_surf_pos = surf_pos_np.mean(0)
                    else:
                        new_surf_total = total_surf_points + m
                        mean_surf = (mean_surf * total_surf_points + surf_np.sum(0)) / new_surf_total
                        mean_surf_pos = (mean_surf_pos * total_surf_points + surf_pos_np.sum(0)) / new_surf_total
                    total_surf_points += m

            var_t = np.zeros_like(mean_t)
            var_pos = np.zeros_like(mean_pos)
            var_q = np.zeros_like(mean_q)
            var_surf = np.zeros_like(mean_surf) if use_surf else None
            var_surf_pos = np.zeros_like(mean_surf_pos) if use_surf else None

            for d in data_list:
                t = d.t
                pos = d.pos
                q = d.q
                var_t += ((t - torch.from_numpy(mean_t)) ** 2).sum(0).numpy()
                var_pos += ((pos - torch.from_numpy(mean_pos)) ** 2).sum(0).numpy()
                var_q += ((q - torch.from_numpy(mean_q)) ** 2).sum(0).numpy()

                if use_surf and hasattr(d, 'surf'):
                    surf = d.surf
                    surf_pos = d.surf_pos
                    var_surf += ((surf - torch.from_numpy(mean_surf)) ** 2).sum(0).numpy()
                    var_surf_pos += ((surf_pos - torch.from_numpy(mean_surf_pos)) ** 2).sum(0).numpy()

            std_t = np.sqrt(np.clip(var_t / total_main_points, a_min=0, a_max=None))
            std_pos = np.sqrt(np.clip(var_pos / total_main_points, a_min=0, a_max=None))
            std_q = np.sqrt(np.clip(var_q / total_main_points, a_min=0, a_max=None))
            std_surf = np.sqrt(np.clip(var_surf / total_surf_points, a_min=0, a_max=None)) if use_surf else None
            std_surf_pos = np.sqrt(np.clip(var_surf_pos / total_surf_points, a_min=0, a_max=None)) if use_surf else None

            for d in data_list:
                d.t = (d.t - torch.from_numpy(mean_t).float()) / (torch.from_numpy(std_t).float() + 1e-8)
                d.pos = (d.pos - torch.from_numpy(mean_pos).float()) / (torch.from_numpy(std_pos).float() + 1e-8)
                d.q = (d.q - torch.from_numpy(mean_q).float()) / (torch.from_numpy(std_q).float() + 1e-8)
                if use_surf and hasattr(d, 'surf'):
                    d.surf = (d.surf - torch.from_numpy(mean_surf).float()) / (torch.from_numpy(std_surf).float() + 1e-8)
                    d.surf_pos = (d.surf_pos - torch.from_numpy(mean_surf_pos).float()) / (torch.from_numpy(std_surf_pos).float() + 1e-8)

            stats = {'mean_t': mean_t, 'std_t': std_t, 'mean_pos': mean_pos,
                     'std_pos': std_pos, 'mean_q': mean_q, 'std_q': std_q}
            if use_surf:
                stats.update({'mean_surf': mean_surf, 'std_surf': std_surf,
                              'mean_surf_pos': mean_surf_pos, 'std_surf_pos': std_surf_pos})

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
