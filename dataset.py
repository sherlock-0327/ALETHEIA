import os
import glob
import numpy as np
import pyvista as pv
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from typing import List, Dict, Tuple, Optional
import math

# initial temperature setting
init_temperature = 293.15


@torch.no_grad()
def _row_all_finite(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return torch.isfinite(x)
    return torch.isfinite(x).all(dim=1)


def get_freq_value(key: str) -> int:
    # Extract numeric frequency value for sorting
    m = re.search(r'_(\d+)(kHz|Hz|MHz)_', key)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if unit == 'MHz': val *= 1000
        return val
    return 0


def _parse_prefix_and_freq_token(key: str):
    # Regex to identify simulation prefix and frequency token
    _FREQ_RE = re.compile(r'(.+?)_([0-9]+)(kHz|Hz|MHz)_([0-9]+)A$')
    m = _FREQ_RE.match(key)
    if not m:
        toks = re.findall(r'([0-9]+(?:kHz|Hz|MHz))', key)
        freq = toks[-1] if toks else None
        if freq and f"_{freq}_" in key:
            prefix = key.rsplit(f"_{freq}_", 1)[0]
        else:
            prefix = key
        return prefix, freq
    prefix, v, unit, amp = m.groups()
    return prefix, f"{int(v)}{unit}"


# ------------------------------------------------------------------------------
# Loading & Preprocessing Primitives
# ------------------------------------------------------------------------------

def load_vtu_point_data(vtu_path):
    mesh = pv.read(vtu_path)
    return mesh.points, {k: mesh.point_data[k] for k in mesh.point_data}


def drop_nan_rows(data, min_keep: int = None, log_prefix: str = "", drop_surface: bool = True):
    assert hasattr(data, "pos") and hasattr(data, "t") and hasattr(data, "q")
    N0 = data.pos.size(0)

    q = data.q
    if q.dim() == 2 and q.size(1) == 1:
        q_mask = _row_all_finite(q.squeeze(1))
    else:
        q_mask = _row_all_finite(q)

    mask_main = _row_all_finite(data.pos) & _row_all_finite(data.t) & q_mask
    keep_idx = mask_main.nonzero(as_tuple=False).squeeze(1)

    if keep_idx.numel() == 0:
        raise ValueError(f"{log_prefix} all points are invalid (NaN/Inf).")

    if min_keep is not None and keep_idx.numel() < min_keep:
        raise ValueError(f"{log_prefix} valid points {keep_idx.numel()} < min_keep {min_keep}.")

    data.pos = data.pos[keep_idx]
    data.t = data.t[keep_idx]
    data.q = data.q[keep_idx]

    if drop_surface and hasattr(data, "surf") and hasattr(data, "surf_pos"):
        Ns0 = data.surf_pos.size(0)
        mask_surf = _row_all_finite(data.surf_pos) & _row_all_finite(data.surf)
        keep_sidx = mask_surf.nonzero(as_tuple=False).squeeze(1)
        if keep_sidx.numel() > 0:
            data.surf_pos = data.surf_pos[keep_sidx]
            data.surf = data.surf[keep_sidx]
        else:
            delattr(data, "surf_pos")
            delattr(data, "surf")

    dropped = N0 - keep_idx.numel()
    data.nan_drop_ratio = float(dropped) / float(max(1, N0))
    return data


def downsample_data(data, downsample_count=None, ratio=None, surf_downsample_count=None, surf_ratio=None,
                    data_type=None, seed=42):
    rng = np.random.default_rng(seed)
    num_points = data.pos.size(0)

    if downsample_count is not None:
        keep = min(num_points, downsample_count)
    elif ratio is not None:
        keep = max(1, int(num_points * ratio))
    else:
        keep = num_points

    if data_type == 'unstructured_data':
        idx = rng.choice(num_points, keep, replace=False)
    else:
        idx = np.linspace(0, num_points - 1, num=keep, dtype=int)

    downsampled = Data(
        pos=data.pos[idx],
        t=data.t[idx],
        q=(data.q[idx] if data.q.dim() > 0 else torch.tensor(data.q, dtype=torch.float)[idx])
    )
    if hasattr(data, 'sim_key'): downsampled.sim_key = data.sim_key
    if hasattr(data, 'freq_val'): downsampled.freq_val = data.freq_val

    if hasattr(data, 'surf'):
        surf_points = data.surf.size(0)
        if surf_downsample_count is not None:
            surf_keep = min(surf_points, surf_downsample_count)
        elif surf_ratio is not None:
            surf_keep = max(1, int(surf_points * surf_ratio))
        else:
            surf_keep = surf_points

        if data_type == 'unstructured_data':
            surf_idx = rng.choice(surf_points, surf_keep, replace=False)
        else:
            surf_idx = np.linspace(0, surf_points - 1, num=surf_keep, dtype=int)
        downsampled.surf = data.surf[surf_idx]
        downsampled.surf_pos = data.surf_pos[surf_idx]

    return downsampled


def load_combined_vtu(key, q_path, t_path, surf_path=None):
    q_pts, q_data = load_vtu_point_data(q_path)
    t_pts, t_data = load_vtu_point_data(t_path)

    def sorted_t_keys(t_data):
        keys = list(t_data.keys())

        def _k(x):
            m = re.findall(r'\d+', x)
            return (int(m[0]) if m else 10 ** 9, x)

        return sorted(keys, key=_k)

    if not np.allclose(q_pts, t_pts, rtol=1e-5, atol=1e-8):
        raise ValueError(f"Point mismatch at {key}")

    pos = torch.tensor(q_pts, dtype=torch.float)
    t_keys = sorted_t_keys(t_data)
    t_stack = np.stack([t_data[k] for k in t_keys], axis=1)
    t = torch.tensor(t_stack, dtype=torch.float)

    q_raw = next(iter(q_data.values()))
    q = torch.tensor(q_raw[:, None] if q_raw.ndim == 1 else q_raw, dtype=torch.float)

    data = Data(pos=pos, t=t, q=q)

    if surf_path is not None:
        surf_pts, surf_data = load_vtu_point_data(surf_path)
        surf_keys = sorted_t_keys(surf_data)
        surf_array = np.stack([surf_data[k] for k in surf_keys], axis=1) if len(surf_keys) > 1 \
            else next(iter(surf_data.values()))
        data.surf = torch.tensor(surf_array, dtype=torch.float)
        data.surf_pos = torch.tensor(surf_pts, dtype=torch.float)

    return data


# ------------------------------------------------------------------------------
# Normalization (Decoupled & Corrected)
# ------------------------------------------------------------------------------

def normalize_datasets(train_data: List[Data], test_data: List[Data],
                       use_surf: bool = True, unit_normalize: bool = True):
    print(f"[Normalization] unit_normalize={unit_normalize}")

    if len(train_data) == 0:
        print("[Warn] No training data to compute stats. Skipping normalization.")
        return {}

    def gather(dataset, attr):
        return [getattr(d, attr) for d in dataset if hasattr(d, attr)]

    stats = {}

    # --- Compute GLOBAL Stats from Training Set ---
    all_pos = torch.cat(gather(train_data, 'pos'), dim=0)
    mean_pos = all_pos.mean(0, keepdim=True)
    std_pos = all_pos.std(0, unbiased=False, keepdim=True) + 1e-8
    stats.update({'mean_pos': mean_pos, 'std_pos': std_pos})

    mean_surf_pos, std_surf_pos = None, None
    if use_surf:
        s_pos_list = gather(train_data, 'surf_pos')
        if s_pos_list:
            all_surf_pos = torch.cat(s_pos_list, dim=0)
            mean_surf_pos = all_surf_pos.mean(0, keepdim=True)
            std_surf_pos = all_surf_pos.std(0, unbiased=False, keepdim=True) + 1e-8
            stats.update({'mean_surf_pos': mean_surf_pos, 'std_surf_pos': std_surf_pos})

    # Stats for fields (Used for global normalization or return)
    all_t = torch.cat(gather(train_data, 't'), dim=0)
    if unit_normalize:
        all_t = all_t - init_temperature
    mean_t_global = all_t.mean(0, keepdim=True)
    std_t_global = all_t.std(0, unbiased=False, keepdim=True) + 1e-8

    all_q = torch.cat(gather(train_data, 'q'), dim=0)
    mean_q_global = all_q.mean(0, keepdim=True)
    std_q_global = all_q.std(0, unbiased=False, keepdim=True) + 1e-8

    stats.update({'mean_t': mean_t_global, 'std_t': std_t_global,
                  'mean_q': mean_q_global, 'std_q': std_q_global})

    mean_surf_global, std_surf_global = None, None
    if use_surf:
        s_list = gather(train_data, 'surf')
        if s_list:
            all_surf = torch.cat(s_list, dim=0)
            if unit_normalize: all_surf = all_surf - init_temperature
            mean_surf_global = all_surf.mean(0, keepdim=True)
            std_surf_global = all_surf.std(0, unbiased=False, keepdim=True) + 1e-8
            stats.update({'mean_surf': mean_surf_global, 'std_surf': std_surf_global})

    # --- Apply Normalization ---
    all_datasets = train_data + test_data
    for d in all_datasets:
        d.pos = (d.pos - mean_pos) / std_pos
        if use_surf and hasattr(d, 'surf_pos') and mean_surf_pos is not None:
            d.surf_pos = (d.surf_pos - mean_surf_pos) / std_surf_pos

        if unit_normalize:
            # Instance Norm
            d.t = d.t - init_temperature
            inst_mean_t = d.t.mean(0, keepdim=True)
            inst_std_t = d.t.std(0, unbiased=False, keepdim=True) + 1e-8
            d.t = (d.t - inst_mean_t) / inst_std_t

            inst_mean_q = d.q.mean(0, keepdim=True)
            inst_std_q = d.q.std(0, unbiased=False, keepdim=True) + 1e-8
            d.q = (d.q - inst_mean_q) / inst_std_q

            if use_surf and hasattr(d, 'surf'):
                d.surf = d.surf - init_temperature
                inst_mean_s = d.surf.mean(0, keepdim=True)
                inst_std_s = d.surf.std(0, unbiased=False, keepdim=True) + 1e-8
                d.surf = (d.surf - inst_mean_s) / inst_std_s
        else:
            # Global Norm
            d.t = (d.t - mean_t_global) / std_t_global
            d.q = (d.q - mean_q_global) / std_q_global
            if use_surf and hasattr(d, 'surf') and mean_surf_global is not None:
                d.surf = (d.surf - mean_surf_global) / std_surf_global

    return stats


# ------------------------------------------------------------------------------
# Common Task Logic (Shared by both strategies)
# ------------------------------------------------------------------------------

def _scan_simulations(root_dirs: List[str], data_type: str, use_surf: bool) -> Dict[str, List[Tuple]]:
    """
    Scans directories and groups files by Simulation Prefix.
    Returns: {prefix: [(key, q_path, t_path, surf_path, freq_val), ...sorted by freq...]}
    """
    groups = {}  # prefix -> list of tasks (sorted by freq)

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"[Warn] Directory not found: {root_dir}")
            continue

        sim_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))]

        if glob.glob(os.path.join(root_dir, data_type, '*_Q.vtu')):
            sim_dirs = [root_dir]

        for sim in sim_dirs:
            data_dir = os.path.join(sim, data_type)
            q_files = sorted(glob.glob(os.path.join(data_dir, '*_Q.vtu')))
            t_files = sorted(glob.glob(os.path.join(data_dir, '*_T.vtu')))

            if len(q_files) == 0: continue

            q_map = {os.path.basename(p).replace('_Q.vtu', ''): p for p in q_files}
            t_map = {os.path.basename(p).replace('_T.vtu', ''): p for p in t_files}
            surf_map = {}
            if use_surf:
                surf_dir = os.path.join(sim, 'surf_input')
                s_files = sorted(glob.glob(os.path.join(surf_dir, '*_Tsurf.vtu')))
                surf_map = {os.path.basename(p).replace('_Tsurf.vtu', ''): p for p in s_files}

            for key in q_map:
                if key not in t_map: continue
                if use_surf and key not in surf_map: continue

                # Prefix identifies the "Simulation Sample"
                prefix, _ = _parse_prefix_and_freq_token(key)
                freq_val = get_freq_value(key)

                if prefix not in groups: groups[prefix] = []
                groups[prefix].append((key, q_map[key], t_map[key], surf_map.get(key), freq_val))

    # Sort each group by frequency value
    for p in groups:
        groups[p].sort(key=lambda x: x[4])

    return groups


def _apply_freq_filter(tasks_sorted, mode, is_train_set, sfo_idx=None):
    """
    Input: List of files for ONE Simulation, sorted by freq.
    Output: List of files to keep for this Simulation based on OOD rules.
    """
    n = len(tasks_sorted)
    if n < 10: return []  # Strict check for complete simulations

    if mode == 'normal' or mode is None:
        return tasks_sorted  # All 10 files

    elif mode == 'sfo':
        if sfo_idx is None or not (1 <= sfo_idx <= 10):
            raise ValueError(f"SFO mode requires sfo_freq_index in [1,10], got {sfo_idx}")
        idx = sfo_idx - 1
        if is_train_set:
            # Train: Only target freq
            return [tasks_sorted[idx]]
        else:
            # Test: All OTHER freqs (exclude target)
            return tasks_sorted[:idx] + tasks_sorted[idx + 1:]

    elif mode == 'high':
        # Train: Low Freqs (0-7), Test: High Freqs (8-9)
        if is_train_set:
            return tasks_sorted[:8]
        else:
            return tasks_sorted[8:]

    elif mode == 'low':
        # Train: High Freqs (2-9), Test: Low Freqs (0-1)
        if is_train_set:
            return tasks_sorted[2:]
        else:
            return tasks_sorted[:2]

    elif mode == 'mid':
        # Train: Outer (0-3, 6-9), Test: Mid (4-5)
        if is_train_set:
            return tasks_sorted[:4] + tasks_sorted[6:]
        else:
            return tasks_sorted[4:6]

    return tasks_sorted


def _process_loading(tasks, max_workers, desc,
                     downsample_count, downsample_ratio, surf_downsample_count, surf_ratio, data_type, use_surf):
    """Parallel loading helper."""
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(load_combined_vtu, t[0], t[1], t[2], t[3]): i
            for i, t in enumerate(tasks)
        }

        for fut in tqdm(as_completed(future_to_idx), total=len(tasks), desc=desc, \
                        ncols=120, mininterval=1.0):
            idx = future_to_idx[fut]
            key = tasks[idx][0]
            freq_val = tasks[idx][4]
            try:
                data = fut.result()

                min_keep = None
                if downsample_count:
                    min_keep = int(downsample_count)
                elif downsample_ratio:
                    min_keep = max(1, int(data.pos.size(0) * float(downsample_ratio)))

                data = drop_nan_rows(data, min_keep=min_keep, log_prefix=f"[{key}] ", drop_surface=True)
                data = downsample_data(data,
                                       downsample_count=downsample_count, ratio=downsample_ratio,
                                       surf_downsample_count=surf_downsample_count, surf_ratio=surf_ratio,
                                       data_type=data_type)

                data.sim_key = key
                data.freq_val = freq_val
                results[idx] = data
            except Exception as e:
                print(f"Error loading {key}: {e}")
                pass

    return [r for r in results if r is not None]


# ------------------------------------------------------------------------------
# Strategy 1: Load Data (Global Diversity - The New Default)
# ------------------------------------------------------------------------------

def load_data(
        data_root: str,
        data_type: str = 'unstructured_data',
        use_surf: bool = True,
        train_num: int = None,
        test_num: int = None,
        ood_mode: str = 'normal',  # normal, high, mid, low, sfo
        sfo_freq_index: int = None,
        extra_train_dirs: List[str] = None,
        extra_test_dir: str = None,
        downsample_count: int = None,
        surf_downsample_count: int = None,
        max_workers: int = 1,
        normalize: bool = True,
        unit_normalize: bool = True,
        seed: int = 42
):
    """
    Splits SAMPLES first, collects all valid files, then SHUFFLES files globally.
    Maximizes geometric diversity.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_tasks = []
    test_tasks = []

    # --- Case A: Explicit Folder Split ---
    if extra_train_dirs is not None and extra_test_dir is not None:
        print(f"[LoadData] Mode: Folder Split (Explicit Dirs)")

        # 1. Train Candidates
        for d in extra_train_dirs:
            groups = _scan_simulations([d], data_type, use_surf)
            for p, tasks in groups.items():
                kept = _apply_freq_filter(tasks, ood_mode, is_train_set=True, sfo_idx=sfo_freq_index)
                train_tasks.extend(kept)

        # 2. Test Candidates
        groups_test = _scan_simulations([extra_test_dir], data_type, use_surf)
        for p, tasks in groups_test.items():
            kept = _apply_freq_filter(tasks, ood_mode, is_train_set=False, sfo_idx=sfo_freq_index)
            test_tasks.extend(kept)

    # --- Case B: Single Root Split ---
    else:
        print(f"[LoadData] Mode: {str(ood_mode).upper()} | Root: {data_root}")

        all_groups = _scan_simulations([data_root], data_type, use_surf)
        all_prefixes = list(all_groups.keys())
        valid_prefixes = [p for p in all_prefixes if len(all_groups[p]) >= 10]

        # 1. Split SAMPLES (Prefixes)
        random.shuffle(valid_prefixes)
        n_total_sims = len(valid_prefixes)

        if n_total_sims == 0:
            raise ValueError("No valid simulations found.")

        total_req = (train_num or 0) + (test_num or 0)
        if total_req == 0: total_req = 1

        train_ratio = (train_num or 0) / total_req
        n_train_sims = int(n_total_sims * train_ratio)

        if n_train_sims == 0 and train_num > 0: n_train_sims = 1
        if n_train_sims == n_total_sims and test_num > 0: n_train_sims = n_total_sims - 1

        train_pool_prefixes = valid_prefixes[:n_train_sims]
        test_pool_prefixes = valid_prefixes[n_train_sims:]

        # print(f"[Split] Sim Pool: {len(train_pool_prefixes)} Train, {len(test_pool_prefixes)} Test")

        # 2. Collect ALL valid files
        for p in train_pool_prefixes:
            kept = _apply_freq_filter(all_groups[p], ood_mode, is_train_set=True, sfo_idx=sfo_freq_index)
            train_tasks.extend(kept)

        for p in test_pool_prefixes:
            kept = _apply_freq_filter(all_groups[p], ood_mode, is_train_set=False, sfo_idx=sfo_freq_index)
            test_tasks.extend(kept)

    # --- Global Shuffle & Select ---
    random.shuffle(train_tasks)
    random.shuffle(test_tasks)

    if len(train_tasks) < (train_num or 0):
        print(f"[Warn] Train pool {len(train_tasks)} < Requested {train_num}")
    if len(test_tasks) < (test_num or 0):
        print(f"[Warn] Test pool {len(test_tasks)} < Requested {test_num}")

    train_tasks = train_tasks[:train_num]
    test_tasks = test_tasks[:test_num]

    print(f"[LoadData] Loading {len(train_tasks)} Train files, {len(test_tasks)} Test files.")

    train_data = _process_loading(train_tasks, max_workers, "Loading Train",
                                  downsample_count, None, surf_downsample_count, None, data_type, use_surf)
    test_data = _process_loading(test_tasks, max_workers, "Loading Test",
                                 downsample_count, None, surf_downsample_count, None, data_type, use_surf)

    stats = {}
    if normalize:
        stats = normalize_datasets(train_data, test_data, use_surf=use_surf, unit_normalize=unit_normalize)

    return train_data, test_data, stats


# ------------------------------------------------------------------------------
# Strategy 2: Load Data By Cases (The Old Way / Backup)
# ------------------------------------------------------------------------------

def load_data_by_cases(
        data_root: str,
        data_type: str = 'unstructured_data',
        use_surf: bool = True,
        train_num: int = None,
        test_num: int = None,
        ood_mode: str = 'normal',
        sfo_freq_index: int = None,
        downsample_count: int = None,
        surf_downsample_count: int = None,
        max_workers: int = 1,
        normalize: bool = True,
        unit_normalize: bool = True,
        seed: int = 42,
        **kwargs
):
    """
    Splits SAMPLES, then takes ALL files from selected samples until count is reached.
    (This was the previous implementation).
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"[LoadDataByCases] Mode: {str(ood_mode).upper()} | Root: {data_root}")

    all_groups = _scan_simulations([data_root], data_type, use_surf)
    all_prefixes = list(all_groups.keys())
    valid_prefixes = [p for p in all_prefixes if len(all_groups[p]) >= 10]
    random.shuffle(valid_prefixes)

    # Determine files per sim
    files_per_sim_train = 10
    files_per_sim_test = 10
    if ood_mode == 'sfo':
        files_per_sim_train = 1
        files_per_sim_test = 9
    elif ood_mode in ['high', 'low', 'mid']:
        files_per_sim_train = 8
        files_per_sim_test = 2

    # Calculate needed simulations
    req_train_sims = math.ceil((train_num or 0) / files_per_sim_train)
    req_test_sims = math.ceil((test_num or 0) / files_per_sim_test)

    if req_train_sims + req_test_sims > len(valid_prefixes):
        print(
            f"[Warn] Not enough sims for By-Case loading. Need {req_train_sims + req_test_sims}, have {len(valid_prefixes)}.")
        # Handle overflow roughly
        ratio = len(valid_prefixes) / (req_train_sims + req_test_sims)
        req_train_sims = int(req_train_sims * ratio)

    train_prefixes = valid_prefixes[:req_train_sims]
    test_prefixes = valid_prefixes[req_train_sims: req_train_sims + req_test_sims]

    train_tasks = []
    for p in train_prefixes:
        kept = _apply_freq_filter(all_groups[p], ood_mode, is_train_set=True, sfo_idx=sfo_freq_index)
        train_tasks.extend(kept)

    test_tasks = []
    for p in test_prefixes:
        kept = _apply_freq_filter(all_groups[p], ood_mode, is_train_set=False, sfo_idx=sfo_freq_index)
        test_tasks.extend(kept)

    # Optional shuffle of result lists (files within cases were sorted, now we mix them)
    random.shuffle(train_tasks)
    random.shuffle(test_tasks)

    # Truncate to exact number if needed
    if train_num: train_tasks = train_tasks[:train_num]
    if test_num: test_tasks = test_tasks[:test_num]

    print(f"[LoadDataByCases] Loading {len(train_tasks)} Train, {len(test_tasks)} Test files.")

    train_data = _process_loading(train_tasks, max_workers, "Loading Train",
                                  downsample_count, None, surf_downsample_count, None, data_type, use_surf)
    test_data = _process_loading(test_tasks, max_workers, "Loading Test",
                                 downsample_count, None, surf_downsample_count, None, data_type, use_surf)

    stats = {}
    if normalize:
        stats = normalize_datasets(train_data, test_data, use_surf=use_surf, unit_normalize=unit_normalize)

    return train_data, test_data, stats


if __name__ == '__main__':
    # Usage Example
    root = "/home/omnisky/Wyk_team/sherlock/dataset/PDE/Irregular/Aletheia/typeI_double-layer"

    # Example SFO: Train on 1kHz (Sim set A), Test on others (Sim set B)
    tr, te, st = load_data(root, train_num=100, test_num=20, ood_mode='sfo', sfo_freq_index=5, use_surf=True, downsample_count=8000)
    # print(f"SFO: Train {len(tr)} (FreqIdx 1), Test {len(te)} (Others)")