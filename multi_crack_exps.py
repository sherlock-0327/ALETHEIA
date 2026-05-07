import os
import math
import argparse
import torch
import random
import numpy as np

import train
from dataset import load_data
from model_builder import create_model, ALL_MODELS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CRACK_SUBDIR = {
    '1': 'single_layer',
    '2': 'typeI_double-layer',
    '3': 'typeII_double-layer',
    '4': 'typeIII_double-layer',
    '5': 'typeI_multi-layer',
    '6': 'typeII_multi-layer',
}


def get_device(gpu_index: int):
    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= gpu_index < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu_index}' if use_cuda else 'cpu')
    print(f"[Device] Using: {device}  (CUDA available={torch.cuda.is_available()}, count={n_gpu})")
    return device


def build_datasets(args):
    crack_ids = list(args.multi_crack)
    multi_roots = [os.path.join(args.data_root, CRACK_SUBDIR[c]) for c in crack_ids]
    n_cracks = len(crack_ids)

    if args.OOD == 'sfo':
        per_crack_train = args.sfo_train_num // n_cracks
        per_crack_test = args.sfo_test_num // n_cracks
    else:
        per_crack = args.data_num // n_cracks
        if per_crack < 2:
            raise ValueError(
                f"data_num={args.data_num} too small for {n_cracks} cracks. "
                f"Need at least {2 * n_cracks} (2 per crack).")
        per_crack_train = math.ceil(per_crack * (1 - args.test_split))
        per_crack_test = math.floor(per_crack * args.test_split)
        if per_crack_test < 1:
            raise ValueError(
                f"Test split too small: {per_crack} samples/crack × {args.test_split} = 0 test. "
                f"Increase data_num or test_split.")

    ood_mode = args.OOD if args.OOD else 'normal'

    train_graphs, test_graphs, stats = load_data(
        data_root="",  # ignored when multi_data_roots is set
        multi_data_roots=multi_roots,
        data_type=args.data_type,
        use_surf=args.use_surf,
        train_num=per_crack_train * n_cracks,
        test_num=per_crack_test * n_cracks,
        ood_mode=ood_mode,
        sfo_freq_index=args.sfo_freq,
        downsample_count=args.downsample_count,
        surf_downsample_count=args.surf_downsample_count,
        max_workers=1,
        normalize=True,
        unit_normalize=True,
        per_crack_train=per_crack_train,
        per_crack_test=per_crack_test,
    )

    print(f"[Dataset] multi_crack={args.multi_crack} | "
          f"train={len(train_graphs)}, test={len(test_graphs)}, OOD: {ood_mode}")
    return train_graphs, test_graphs


def build_save_dir(model_name: str, args) -> str:
    dtype_tag = "unstructured" if args.data_type == "unstructured_data" else "structured"

    sub = os.path.join(args.mode, f'multi-type_{args.multi_crack}', dtype_tag)

    if args.OOD:
        sub = os.path.join(sub, f'OOD_{args.OOD}')
    else:
        sub = os.path.join(sub, 'normal')

    save_dir = os.path.join('checkpoints', model_name, sub)
    if args.OOD == 'sfo' and args.sfo_freq is not None:
        save_dir = os.path.join(save_dir, str(args.sfo_freq ** 2) + 'kHz')

    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def run_one_model(model_name: str, args, device, train_graphs, test_graphs):
    if model_name == 'DeepONet' and args.out_channels != 1:
        print(f"[Skip] {model_name}: DeepONet only supports out_channels=1, current={args.out_channels}")
        return

    model = create_model(model_name, args)
    print(f"[Model] {model_name}, Task={args.mode}")

    save_dir = build_save_dir(model_name, args)

    hparams = {
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'data_num': args.data_num, 'data_type': args.data_type,
        'mode': args.mode, 'model_name': model_name,
        'in_channels': args.in_channels, 'out_channels': args.out_channels,
        'point_num': args.surf_downsample_count if args.use_surf else args.downsample_count,
        'OOD_scheme': 'multi-type',
        'multi_crack': args.multi_crack,
        'OOD': args.OOD or '',
    }

    if args.eval:
        ckpt_path = os.path.join(save_dir, f'{hparams["model_name"]}_{hparams["epochs"]}_{hparams["mode"]}.pth')
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state, dict):
                model.load_state_dict(state)
                print(f"[Eval] Loaded state_dict from {ckpt_path}")
            else:
                model = state
                print(f"[Eval] Loaded whole model object from {ckpt_path}")
        except Exception as e:
            print(f"[Warn] Could not load checkpoint: {e}")
        train.evaluate(device, model, test_graphs, mode=args.mode)
    else:
        for it in range(args.training_num):
            try:
                model = create_model(model_name, args)
                _ = train.main(device, train_graphs, test_graphs, model, hparams, save_dir, mode=args.mode)
                print(f"[Train] {model_name} | iteration {it} finished.")
            except Exception as e:
                print(f"[Error] {model_name} | iteration {it}: {e}")
                continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-crack training: train on multiple crack types simultaneously.")

    parser.add_argument('--data_root', type=str,
                        default='/dataset path/',
                        help='Root directory containing crack type subdirectories')
    parser.add_argument('--multi_crack', type=str, default='123456',
                        help='Crack type combination, e.g. "123456" or "246"')
    parser.add_argument('--data_type', type=str, default='unstructured_data',
                        choices=['unstructured_data', 'structured_data'])
    parser.add_argument('--OOD', default='', type=str,
                        choices=['', 'high', 'low', 'mid', 'sfo'],
                        help='OOD experiment type')
    parser.add_argument('--sfo_freq', default=None, type=int,
                        help='SFO frequency index (1..10)', choices=range(1, 11))
    parser.add_argument('--sfo_train_num', default=None, type=int,
                        help='SFO train sample count')
    parser.add_argument('--sfo_test_num', default=None, type=int,
                        help='SFO test sample count')
    parser.add_argument('--data_num', type=int, default=600,
                        help='Total data num (split equally among crack types)')
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--downsample_count', type=int, default=8000)
    parser.add_argument('--surf_downsample_count', type=int, default=8000)
    parser.add_argument('--use_surf', action="store_true")

    parser.add_argument('--mode', type=str, default='T2Q',
                        choices=['T2Q', 'Q2T', 'T2T', 'S2Q', 'S2Q_sp'])
    parser.add_argument('--in_channels', type=int, default=13)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--training_num', type=int, default=1)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', type=str, default='ALL',
                        help='Comma-separated model list, or "ALL".')

    return parser.parse_args()


def parse_model_list(args):
    if args.models and args.models.upper() != 'ALL':
        names = [s.strip() for s in args.models.split(',') if s.strip()]
        unknown = [n for n in names if n not in ALL_MODELS]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {ALL_MODELS}")
        return names
    return ALL_MODELS


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)

    print(args)

    crack_ids = list(args.multi_crack)
    for c in crack_ids:
        if c not in CRACK_SUBDIR:
            raise ValueError(f"Invalid crack type '{c}'. Valid: {list(CRACK_SUBDIR.keys())}")

    train_graphs, test_graphs = build_datasets(args)
    model_list = parse_model_list(args)
    print(f"[RunList] {model_list}")

    for name in model_list:
        print('=' * 50 + ' ' + name + ' ' + '=' * 50)
        run_one_model(name, args, device, train_graphs, test_graphs)
        print('=' * 50 + ' ' + name + ' ' + 'Done' + ' ' + '=' * 50)

    print('All Done')


if __name__ == '__main__':
    main()
