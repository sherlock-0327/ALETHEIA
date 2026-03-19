# run_all_models.py
import os
import argparse
import torch
import random
import numpy as np

import train
from dataset import load_data

from model_builder import create_model, ALL_MODELS, MODEL_REGISTRY


def get_device(gpu_index: int):
    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= gpu_index < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu_index}' if use_cuda else 'cpu')
    print(f"[Device] Using: {device}  (CUDA available={torch.cuda.is_available()}, count={n_gpu})")
    return device


CRACK_TYPE = {
    '1': 'single',
    '2': 'I-double',
    '3': 'II-double',
    '4': 'III-double',
    '5': 'I-multi',
    '6': 'II-multi',
}


def build_datasets(args):
    train_dirs = [p.strip() for p in args.ood_train_dirs.split(',') if p.strip()]
    num_train_folders = len(train_dirs)
    if len(train_dirs) != 5:
        print(f"[Warn] Expected 5 train folders, got {len(train_dirs)}. Will proceed anyway.")
    test_dir = args.ood_test_dir
    if not test_dir:
        raise ValueError("Please set --ood_test_dir to the validation folder")

    total_train_num = args.per_folder_train * num_train_folders
    total_test_num = args.per_folder_test

    train_graphs, test_graphs, stats = load_data(
        data_root="", # ignore
        extra_train_dirs=train_dirs,
        extra_test_dir=test_dir,
        train_num=total_train_num,
        test_num=total_test_num,
        ood_mode='normal', # ignore
        data_type=args.data_type,
        use_surf=args.use_surf,
        downsample_count=args.downsample_count,
        surf_downsample_count=args.surf_downsample_count,
        normalize=True,
        unit_normalize=True,
        max_workers=1,
    )
    print(f"[Dataset OOD-fsplit] "
          f"train_folders={num_train_folders} -> total_train={len(train_graphs)}, "
          f"test_folder=1 -> total_test={len(test_graphs)} ")

    if args.data_type == 'unstructured_data':
        print("[DataType] Irregular (unstructured)")
    elif args.data_type == 'structured_data':
        print("[DataType] Regular (structured)")

    return train_graphs, test_graphs


def build_save_dir_and_ckpt_name(model_name: str, args) -> (str, str):
    dtype_tag = "unstructured" if args.data_type == "unstructured_data" else "structured"

    scheme_tag = "OOD-fsplit"

    test_crack_type = args.test_crack_type
    crack_gather = '123456'.replace(test_crack_type, "")

    sub = os.path.join(f'{args.mode}', f'{dtype_tag}', f'{scheme_tag}', f"train_{crack_gather}_test_{test_crack_type}")

    save_dir = os.path.join('checkpoints', model_name, sub)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def run_one_model(model_name: str, args, device, train_graphs, test_graphs):
    if model_name == 'DeepONet' and args.out_channels != 1:
        print(f"[Skip] {model_name}: DeepONet only supports out_channels=1, current={args.out_channels}")
        return

    model = create_model(model_name, args)
    print(f"[Model] {model_name}, Task={args.mode}")

    save_dir = build_save_dir_and_ckpt_name(model_name, args)


    hparams = {
        'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'epochs': args.epochs,
        'in_channels': args.in_channels, 'out_channels': args.out_channels,
        'point_num': args.surf_downsample_count if args.use_surf else args.downsample_count,
        'model_name': model_name, 'mode': args.mode,
        'OOD_scheme': 'folder-split',
        'OOD_train_dirs': args.ood_train_dirs,
        'OOD_test_dir': args.ood_test_dir,
        'per_folder_train': args.per_folder_train,
        'per_folder_test': args.per_folder_test,
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
    parser = argparse.ArgumentParser(description="Run all models on a folder-split OOD task (5 train folders + 1 test folder).")

    parser.add_argument('--ood_train_dirs', type=str, required=True,
                        help='comma-separated 5 train folders, each loads per_folder_train samples')
    parser.add_argument('--ood_test_dir', type=str, required=True,
                        help='one test folder, loads per_folder_test samples')
    parser.add_argument('--per_folder_train', type=int, default=100)
    parser.add_argument('--per_folder_test', type=int, default=100)
    parser.add_argument('--data_type', type=str, default='unstructured_data', choices=['unstructured_data', 'structured_data'])
    parser.add_argument('--test_crack_type', type=str, default='6', choices=['1', '2', '3', '4', '5', '6'])
    parser.add_argument('--downsample_count', type=int, default=8000)
    parser.add_argument('--surf_downsample_count', type=int, default=8000)
    parser.add_argument('--use_surf', action="store_true", help='use surface data or not')

    parser.add_argument('--mode', type=str, default='T2Q', choices=['T2Q','Q2T','T2T','S2Q','S2Q_sp'])
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
                        help='comma-separated model list, e.g. "Transolver,Transolver_plus,FNO3d". '
                             'Use "ALL" to run every registered model.')

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
    device = get_device(args.gpu)

    train_graphs, test_graphs = build_datasets(args)
    model_list = parse_model_list(args)
    print(f"[RunList] {model_list}")

    for name in model_list:
        print('='*50 + ' ' + name + ' ' + '='*50)
        run_one_model(name, args, device, train_graphs, test_graphs)
        print('='*50 + ' ' + name + ' ' + 'Done' + ' ' + '='*50)

    print('All Done')


if __name__ == '__main__':
    main()
