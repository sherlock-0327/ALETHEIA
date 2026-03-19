import os
import argparse
import torch
import random
import numpy as np

import train
from dataset import load_data

from model_builder import create_model, ALL_MODELS, MODEL_REGISTRY


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CRACK_TYPE = {
    '1': 'single',
    '2': 'I-double',
    '3': 'II-double',
    '4': 'III-double',
    '5': 'I-multi',
    '6': 'II-multi',
}


def build_datasets(args):
    print("Loading VTU data once ...")
    if args.OOD == 'sfo':
        if getattr(args, 'sfo_train_num', None) is None:
            # Fallback if not provided (though your args check usually enforces it)
            calc_train_num = int(args.data_num * (1 - args.test_split))
            calc_test_num = int(args.data_num * args.test_split)
        else:
            calc_train_num = args.sfo_train_num
            calc_test_num = args.sfo_test_num
    else:
        calc_train_num = int(args.data_num * (1 - args.test_split))
        calc_test_num = int(args.data_num * args.test_split)

    ood_mode = args.OOD if args.OOD else 'normal'

    train_graphs, test_graphs, stats = load_data(
        data_root=args.data_root,
        data_type=args.data_type,
        use_surf=args.use_surf,
        train_num=calc_train_num,
        test_num=calc_test_num,
        ood_mode=ood_mode,
        sfo_freq_index=args.sfo_freq,
        downsample_count=args.downsample_count,
        surf_downsample_count=args.surf_downsample_count,
        max_workers=1,
        normalize=True,
        unit_normalize=True
    )

    print(f"Dataset size: train={len(train_graphs)}, test={len(test_graphs)}, OOD type: {ood_mode}")
    return train_graphs, test_graphs

def get_device(gpu_index: int):
    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= gpu_index < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu_index}' if use_cuda else 'cpu')
    print(f"[Device] Using: {device}  (CUDA available={torch.cuda.is_available()}, count={n_gpu})")
    return device

def run_one_model(model_name: str, args, device, train_graphs, test_graphs):
    if model_name == 'DeepONet' and args.out_channels != 1:
        print(f"[Skip] {model_name}: DeepONet only support out_channels=1, Current parameter={args.out_channels}")
        return

    model = create_model(model_name, args)
    print(f"[Model] {model_name}, Task (mode) = {args.mode}")

    if args.data_type == 'unstructured_data':
        sub = os.path.join(f'{args.mode}', f'{CRACK_TYPE[args.crack_type]}', 'unstructured',
                           (f'OOD_{args.OOD}' if args.OOD else 'normal'))
        print("[DataType] Irregular (unstructured)" + (f' OOD Type: {args.OOD}' if args.OOD else 'normal'))
    elif args.data_type == 'structured_data':
        sub = os.path.join(f'{args.mode}', f'{CRACK_TYPE[args.crack_type]}', 'structured',
                           (f'OOD_{args.OOD}' if args.OOD else 'normal'))
        print("[DataType] Regular (structured)" + (f' OOD Type: {args.OOD}' if args.OOD else 'normal'))
    else:
        sub = f'{args.mode}'


    save_dir = os.path.join('checkpoints', model_name, sub)
    if args.OOD == 'sfo':
        save_dir = os.path.join(save_dir, str(args.sfo_freq**2) + 'kHz')
    os.makedirs(save_dir, exist_ok=True)

    hparams = {'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'epochs': args.epochs,
               'data_num': args.data_num, 'data_type':args.data_type, 'crack_type': CRACK_TYPE[args.crack_type],
               'mode':args.mode, 'model_name': model_name, 'in_channels': args.in_channels, 'out_channels': args.out_channels,
               'point_num': args.downsample_count if not args.use_surf else args.surf_downsample_count,
               'OOD': args.OOD}


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
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='/dataset path/')
    parser.add_argument('--data_type', type=str, default='unstructured_data', choices=['unstructured_data', 'structured_data'])
    parser.add_argument('--crack_type', type=str, default='2', choices=['1', '2', '3', '4', '5', '6'])
    parser.add_argument('--downsample_count', type=int, default=8000, help='downsample points count')
    parser.add_argument('--surf_downsample_count', type=int, default=8000, help='surface downsample points count')
    parser.add_argument('--data_num', type=int, default=100, help='data num')
    parser.add_argument('--test_split', type=float, default=0.2, help='train/test split')
    parser.add_argument('--use_surf', action="store_true", help='use surface data or not')
    parser.add_argument("--OOD", default=None, type=str, help="OOD experiment",
                        choices=[ '', 'high', 'low', 'mid', 'sfo'])

    parser.add_argument("--sfo_freq", default=None, type=int, help="when --OOD sfo: fixed training frequency index (1..10 maps to 1,4,...,100 kHz)",
                        choices=range(1, 11))
    parser.add_argument("--sfo_train_num", default=None, type=int, help="when --OOD sfo: choose train num")
    parser.add_argument("--sfo_test_num", default=None, type=int, help="when --OOD sfo: choose test num")

    parser.add_argument('--mode', type=str, default='T2Q', choices=['T2Q', 'Q2T', 'T2T', 'S2Q', 'S2Q_sp'], help='task type')
    parser.add_argument('--in_channels', type=int, default=13)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--training_num', type=int, default=1, help='repeat times for each model')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="evaluate mode")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--models', type=str, default='ALL',
                        help='comma-separated model list, e.g., "Transolver,Transolver_plus,FNO3d". '
                             'Use "ALL" to run every registered model.')

    parser.add_argument('--model', default=None, type=str, help='(deprecated) single model name')

    args = parser.parse_args()
    return args


def parse_model_list(args):
    if args.models and args.models.upper() != 'ALL':
        names = [s.strip() for s in args.models.split(',') if s.strip()]
        unknown = [n for n in names if n not in ALL_MODELS]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {ALL_MODELS}")
        return names

    if args.model:
        if args.model not in ALL_MODELS:
            raise ValueError(f"Unknown model: {args.model}. Available: {ALL_MODELS}")
        return [args.model]

    return ALL_MODELS


def main():
    args = parse_args()
    print(args)

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
