import train
import os
import torch
import argparse

from dataset import load_data

from model_builder import create_model, ALL_MODELS, MODEL_REGISTRY

CRACK_TYPE = {
    '1': 'single',
    '2': 'I-double',
    '3': 'II-double',
    '4': 'III-double',
    '5': 'I-multi',
    '6': 'II-multi',
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/dataset path/')
parser.add_argument('--data_type', type=str,
                    default='unstructured_data')
parser.add_argument('--crack_type', type=str, default='2', choices=['1', '2', '3', '4', '5', '6'])
parser.add_argument('--downsample_count', type=int, default=8000,
                    help='downsample points count')
parser.add_argument('--surf_downsample_count', type=int, default=8000,
                    help=' surface downsample points count')
parser.add_argument('--data_num', type=int, default=100,
                    help='data num')
parser.add_argument('--test_split', type=float, default=0.2, help='train/test split')
parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--model', default='Transolver', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--in_channels', default=13, type=int)
parser.add_argument('--out_channels', default=1, type=int)


parser.add_argument("--OOD", default=None, type=str, help="OOD experiment", choices=[ '', 'high', 'low', 'mid', 'sfo'])
parser.add_argument("--sfo_freq", default=None, type=int,
                    help="when --OOD sfo: fixed training frequency index (1..10 maps to 1,4,...,100 kHz)",
                    choices=range(1, 11))
parser.add_argument("--sfo_train_num", default=None, type=int, help="when --OOD sfo: choose train num")
parser.add_argument("--sfo_test_num", default=None, type=int, help="when --OOD sfo: choose test num")


parser.add_argument("--eval", action="store_true", help="evaluate model or not")
parser.add_argument('--use_surf', action="store_true", help="use surface data or not")
parser.add_argument("--mode", type=str, default='T2Q', choices=['T2Q', 'Q2T', 'T2T', 'S2Q'], help='task type')
parser.add_argument('--training_num', type=int, default=1, help="The number of training sessions")

args = parser.parse_args()
print(args)
n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

print("Loading VTU data...")

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


model = create_model(args.model, args)
print(f'Model: {args.model}, Training Type:{args.mode}')

if args.data_type == 'unstructured_data':
    sub = os.path.join(f'{args.mode}', f'{CRACK_TYPE[args.crack_type]}_uns', 'unstructured',
                       (f'OOD_{args.OOD}' if args.OOD else 'normal'))
    print("[DataType] Irregular (unstructured)" + (f' OOD Type: {args.OOD}' if args.OOD else 'normal'))
elif args.data_type == 'structured_data':
    sub = os.path.join(f'{args.mode}', f'{CRACK_TYPE[args.crack_type]}', 'structured',
                       (f'OOD_{args.OOD}' if args.OOD else 'normal'))
    print("[DataType] Regular (structured)" + (f' OOD Type: {args.OOD}' if args.OOD else 'normal'))
else:
    sub = f'{args.mode}'

save_dir = os.path.join('checkpoints', args.model, sub)
if args.OOD == 'sfo':
    save_dir = os.path.join(save_dir, str(args.sfo_freq ** 2) + 'kHz')
os.makedirs(save_dir, exist_ok=True)

hparams = {'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'epochs': args.epochs,
           'data_num': args.data_num, 'data_type': args.data_type, 'crack_type': CRACK_TYPE[args.crack_type],
           'mode': args.mode, 'model_name': args.model, 'in_channels': args.in_channels, 'out_channels': args.out_channels,
           'point_num': args.downsample_count if not args.use_surf else args.surf_downsample_count,
           'OOD': args.OOD}
if args.OOD == 'sfo':
    save_dir = os.path.join(save_dir, str(args.sfo_freq ** 2) + 'kHz')

print(f'[MODEL PARAMETERS]: {sum(p.numel() for p in model.parameters())}]')

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
    except  Exception as e:
        print(f"[Warn] Could not load checkpoint: {e}")
    train.evaluate(device, model, test_graphs, mode=args.mode)
else:
    for i in range(args.training_num):
        try:
            model = create_model(args.model, args)
            _ = train.main(device, train_graphs, test_graphs, model, hparams, save_dir, mode=args.mode)
            print(f"[Train] {args.model} | iteration {i} finished.")
        except Exception as e:
            print(f"[Error] {args.model} | iteration {i}: {e}")
            continue