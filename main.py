import train
import os
import torch
import argparse

from dataset import load_all_data
from model.Transolver.Transolver_Irregular_3D import Model as Transolver
from model.GeoFNO.FNO3d import FNO3d
from model.GeoFNO.GNO import GNO
from model.GeoFNO.GeoFNO import GeoFNO
from model.mlp import MLP
from model.GeoFNO.FFNO import FNOFactorizedMesh3D as FFNO
from model.GeoFNO.FCNO import CNOFactorizedMesh3D as FCNO
from model.LNO.LNO import LNO, LNO_single, LNO_triple
from model.DeepONet.deeponet import DeepONet

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/dataset path/')
parser.add_argument('--data_type', type=str,
                    default='unstructured_data')
parser.add_argument('--downsample_count', type=int, default=8000,
                    help='downsample points count')
parser.add_argument('--surf_downsample_count', type=int, default=8000,
                    help=' surface downsample points count')
parser.add_argument('--data_num', type=int, default=100,
                    help='data num')
parser.add_argument('--test_split', type=float, default=0.2, help='train/test split')
parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--model', default='Transolver', type=str)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--in_channels', default=13, type=int)
parser.add_argument('--out_channels', default=1, type=int)


parser.add_argument("--eval", action="store_true", help="evaluate model or not")
parser.add_argument("--OOD", action="store_true", help="OOD experiment or not")
parser.add_argument('--use_surf', action="store_true", help="use surface data or not")
parser.add_argument("--mode", type=str, default='T2Q', help="different tasks, include T2Q, Q2T and T2T")

args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'epochs': args.epochs}

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

print("Loading VTU data...")
if args.OOD:
    all_graphs, stats = load_all_data(
        args.data_root,
        max_workers=1, # Read the data evenly and split it directly
        data_num=args.data_num,
        downsample_count=args.downsample_count,
        surf_downsample_count=args.surf_downsample_count,
        data_type=args.data_type,
        normalize=True,
        use_surf=args.use_surf
    )
    # print(stats)
    N = len(all_graphs)
    split = int(10 - args.test_split * 10)
    train_graphs = []
    test_graphs = []
    for i in range(0, len(all_graphs), 10):
        block = all_graphs[i:i+10]
        if len(block) < 10:
            # Ignore incomplete data blocks
            continue
        train_graphs.extend(block[:split])
        test_graphs.extend(block[split:])

    print("OOD　Experiment")
    print(f"Dataset size: total={N}, train={len(train_graphs)}, test={len(test_graphs)}")
else:
    all_graphs, stats = load_all_data(
        args.data_root,
        max_workers=8,
        data_num=args.data_num,
        downsample_count=args.downsample_count,
        surf_downsample_count=args.surf_downsample_count,
        data_type=args.data_type,
        normalize=True,
        use_surf=args.use_surf
    )
    # print(stats)
    N = len(all_graphs)
    split = int((1 - args.test_split) * N)
    train_graphs = all_graphs[:split]
    test_graphs = all_graphs[split:]
    print("All Freq　Experiment")
    print(f"Dataset size: total={N}, train={len(train_graphs)}, test={len(test_graphs)}")

if args.model == 'Transolver':
    model = Transolver(n_hidden=256, n_layers=8, space_dim=3,
                  fun_dim=args.in_channels,
                  n_head=8,
                  mlp_ratio=2, out_dim=args.out_channels,
                  slice_num=32,
                  unified_pos=0).cuda()
elif args.model == 'FNO3d':
    model = FNO3d(modes1=12, modes2=12, modes3=8, width=32, in_channels=args.in_channels, out_channels=args.out_channels)
elif args.model == 'GNO':
    model = GNO(width=32, in_channel=args.in_channels, out_channel=args.out_channels, r=1e-8)
elif args.model == 'GeoFNO':
    model = GeoFNO(modes1=12, modes2=12, modes3=8, width=32, in_channels=args.in_channels, out_channels=args.out_channels, s=20)
elif args.model == 'MLP':
    model = MLP(in_channels=args.in_channels, out_channels=args.out_channels, hidden_channels=32, n_layers=4, n_dim=1)
elif args.model == 'FFNO':
    model = FFNO(modes_x=12, modes_y=12, modes_z=8, input_dim=args.in_channels, output_dim=args.out_channels, width=32, n_layers=4,
                 share_weight=False, factor=4, n_ff_layers=2, ff_weight_norm=True, layer_norm=False)
elif args.model == 'FFNO-share':
    model = FFNO(modes_x=12, modes_y=12, modes_z=8, input_dim=args.in_channels, output_dim=args.out_channels, width=32, n_layers=4,
                 share_weight=True, factor=4, n_ff_layers=2, ff_weight_norm=True, layer_norm=False)
elif args.model == 'FCNO':
    model = FCNO(modes_x=12, modes_y=12, modes_z=8, input_dim=args.in_channels, output_dim=args.out_channels, width=32, n_layers=4,
                 share_weight=False, factor=4, n_ff_layers=2, ff_weight_norm=True, layer_norm=False)
elif args.model == 'LNO_single':
    model_attr = dict()
    model_attr['time'] = False
    model = LNO_single(x_dim=None, y1_dim=args.in_channels, y2_dim=args.out_channels, n_block=4, n_mode=256, n_dim=128,
                       n_head=8, n_layer=2, attn='Attention_Vanilla', act='GELU', model_attr=model_attr)
elif args.model == 'DeepONet': # Output can only be 1
    model = DeepONet(branch_dim=3, trunk_dim=args.in_channels, branch_depth=2, trunk_depth=3, width=32)
else:
    raise NotImplementedError("No model type found")

print(f"Model type: {args.model}, Model params: {sum(p.numel() for p in model.parameters())}, Train mode: {args.mode}")


path = f'checkpoints/{args.model}/{args.mode}/'

if args.data_type == 'unstructured_data':
    path = f'checkpoints/{args.model}/{args.mode}_uns/'
    if args.OOD:
        path = f'checkpoints/{args.model}/{args.mode}_uns_OOD/'
    print("Irregular Data　Experiment")
elif args.data_type == 'structured_data':
    path = f'checkpoints/{args.model}/{args.mode}_s/'
    if args.OOD:
        path = f'checkpoints/{args.model}/{args.mode}_s_OOD/'
    print("Regular Data　Experiment")

if not os.path.exists(path):
    os.makedirs(path)

if args.eval:
    model_path = path + f'model_{hparams["epochs"]}_{args.mode}.pth'
    try:
        state_dict = torch.load(model_path)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
            print("Loaded model from state_dict.")
        else:
            model = state_dict
            print("Loaded entire model object.")
    except FileNotFoundError:
        print("No checkpoint found at", model_path)
    except AttributeError:
        print("Loaded object has no state_dict, might be malformed.")
    except TypeError as e:
        print("Checkpoint type error:", e)
    except RuntimeError as e:
        print("Runtime error when loading model:", e)
    train.evaluate(device, model, test_graphs, mode=args.mode)
else:
    model = train.main(device, train_graphs, test_graphs, model, hparams, path, mode=args.mode, OOD=args.OOD)
