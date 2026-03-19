import torch
import torch.nn as nn

from model.Transolver.Transolver_Irregular_3D import Model as Transolver
from model.Transolver.Transolver_plus import Model as Transolver_plus
from model.Transolver.benchmark.Transolver_Structured_Mesh_3D import Model as Transolver_Structured_Mesh_3D
from model.GeoFNO.FNO3d import FNO3d
from model.GeoFNO.GFNO import GFNO3d as GFNO
from model.GeoFNO.GNO import GNO
from model.GeoFNO.GeoFNO import GeoFNO
from model.mlp import MLP
from model.GeoFNO.FFNO import FNOFactorizedMesh3D as FFNO
from model.GeoFNO.FCNO import CNOFactorizedMesh3D as FCNO
from model.LNO.LNO import LNO, LNO_single
from model.DeepONet.deeponet import DeepONet
from model.FEM import FEMHeatSolver
from model.TestModel.Test_Model import Model as TestModel
from model.LaMO.LaMO_Irregular_Mesh_unshared import Model as LaMO

# ==========================================
# Builder Functions
# ==========================================

def build_LaMO(args):
    return LaMO(
        n_hidden=256, n_layers=8, space_dim=3,
        fun_dim=args.in_channels, n_head=8, mlp_ratio=2,
        out_dim=args.out_channels, slice_num=32, unified_pos=0,
        dropout=0.1
    )

def build_Transolver(args):
    return Transolver(
        n_hidden=256, n_layers=8, space_dim=3,
        fun_dim=args.in_channels, n_head=8, mlp_ratio=2,
        out_dim=args.out_channels, slice_num=32, unified_pos=0,
        dropout=0.1
    )


def build_Transolver_R(args):
    return Transolver_Structured_Mesh_3D(
        n_hidden=256, n_layers=8, space_dim=3,
        fun_dim=args.in_channels, n_head=8, mlp_ratio=2,
        out_dim=args.out_channels, slice_num=32,
        H=20, W=20, D=20, unified_pos=0,
        dropout=0.1
    )


def build_Transolver_plus(args):
    return Transolver_plus(
        n_hidden=256, n_layers=8, space_dim=3,
        fun_dim=args.in_channels, n_head=8, mlp_ratio=2,
        out_dim=args.out_channels, slice_num=32, unified_pos=0,
        dropout=0.1
    )


def build_FNO3d(args):
    return FNO3d(
        modes1=12, modes2=12, modes3=8, width=32,
        in_channels=args.in_channels, out_channels=args.out_channels, H=20, W=20, D=20
    )


def build_GFNO(args):
    return GFNO(
        modes1=12, modes2=12, modes3=8, width=32,
        in_channels=args.in_channels, out_channels=args.out_channels, H=20, W=20, D=20
    )


def build_GNO(args):
    return GNO(width=32, in_channel=args.in_channels, out_channel=args.out_channels, r=1e-8)


def build_GeoFNO(args):
    return GeoFNO(modes1=12, modes2=12, modes3=8, width=32,
                  in_channels=args.in_channels, out_channels=args.out_channels, s=20)


def build_MLP(args):
    return MLP(in_channels=args.in_channels, out_channels=args.out_channels,
               hidden_channels=32, n_layers=4, n_dim=1, dropout=0.1)


def build_FFNO(args):
    return FFNO(modes_x=12, modes_y=12, modes_z=8,
                input_dim=args.in_channels, output_dim=args.out_channels,
                width=32, n_layers=4, share_weight=False, factor=4,
                n_ff_layers=2, ff_weight_norm=True, layer_norm=False, H=20, W=20, D=20)


def build_FFNO_share(args):
    return FFNO(modes_x=12, modes_y=12, modes_z=8,
                input_dim=args.in_channels, output_dim=args.out_channels,
                width=32, n_layers=4, share_weight=True, factor=4,
                n_ff_layers=2, ff_weight_norm=True, layer_norm=False, H=20, W=20, D=20)


def build_FCNO(args):
    return FCNO(modes_x=12, modes_y=12, modes_z=8,
                input_dim=args.in_channels, output_dim=args.out_channels,
                width=32, n_layers=4, share_weight=False, factor=4,
                n_ff_layers=2, ff_weight_norm=True, layer_norm=False, H=20, W=20, D=20)


def build_LNO(args):
    model_attr = dict(time=False)
    return LNO(x_dim=3, y1_dim=args.in_channels, y2_dim=args.out_channels,
               n_block=4, n_mode=256, n_dim=128,
               n_head=8, n_layer=2, attn='Attention_Vanilla', act='GELU', model_attr=model_attr)


def build_LNO_single(args):
    model_attr = dict(time=False)
    return LNO_single(x_dim=None, y1_dim=args.in_channels, y2_dim=args.out_channels,
                      n_block=4, n_mode=256, n_dim=128,
                      n_head=8, n_layer=2, attn='Attention_Vanilla', act='GELU', model_attr=model_attr)


def build_DeepONet(args):  # Only support out_channels == 1
    return DeepONet(branch_dim=3, trunk_dim=args.in_channels, branch_depth=2, trunk_depth=3, width=32)


def build_FEM(args):
    return FEMHeatSolver(mode=args.mode, num_time_steps=args.out_channels, num_points=args.downsample_count)


def build_TestModel(args):
    return TestModel(fun_dim=args.in_channels, out_dim=args.out_channels)

# ==========================================
# Registry
# ==========================================

MODEL_REGISTRY = {
    'Transolver': build_Transolver,
    'Transolver_R': build_Transolver_R,
    'Transolver_plus': build_Transolver_plus,
    'FNO3d': build_FNO3d,
    'GFNO': build_GFNO,
    'GNO': build_GNO,
    'GeoFNO': build_GeoFNO,
    'MLP': build_MLP,
    'FFNO': build_FFNO,
    'FFNO-share': build_FFNO_share,
    'FCNO': build_FCNO,
    'LNO': build_LNO,
    'LNO_single': build_LNO_single,
    'DeepONet': build_DeepONet,
    'FEM': build_FEM,
    'TestModel': build_TestModel,
    'LaMO': build_LaMO,
}

ALL_MODELS = list(MODEL_REGISTRY.keys())

def create_model(model_name: str, args):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {ALL_MODELS}")
    return MODEL_REGISTRY[model_name](args)