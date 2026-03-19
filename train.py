import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.ssim import point_ssim

def to_packed(data, device, fields=("pos","t","q"), squeeze_1d=True, non_blocking=True):
    data = data.to(device, non_blocking=non_blocking)

    B = int(data.num_graphs)
    sizes = (data.ptr[1:] - data.ptr[:-1]).to(torch.long)
    N = int(sizes[0])
    if not torch.all(sizes.eq(N)):
        raise RuntimeError(f"Each graph must have same #nodes, got {sizes.tolist()}")

    outs = []
    for name in fields:
        x = getattr(data, name)
        feat_shape = x.shape[1:]
        x = x.view(B, N, *feat_shape)
        if squeeze_1d and len(feat_shape)==1 and feat_shape[0]==1:
            x = x.squeeze(-1)
        outs.append(x)
    return outs[0] if len(outs)==1 else tuple(outs)

def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, mode):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    total_losses = []
    if mode == 'T2Q':
        for data in train_loader:
            pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
            optimizer.zero_grad()

            input = t
            targets = q

            out = model(input, pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())

    elif mode == 'Q2T':
        for data in train_loader:
            pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
            optimizer.zero_grad()

            input = q
            targets = t

            out = model(input, pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())

    elif mode == 'T2T':
        for data in train_loader:
            pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
            optimizer.zero_grad()

            input = t[:, :, :11]
            targets = t[:, :, 11:]

            out = model(input, pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())

    elif mode == 'S2Q':
        for data in train_loader:
            pos, t, q, surf, surf_pos = to_packed(data, device=device, fields=('pos', 't', 'q', 'surf', 'surf_pos'), squeeze_1d=False, non_blocking=True)
            optimizer.zero_grad()

            input = surf
            targets = q

            out = model(input, pos, surf_pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())

    return np.mean(total_losses)


@torch.no_grad()
def test(device, model, test_loader, mode):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    total_losses = []
    total_ssim = []
    if mode == 'T2Q':
        for data in test_loader:
            pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
            input = t
            targets = q

            out = model(input, pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())
            ssim_loss = point_ssim(out, targets, pos)
            total_ssim.append(ssim_loss.item())


    elif mode == 'Q2T':
        for data in test_loader:
            pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
            input = q
            targets = t

            out = model(input, pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())
            ssim_loss = point_ssim(out, targets, pos)
            total_ssim.append(ssim_loss.item())

    elif mode == 'T2T':
        for data in test_loader:
            pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
            input = t[:, :, :11]
            targets = t[:, :, 11:]

            out = model(input, pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())
            ssim_loss = point_ssim(out, targets, pos)
            total_ssim.append(ssim_loss.item())

    elif mode == 'S2Q':
        for data in test_loader:
            pos, t, q, surf, surf_pos = to_packed(data, device=device, fields=('pos', 't', 'q','surf', 'surf_pos'), squeeze_1d=False, non_blocking=True)
            input = surf
            targets = q

            out = model(input, pos, surf_pos)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())
            ssim_loss = point_ssim(out, targets, pos)
            total_ssim.append(ssim_loss.item())

    return np.mean(total_losses), np.mean(total_ssim)

def evaluate(device, model, test_dataset, mode):
    from utils.metrics import metrics
    model.to(device)
    model.eval()

    data_range = None
    total_losses = []
    total_ssim = []
    test_loader = DataLoader(test_dataset, batch_size=1)
    criterion_func = nn.MSELoss(reduction='none')

    with torch.inference_mode():
        if mode == 'T2Q':
            for data in test_loader:
                pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False,non_blocking=True)
                input = t
                targets = q

                out = model(input, pos)

                loss_var = criterion_func(out, targets).mean(dim=0)
                loss = loss_var.mean()
                total_losses.append(loss.item())

                loss_ssim = point_ssim(out, targets, pos)
                total_ssim.append(loss_ssim.item())

            print(f'Average MSE loss: {np.mean(total_losses)}')
            print(f'Average SSIM loss: {np.mean(total_ssim)}')
            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='T2Q')


        elif mode == 'Q2T':
            for data in test_loader:
                pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False,non_blocking=True)
                input = q
                targets = t

                out = model(input, pos)

                loss_var = criterion_func(out, targets).mean(dim=0)
                loss = loss_var.mean()
                total_losses.append(loss.item())

                loss_ssim = point_ssim(out, targets, pos)
                total_ssim.append(loss_ssim.item())

            print(f'Average MSE loss: {np.mean(total_losses)}')
            print(f'Average SSIM loss: {np.mean(total_ssim)}')
            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='Q2T')


        elif mode == 'T2T':
            for data in test_loader:
                pos, t, q = to_packed(data, device=device, fields=('pos', 't', 'q'), squeeze_1d=False, non_blocking=True)
                input = t[:, :, :11]
                targets = t[:, :, 11:]

                out = model(input, pos)

                loss_var = criterion_func(out, targets).mean(dim=0)
                loss = loss_var.mean()
                total_losses.append(loss.item())

                loss_ssim = point_ssim(out, targets, pos)
                total_ssim.append(loss_ssim.item())

            print(f'Average MSE loss: {np.mean(total_losses)}')
            print(f'Average SSIM loss: {np.mean(total_ssim)}')
            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='T2T')


        elif mode == 'S2Q':
            for data in test_loader:
                pos, t, q, surf, surf_pos = to_packed(data, device=device, fields=('pos', 't', 'q', 'surf', 'surf_pos'), squeeze_1d=False, non_blocking=True)
                input = surf
                targets = q

                out = model(input, pos, surf_pos)

                loss_var = criterion_func(out, targets).mean(dim=0)
                loss = loss_var.mean()
                total_losses.append(loss.item())

                loss_ssim = point_ssim(out, targets, pos)
                total_ssim.append(loss_ssim.item())

            print(f'Average MSE loss: {np.mean(total_losses)}')
            print(f'Average SSIM loss: {np.mean(total_ssim)}')
            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='S2Q')


        return np.mean(total_losses), np.mean(total_ssim), err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, test_dataset, Net, hparams, path, mode):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['epochs'],
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss, test_loss, test_ssim = 1e5, 1e5, 1e5
    pbar_train = tqdm(range(hparams['epochs']), position=0, ncols=120, mininterval=1.0)
    train_loss_list = []
    test_loss_list = []

    sampler = RandomSampler(train_dataset, replacement=False)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=hparams['batch_size'],
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)

    for epoch in pbar_train:
        sampler.generator = torch.Generator().manual_seed(epoch)
        train_loss = train(device, model, train_loader, optimizer, lr_scheduler, mode)
        train_loss_list.append(train_loss)

        interval = max(1, hparams['epochs'] // 10)
        if (epoch + 1) % interval == 0:
            test_loss, test_ssim = test(device, model, test_loader, mode)
            test_loss_list.append(test_loss)
            # del (test_loader)

            plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
            plt.plot([(i + 1) * interval for i in range(len(test_loss_list))],
                     test_loss_list, label='Test Loss', linestyle='--', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()

            # Dynamic switching of axis types
            threshold = 10  # Adjust this threshold according to the actual
            current_max_loss = max(max(train_loss_list), max(test_loss_list))

            if current_max_loss > threshold:
                plt.yscale('log')
                plt.grid(alpha=0.3, which='both')  #  Displaying finer grids in logarithmic coordinates
            else:
                plt.yscale('linear')
                plt.grid(alpha=0.3)

            plt.savefig(os.path.join(path, 'loss_curve.png'))
            plt.clf()

        pbar_train.set_postfix(loss=train_loss, mse=test_loss, ssim=test_ssim)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    # print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model.state_dict(), path + os.sep + f'{hparams["model_name"]}_{hparams["epochs"]}_{hparams["mode"]}.pth')

    mse_loss, ssim_loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = evaluate(device, model, test_dataset, mode=mode)

    log_path = os.path.join(path, f'log_{hparams["model_name"]}_{hparams["epochs"]}_{mode}.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    new_entry = {
        'nb_parameters': params_model,
        'time_elapsed': time_elapsed,
        'hparams': hparams,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'ssim_loss': ssim_loss,
        'RMSE': err_RMSE,
        'normalized RMSE': err_nRMSE,
        'RMSE of conserved variables': err_CSV,
        'Maximum value of rms error': err_Max,
        'RMSE at boundaries': err_BD,
        'RMSE in Fourier space': err_F,
    }
    data.append(new_entry)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=12, cls=NumpyEncoder)

    return model
