import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


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
            data = data.to(device)
            optimizer.zero_grad()

            input = data.t.unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.q.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())
    elif mode == 'Q2T':
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            input = data.q.reshape(1, data.q.shape[0], -1)
            pos = data.pos.unsqueeze(0)
            targets = data.t.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())
    elif mode == 'T2T':
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            input = data.t[:, :11].reshape(1, data.t.shape[0], -1)
            pos = data.pos.unsqueeze(0)
            targets = data.t[:, 11:].reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_losses.append(loss.item())

    elif mode == 'S2Q':
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            input = data.surf.reshape(1, data.surf.shape[0], -1)
            pos = data.pos.unsqueeze(0)
            targets = data.q.reshape(-1)

            out = model(input, pos).reshape(-1)

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
            data = data.to(device)
            input = data.t.unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.q.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())


    elif mode == 'Q2T':
        for data in test_loader:
            data = data.to(device)
            input = data.q.reshape(1, data.q.shape[0], 1)
            pos = data.pos.unsqueeze(0)
            targets = data.t.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())

    elif mode == 'T2T':
        for data in test_loader:
            data = data.to(device)
            input = data.t[:, :11].unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.t[:, 11:].reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            total_losses.append(loss.item())

    elif mode == 'S2Q':
        for data in test_loader:
            data = data.to(device)
            input = data.surf.unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.q.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()

            total_losses.append(loss.item())

    return np.mean(total_losses)

def evaluate(device, model, test_dataset, mode):
    from utils.metrics import metrics
    model.to(device)
    model.eval()

    total_losses = []
    total_ssim = []
    test_loader = DataLoader(test_dataset, batch_size=1)
    criterion_func = nn.MSELoss(reduction='none')
    if mode == 'T2Q':
        for data in test_loader:
            data = data.to(device)
            input = data.t.unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.q.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())

            loss_ssim = ssim(out.detach().cpu().numpy(), targets.detach().cpu().numpy())
            total_ssim.append(loss_ssim)

        print(f'Average MSE loss: {np.mean(total_losses)}')
        print(f'Average SSIM loss: {np.mean(total_ssim)}')
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='T2Q')

        return np.mean(total_losses), np.mean(total_ssim), err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F

    elif mode == 'Q2T':
        for data in test_loader:
            data = data.to(device)
            input = data.q.reshape(1, data.q.shape[0], 1)
            pos = data.pos.unsqueeze(0)
            targets = data.t.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())

            loss_ssim = ssim(out.detach().cpu().numpy(), targets.detach().cpu().numpy())
            total_ssim.append(loss_ssim)

        print(f'Average MSE loss: {np.mean(total_losses)}')
        print(f'Average SSIM loss: {np.mean(total_ssim)}')
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='Q2T')

        return np.mean(total_losses), np.mean(total_ssim), err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F

    elif mode == 'T2T':
        for data in test_loader:
            data = data.to(device)
            input = data.t[:, :11].unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.t[:, 11:].reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())

            loss_ssim = ssim(out.detach().cpu().numpy(), targets.detach().cpu().numpy())
            total_ssim.append(loss_ssim)

        print(f'Average MSE loss: {np.mean(total_losses)}')
        print(f'Average SSIM loss: {np.mean(total_ssim)}')
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='T2T')

        return np.mean(total_losses), np.mean(total_ssim), err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F

    elif mode == 'S2Q':
        for data in test_loader:
            data = data.to(device)
            input = data.surf.unsqueeze(0)
            pos = data.pos.unsqueeze(0)
            targets = data.q.reshape(-1)

            out = model(input, pos).reshape(-1)

            loss_var = criterion_func(out, targets).mean(dim=0)
            loss = loss_var.mean()
            total_losses.append(loss.item())

            loss_ssim = ssim(out.detach().cpu().numpy(), targets.detach().cpu().numpy())
            total_ssim.append(loss_ssim)

        print(f'Average MSE loss: {np.mean(total_losses)}')
        print(f'Average SSIM loss: {np.mean(total_ssim)}')
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metrics(test_loader, model, 1.0, 1.0, 1.0, mode='S2Q')

        return np.mean(total_losses), np.mean(total_ssim), err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, test_dataset, Net, hparams, path, mode, OOD=False):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['epochs'],
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss, test_loss = 1e5, 1e5
    pbar_train = tqdm(range(hparams['epochs']), position=0)
    train_loss_list = []
    test_loss_list = []
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        train_loss = train(device, model, train_loader, optimizer, lr_scheduler, mode)
        train_loss_list.append(train_loss)
        del (train_loader)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_loss = test(device, model, test_loader, mode)
        test_loss_list.append(test_loss)
        del (test_loader)

        if epoch % 5 == 0:
            plt.plot(train_loss_list, label='Train Loss')
            plt.plot(test_loss_list, label='Test Loss', linestyle='--')
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

            plt.savefig(path + 'loss_curve.png')
            plt.clf()

        pbar_train.set_postfix(train_loss=train_loss, test_loss=test_loss)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model.state_dict(), path + os.sep + f'model_{hparams["epochs"]}_{mode}.pth')

    mse_loss, ssim_loss, err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = evaluate(device, model, test_dataset, mode=mode)

    with open(path + os.sep + f'log_{hparams["epochs"]}_{mode}.json', 'a') as f:
        json.dump(
            {
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
            }, f, indent=12, cls=NumpyEncoder
        )

    return model
