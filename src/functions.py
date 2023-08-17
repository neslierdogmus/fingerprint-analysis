import torch
from torch.utils.data import DataLoader

import numpy as np

from foe_ae import FOE_AE
from foe_mlp import FOE_MLP
from foe_cnn import FOE_CNN
from foe_loss_history import FOELossHistory

from time import time


def timing(f):
    def timed(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()

        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))

        return result

    return timed


def create_dataloaders(fpd_gd, fpd_bd, fpd_sn, fold_id,
                       num_workers, use_gpu, args):
    tset_gd, vset_gd = fpd_gd.get_patch_datasets(fold_id, args['patch_size']*2,
                                                 args['patch_size'],
                                                 args['num_classes'])
    tset_bd, vset_bd = fpd_bd.get_patch_datasets(fold_id, args['patch_size']*2,
                                                 args['patch_size'],
                                                 args['num_classes'])
    # tset_sn, _ = fpd_sn.get_patch_datasets(fold_id, args['patch_size']*2,
    #                                        args['patch_size'],
    #                                        args['num_classes'])
    tset = tset_gd.merge(tset_bd)
    # tset = tset.merge(tset_sn)
    vset = vset_gd.merge(vset_bd)

    if args['train_with_bad']:
        ae_tset = tset
    else:
        ae_tset = tset_gd

    ae_tset.set_hflip(args['hflip'])
    tset.set_hflip(args['hflip'])
    ae_tset.set_rotate(args['rotate'])
    tset.set_rotate(args['rotate'])

    ae_train_loader = DataLoader(ae_tset, batch_size=args['batch_size'],
                                 num_workers=num_workers,
                                 shuffle=True, pin_memory=use_gpu)
    train_loader_gd = DataLoader(tset_gd, batch_size=args['batch_size'],
                                 num_workers=num_workers,
                                 shuffle=True, pin_memory=use_gpu)
    train_loader_bd = DataLoader(tset_bd, batch_size=args['batch_size'],
                                 num_workers=num_workers,
                                 shuffle=True, pin_memory=use_gpu)
    train_loader = DataLoader(tset, batch_size=args['batch_size'],
                              num_workers=num_workers,
                              shuffle=True, pin_memory=use_gpu)
    val_loader_gd = DataLoader(vset_gd, batch_size=args['batch_size'],
                               num_workers=num_workers,
                               shuffle=True, pin_memory=use_gpu)
    val_loader_bd = DataLoader(vset_bd, batch_size=args['batch_size'],
                               num_workers=num_workers,
                               shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(vset, batch_size=args['batch_size'],
                            num_workers=num_workers,
                            shuffle=True, pin_memory=use_gpu)

    print("""Training and validating with:
        Autoencoder training set size: {}
        Orientation estimation training set size: {}
        Validation set size (good): {}
        Validation set size (bad): {}""".format(len(ae_train_loader.dataset),
                                                len(train_loader.dataset),
                                                len(val_loader_gd.dataset),
                                                len(val_loader_bd.dataset)))

    return (ae_train_loader, train_loader_gd, train_loader_bd,
            train_loader, val_loader_gd, val_loader_bd, val_loader)


def load_model(model_path, model_type, device, verbose=True):
    if model_type == 'ae':
        FOE_MODEL = FOE_AE
    elif model_type == 'mlp':
        FOE_MODEL = FOE_MLP
    elif model_type == 'cnn':
        FOE_MODEL = FOE_CNN
    chk_dict = torch.load(model_path)
    mstate = chk_dict['mstate']
    inp_dim = chk_dict['inp-dim']
    out_dim = chk_dict['out-dim']

    model = FOE_MODEL(model_path, inp_dim, out_dim, device)
    model.load_state_dict(mstate)

    if verbose:
        print('Loaded model from {}.'.format(model_path))
    return model


def save_model(model, verbose=True):
    chk_dict = {'mstate': model.state_dict(),
                'inp-dim': model.inp_dim,
                'out-dim': model.out_dim}
    torch.save(chk_dict, model.path)
    if verbose:
        print('Saved model to {}'.format(model.path))


def init_model(model_type, models_dir, file_id, fold_id, device, args):
    if model_type == 'ae':
        FOE_MODEL = FOE_AE
        input = args['patch_size']
        output = args['encoded_space_dim']
    elif model_type == 'mlp':
        FOE_MODEL = FOE_MLP
        input = args['encoded_space_dim']
        output = args['num_classes']
    elif model_type == 'cnn':
        FOE_MODEL = FOE_CNN
        input = args['patch_size']
        output = args['num_classes']

    model_name = 'foe_{}_{}_f{}.pt'.format(model_type, file_id, fold_id)
    metrics_name = 'foe_{}_{}_f{}.mt'.format(model_type, file_id, fold_id)
    model_path = models_dir.joinpath(model_name)
    metrics_path = models_dir.joinpath(metrics_name)
    if model_path.exists() and metrics_path.exists():
        model = load_model(model_path, model_type, device)
        metrics = FOELossHistory.load(metrics_path)
    else:
        model = FOE_MODEL(model_path, input, output, device)
        metrics = FOELossHistory(metrics_path)
    return model, metrics


def train_epoch(model, dataloader, loss_fn, optimizer,
                encoder=None, is_ae=False, results=None):
    model.train()
    total_loss = 0.0
    for xi, yi, gt_in_radians, index in dataloader:
        x = y = xi.to(model.device)
        if not is_ae:
            y = yi.to(model.device)
        optimizer.zero_grad()
        if encoder is not None:
            with torch.no_grad():
                encoder.eval()
                x = encoder(x)
                x = x.unsqueeze(1)
        y_out = model(x)
        loss = loss_fn(y_out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if results is not None:
            for idx, gt, yo in zip(index, gt_in_radians, y_out.cpu()):
                patch = dataloader.dataset.patches[idx.item()]
                if patch.fp_type != 2:
                    results.append(patch, False, gt.item(),
                                   yo.detach().numpy())
    return total_loss / len(dataloader.dataset)


def val_epoch(model, dataloader, loss_fn,
              is_ae=False, encoder=None, results=None, is_tr=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for xi, yi, gt_in_radians, index in dataloader:
            x = y = xi.to(model.device)
            if not is_ae:
                y = yi.to(model.device)
            if encoder is not None:
                encoder.eval()
                x = encoder(x)
                x = x.unsqueeze(1)
            y_out = model(x)
            loss = loss_fn(y_out, y)
            total_loss += loss.item()
            if results is not None:
                for idx, gt, yo in zip(index, gt_in_radians, y_out.cpu()):
                    patch = dataloader.dataset.patches[idx.item()]
                    results.append(patch, not is_tr, gt.item(),
                                   yo.detach().numpy())

    return total_loss / len(dataloader.dataset)


def eval_model(model, tra_ldr_gd, tra_ldr_bd, val_ldr_gd, val_ldr_bd, loss_fn,
               encoder, estimator, results, metrics, fold_id):
    tra_loss_gd = val_epoch(model, tra_ldr_gd, loss_fn, encoder=encoder,
                            results=results, is_tr=True)
    tra_loss_bd = val_epoch(model, tra_ldr_bd, loss_fn, encoder=encoder,
                            results=results, is_tr=True)
    val_loss_gd = val_epoch(model, val_ldr_gd, loss_fn,
                            encoder=encoder, results=results)
    val_loss_bd = val_epoch(model, val_ldr_bd, loss_fn,
                            encoder=encoder, results=results)

    num_tra_gd = len(tra_ldr_gd.dataset)
    num_tra_bd = len(tra_ldr_bd.dataset)
    num_val_gd = len(val_ldr_gd.dataset)
    num_val_bd = len(val_ldr_bd.dataset)

    tra_loss = ((num_tra_gd * tra_loss_gd + num_tra_bd * tra_loss_bd) /
                (num_tra_gd + num_tra_bd))
    val_loss = ((num_val_gd * val_loss_gd + num_val_bd * val_loss_bd) /
                (num_val_gd + num_val_bd))
    losses = (tra_loss_gd, tra_loss_bd, val_loss_gd, val_loss_bd)

    print('LOSS for train good/bad and val good/bad:\t'
          '{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(*losses))

    metrics.append(tra_loss, val_loss, val_loss_gd, val_loss_bd)

    rmse_val = results.compute_classification_rmse(fold_id, estimator)
    rmse_val = [rmse / np.pi * 180 for rmse in rmse_val]
    print('RMSE for train good/bad and val good/bad:\t'
          '{:3.2f}° / {:3.2f}° / {:3.2f}° / {:3.2f}°'.format(*rmse_val))

    # if num_classes > 1:
    #     acc_val = results.compute_classification_acc(fold_id)
    #     acc_val = [acc * 100 for acc in acc_val]
    #     print('ACC for good train/bad train good val/bad val:\t\t'
    #           '{:3.1f}% / {:3.1f}% / {:3.1f}% / {:3.1f}%'
    #           ''.format(*acc_val))

    return rmse_val


# TODO: unused, to be revised below

# def find_lr(device, train_loader, loss_fn, init_value=1e-2, final_value=1e3,
#             beta=0.98, mult=1.1):
#     lr = init_value
#     avg_loss = 0.
#     best_loss = 0.
#     losses = []
#     log_lrs = []
#     num = int(np.log(final_value / init_value) / np.log(mult))+1
#     for i in range(1, num+1):
#         print(i, num)
#         encoded_space_dim = 256
#         encoder = Encoder(encoded_space_dim)
#         decoder = Decoder(encoded_space_dim)
#         params_to_optimize = [
#             {'params': encoder.parameters()},
#             {'params': decoder.parameters()}
#         ]

#         # Move both the encoder and the decoder to the selected device
#         encoder.to(device)
#         decoder.to(device)
#         optimizer = optim.Adam(params_to_optimize, lr=lr)
#
#         loss = train_epoch(encoder, decoder, device, train_loader, loss_fn,
#                            optimizer)
#         avg_loss = beta * avg_loss + (1-beta) *loss
#         smoothed_loss = avg_loss / (1 - beta**i)
#         #Stop if the loss is exploding
#         if i > 1 and smoothed_loss > 4 * best_loss:
#             return log_lrs, losses
#         #Record the best loss
#         if smoothed_loss < best_loss or i==1:
#             best_loss = smoothed_loss
#         #Store the values
#         losses.append(loss)
#         log_lrs.append(np.log10(lr))
#         #Update the lr for the next step
#         lr *= mult
#         optimizer.param_groups[0]['lr'] = lr
#     return log_lrs, loss

# logs,losses = find_lr(device, train_loader, loss_fn)
# plt.figure(figsize=(10,8))
# plt.plot(logs,losses)
# print(logs, losses)


# class Orientation_Loss(torch.nn.Module):
#     def __init__(self):
#         super(Orientation_Loss, self).__init__()

#     def forward(self, gt_in_radians, est_in_radians):
#         deltas = gt_in_radians - est_in_radians
#         deltas[deltas > np.pi/2.0] = np.pi - deltas[deltas > np.pi/2.0]
#         delta_sqr = deltas ** 2
#         totloss = torch.mean(delta_sqr)
#         return totloss
