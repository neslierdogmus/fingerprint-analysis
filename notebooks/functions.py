import torch
from torch.utils.data import DataLoader

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


def create_dataloaders(fpd_gd, fpd_bd, fold_id, num_workers, use_gpu, args):
    tset_gd, vset_gd = fpd_gd.get_patch_datasets(fold_id, args.patch_size*2,
                                                 args.patch_size,
                                                 args.num_classes)
    tset_bd, vset_bd = fpd_bd.get_patch_datasets(fold_id, args.patch_size*2,
                                                 args.patch_size,
                                                 args.num_classes)
    tset = tset_gd.merge(tset_bd)
    vset = vset_gd.merge(vset_bd)

    if args.train_with_bad:
        ae_tset = tset
    else:
        ae_tset = tset_gd

    ae_tset.set_hflip(args.hflip)
    tset.set_hflip(args.hflip)
    ae_tset.set_rotate(args.rotate)
    tset.set_rotate(args.rotate)

    ae_train_loader = DataLoader(ae_tset, batch_size=args.batch_size,
                                 num_workers=num_workers,
                                 shuffle=True, pin_memory=use_gpu)
    train_loader = DataLoader(tset, batch_size=args.batch_size,
                              num_workers=num_workers,
                              shuffle=True, pin_memory=use_gpu)
    val_loader_gd = DataLoader(vset_gd, batch_size=args.batch_size,
                               num_workers=num_workers,
                               shuffle=True, pin_memory=use_gpu)
    val_loader_bd = DataLoader(vset_bd, batch_size=args.batch_size,
                               num_workers=num_workers,
                               shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(vset, batch_size=args.batch_size,
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

    return (ae_train_loader, train_loader, val_loader_gd,
            val_loader_bd, val_loader)


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
        input = args.patch_size
        output = args.encoded_space_dim
    elif model_type == 'mlp':
        FOE_MODEL = FOE_MLP
        input = args.encoded_space_dim
        output = args.num_classes
    elif model_type == 'cnn':
        FOE_MODEL = FOE_CNN
        input = args.patch_size
        output = args.num_classes

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
        y_out = model(x)
        loss = loss_fn(y_out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if results is not None:
            for idx, gt, yo in zip(index, gt_in_radians, y_out.cpu()):
                patch = dataloader.dataset.patches[idx.item()]
                results.append(patch, False, gt.item(),
                               yo.detach().numpy())
    return total_loss / len(dataloader)


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
            y_out = model(x)
            loss = loss_fn(y_out, y)
            total_loss += loss.item()
            if results is not None:
                for idx, gt, yo in zip(index, gt_in_radians, y_out.cpu()):
                    patch = dataloader.dataset.patches[idx.item()]
                    results.append(patch, not is_tr, gt.item(),
                                   yo.detach().numpy())

    return total_loss / len(dataloader)
