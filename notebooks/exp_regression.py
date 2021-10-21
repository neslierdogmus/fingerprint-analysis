# %%
# import modules and packages
import yaml
from types import SimpleNamespace
from pathlib import Path
import random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from foe_mlp import FOEMLP
from foe_autoencoder import FOEAutoencoder
from foe_fingerprint_dataset import FOEFingerprintDataset
from foe_results import FOEResults

from time import time

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'widget')


def angle_loss(y, ye):
    delta_sqr = [torch.min(torch.abs(yi-yei), np.pi-torch.abs(yi-yei))
                 for (yi, yei) in zip(y, ye)]
    return torch.mean(torch.stack(delta_sqr))


def plot_loss(metrics):
    plt.figure()
    plt.semilogy(metrics['train_loss'], label='Train')
    plt.semilogy(metrics['val_loss_gd'], label='Valid_Good')
    plt.semilogy(metrics['val_loss_bd'], label='Valid_Bad')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.show()


# %%
# parse and configure the experiment parameters

with open('args.yml') as f:
    args = yaml.safe_load(f)
    args = SimpleNamespace(**args)

dataset_dir = Path(args.dataset_dir)
output_dir = Path(args.output_dir)

models_dir = output_dir.joinpath('models')
splits_dir = output_dir.joinpath('splits')
logs_dir = output_dir.joinpath('logs')
results_dir = output_dir.joinpath('results')

output_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
splits_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

encoded_space_dim = args.encoded_space_dim
patch_size = args.patch_size
radius = patch_size * 2
split_id = args.split_id
num_folds = args.num_folds
batch_size = args.batch_size
use_cpu = args.use_cpu

ae_num_epochs = args.ae_num_epochs
ae_learning_rate = args.ae_learning_rate
mlp_num_epochs = args.mlp_num_epochs
mlp_learning_rate = args.mlp_learning_rate

ae_continue_training = args.ae_continue_training
mlp_continue_training = args.mlp_continue_training


use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'
print('Using {}...'.format(device))
print('Patch size: {}'.format(patch_size))
print('Batch size: {}'.format(batch_size))
print('Rotation delta: {:.1f}°'.format(args.delta_r))

if args.seed is not None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)

NUM_WORKERS = 4
results = FOEResults()

# %%
# configure fingerprint patch dataset in folds and run the experiments

fpd_gd = FOEFingerprintDataset(dataset_dir, 'good')
fpd_gd.set_split_indices(splits_dir, split_id, num_folds)

fpd_bd = FOEFingerprintDataset(dataset_dir, 'bad')
fpd_bd.set_split_indices(splits_dir, split_id, num_folds)

for fold_id in range(num_folds):
    print('* '*30)
    print('Experiments for fold {} starts...'.format(fold_id))
    results.set_fold(fold_id)

    tset_gd, vset_gd = fpd_gd.get_patch_datasets(fold_id, radius, patch_size)
    tset_bd, vset_bd = fpd_bd.get_patch_datasets(fold_id, radius, patch_size)
    tset = tset_gd.merge(tset_bd)

    if args.train_with_bad:
        ae_tset = tset
    else:
        ae_tset = tset_gd

    mlp_tset = tset

    ae_train_loader = DataLoader(ae_tset, batch_size=batch_size,
                                 num_workers=NUM_WORKERS,
                                 shuffle=True, pin_memory=use_gpu)
    mlp_train_loader = DataLoader(mlp_tset, batch_size=batch_size,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True, pin_memory=use_gpu)
    val_loader_gd = DataLoader(vset_gd, batch_size=batch_size,
                               num_workers=NUM_WORKERS,
                               shuffle=True, pin_memory=use_gpu)
    val_loader_bd = DataLoader(vset_bd, batch_size=batch_size,
                               num_workers=NUM_WORKERS,
                               shuffle=True, pin_memory=use_gpu)

    if not args.no_hflip:
        ae_tset.set_hflip(True)
        mlp_tset.set_hflip(True)
    if args.delta_r > 0.0:
        ae_tset.set_delta_r(args.delta_r)
        mlp_tset.set_delta_r(args.delta_r)
        vset_gd.set_delta_r(args.delta_r)
        vset_bd.set_delta_r(args.delta_r)

    ae_name = 'foe_ae_ps{:03d}_es{:02d}_s{}_e{:04d}_f{}.pt'
    if ae_continue_training:
        ae_name = ae_name.format(patch_size, encoded_space_dim, split_id,
                                 ae_num_epochs, fold_id)
        ae_path = list(models_dir.glob(ae_name))[-1]

    ae_name = ae_name.format(patch_size, encoded_space_dim, split_id,
                             ae_num_epochs, fold_id)
    ae_path = models_dir.joinpath(ae_name)
    if ae_path.exists():
        (ae, batch_size, done_epochs, splits_dir, split_id, FOLD_IDX,
         val_results) = FOEAutoencoder.load_checkpoint(ae_path, device, True)
    else:
        ae = FOEAutoencoder(patch_size, encoded_space_dim, device)
        done_epochs = 0

        print("""Training and validating with:
        Encoded space dimension: {}
        Learning rate: {}
        Training set size: {}
        Validation set size (good): {}
        Validation set size (bad): {}""".format(encoded_space_dim,
                                                ae_learning_rate,
                                                len(ae_train_loader.dataset),
                                                len(val_loader_gd.dataset),
                                                len(val_loader_bd.dataset)))

        loss_fn = torch.nn.MSELoss()
        optimizer = optim.Adam(ae.parameters(), lr=ae_learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   [int(0.4 * ae_num_epochs),
                                                    int(0.75 * ae_num_epochs)],
                                                   gamma=0.1)
        ae_metrics = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': []}
        epoch = done_epochs
        for epoch in range(done_epochs+1, done_epochs+ae_num_epochs+1):
            train_loss = ae.train_epoch(ae_train_loader, loss_fn, optimizer)
            val_loss_gd = ae.val_epoch(val_loader_gd, loss_fn)
            val_loss_bd = ae.val_epoch(val_loader_bd, loss_fn)

            print('EPOCH {}/{}\t'
                  'Losses for train/good val/bad val: {:.4f} / {:.4f} / {:.4f}'
                  .format(epoch, done_epochs + ae_num_epochs, train_loss,
                          val_loss_gd, val_loss_bd))

            ae_metrics['train_loss'].append(train_loss)
            ae_metrics['val_loss_gd'].append(val_loss_gd)
            ae_metrics['val_loss_bd'].append(val_loss_bd)

        plot_loss(ae_metrics)
        ae.save_checkpoint(ae_path, batch_size, epoch,
                           splits_dir, split_id, fold_id)

    ae.plot_outputs(vset_gd, 5)
    ae.plot_outputs(vset_bd, 5)

    mlp_tset.set_encoder(ae.encoder.to('cpu'))
    tset_gd.set_encoder(ae.encoder.to('cpu'))
    vset_gd.set_encoder(ae.encoder.to('cpu'))
    tset_bd.set_encoder(ae.encoder.to('cpu'))
    vset_bd.set_encoder(ae.encoder.to('cpu'))

    ae_n_train = len(ae_train_loader.dataset)
    mlp_n_train = len(mlp_train_loader.dataset)
    n_val_gd = len(val_loader_gd.dataset)
    n_val_bd = len(val_loader_bd.dataset)

    mlp_name = 'foe_mlp_inp{:03d}_s{}_e{:04d}_f{}.pt'
    mlp_name = mlp_name.format(encoded_space_dim, split_id,
                               mlp_num_epochs, fold_id)
    mlp_path = models_dir.joinpath(mlp_name)
    loss_fn = angle_loss
    if mlp_path.exists():
        (mlp, batch_size, done_epochs, splits_dir, split_id, fold_id,
         val_results) = FOEMLP.load_checkpoint(mlp_path, device, 0, np.pi,
                                               True)
        # mlp._set_encoder(ae.encoder)
    else:
        mlp = FOEMLP(encoded_space_dim, device, 0, np.pi)
        done_epochs = 0

        print('''
        Input dimension: {}
        Learning rate: {}
        Training set size: {}
        Validation set size (good): {}
        Validation set size (bad): {}'''.format(encoded_space_dim,
                                                mlp_learning_rate,
                                                mlp_n_train, n_val_gd,
                                                n_val_bd))

        optimizer = optim.Adam(mlp.parameters(), lr=mlp_learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   [int(0.5 * mlp_num_epochs)],
                                                   gamma=0.1)

        mlp_metrics = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': [],
                       'train_rmse': [], 'val_rmse_gd': [], 'val_rmse_bd': []}
        epoch = done_epochs
        for epoch in range(done_epochs+1, done_epochs+mlp_num_epochs+1):
            ts = time()
            train_loss = mlp.train_epoch(mlp_train_loader, loss_fn, optimizer)
            _ = mlp.val_epoch(mlp_train_loader, loss_fn, results, False)
            val_loss_gd = mlp.val_epoch(val_loader_gd, loss_fn, results)
            val_loss_bd = mlp.val_epoch(val_loader_bd, loss_fn, results)
            scheduler.step()

            print('EPOCH {}/{}\t'
                  'Losses for train/good val/bad val: {:.4f} / {:.4f} / {:.4f}'
                  .format(epoch, done_epochs + mlp_num_epochs, train_loss,
                          val_loss_gd, val_loss_bd))

            mlp_metrics['train_loss'].append(train_loss)
            mlp_metrics['val_loss_gd'].append(val_loss_gd)
            mlp_metrics['val_loss_bd'].append(val_loss_bd)

            df = results.compute_regression_rmse2()
            rmse_train_gd = df[(df['fold'] == fold_id) & (df.is_good) &
                               (~df.is_test)].rmse.iloc[0] * 180 / np.pi
            rmse_train_bd = df[(df['fold'] == fold_id) & (~df.is_good) &
                               (~df.is_test)].rmse.iloc[0] * 180 / np.pi
            rmse_val_gd = df[(df['fold'] == fold_id) & (df.is_good) &
                             (df.is_test)].rmse.iloc[0] * 180 / np.pi
            rmse_val_bd = df[(df['fold'] == fold_id) & (~df.is_good) &
                             (df.is_test)].rmse.iloc[0] * 180 / np.pi

            print('RESULTS:\ttrain rmse good/bad {:.1f}°/{:.1f}° - '
                  'val rmse good/bad {:.1f}°/{:.1f}°'
                  .format(rmse_train_gd, rmse_train_bd,
                          rmse_val_gd, rmse_val_bd))
            results.plot_hist()
            te = time()
            print('Epoch took %2.4f sec' % (te-ts))
        plot_loss(mlp_metrics)
        mlp.save_checkpoint(mlp_path, batch_size, epoch, splits_dir,
                            split_id, fold_id)

df_final = df.groupby(['is_test',
                       'is_good']).agg({'rmse': ['mean', 'std']}).reset_index()
rmse_train_gd = df_final[(~df_final.is_test) & (df_final.is_good)].rmse
rmse_train_bd = df_final[(~df_final.is_test) & (~df_final.is_good)].rmse
rmse_val_gd = df_final[(df_final.is_test) & (df_final.is_good)].rmse
rmse_val_bd = df_final[(df_final.is_test) & (~df_final.is_good)].rmse
print('FINAL RESULTS:\n\t'
      'Train bad rmse: {:.1f}° ± {:.1f}°\n\t'
      'Train good rmse: {:.1f}° ± {:.1f}°\n\t'
      'Val bad rmse: {:.1f}° ± {:.1f}°\n\t'
      'Val good rmse: {:.1f}° ± {:.1f}°\n'
      .format(rmse_train_gd['mean'].iloc[0] * 180 / np.pi,
              rmse_train_gd['std'].iloc[0] * 180 / np.pi,
              rmse_train_bd['mean'].iloc[0] * 180 / np.pi,
              rmse_train_bd['std'].iloc[0] * 180 / np.pi,
              rmse_val_gd['mean'].iloc[0] * 180 / np.pi,
              rmse_val_gd['std'].iloc[0] * 180 / np.pi,
              rmse_val_bd['mean'].iloc[0] * 180 / np.pi,
              rmse_val_bd['std'].iloc[0] * 180 / np.pi))
