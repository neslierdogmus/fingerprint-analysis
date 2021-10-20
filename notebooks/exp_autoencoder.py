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

from foe_autoencoder import FOEAutoencoder
from foe_fingerprint_dataset import FOEFingerprintDataset

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'widget')

# %%
# parse and configure the experiment parameters

with open("args.yml") as f:
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

model_path = None
if args.ae_model_name is not None:
    model_path = models_dir.joinpath(args.ae_model_name)

encoded_space_dim = args.encoded_space_dim
patch_size = args.patch_size
radius = patch_size * 2
split_id = args.split_id
num_folds = args.num_folds

num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
use_cpu = args.use_cpu

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'

if args.seed is not None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)


FOLD_IDX = 0
NUM_WORKERS = 4


# %%
# configure models to train

if model_path is None:
    model = FOEAutoencoder(patch_size, encoded_space_dim, device=device)
    done_epochs = 0
else:
    (model, batch_size, done_epochs, splits_dir, split_id, FOLD_IDX,
     val_results) = FOEAutoencoder.load_checkpoint(model_path, device, True)

    patch_size = model.patch_size
    encoded_space_dim = model.encoded_space_dim
    print('Model is loaded from {}. Given split data, patch size, encoded'
          ' space dimension, number of folds and batch size will be ignored'
          .format(model_path))


# %%
# configure fingerprint patch datasets

fpd_gd = FOEFingerprintDataset(dataset_dir, 'good')
fpd_gd.set_split_indices(splits_dir, split_id, num_folds)
tset_gd, vset_gd = fpd_gd.get_patch_datasets(FOLD_IDX, radius, patch_size)

fpd_bd = FOEFingerprintDataset(dataset_dir, 'bad')
fpd_bd.set_split_indices(splits_dir, split_id, num_folds)
tset_bd, vset_bd = fpd_bd.get_patch_datasets(FOLD_IDX, radius, patch_size)

tset_all = tset_gd.merge(tset_bd)

if args.train_with_bad:
    tset = tset_all
else:
    tset = tset_gd

if not args.no_hflip:
    tset.set_hflip(True)
if args.delta_r > 0.0:
    tset.set_delta_r(args.delta_r)

train_loader = DataLoader(tset, batch_size=batch_size,
                          num_workers=NUM_WORKERS,
                          shuffle=True, pin_memory=use_gpu)
val_loader_gd = DataLoader(vset_gd, batch_size=batch_size,
                           num_workers=NUM_WORKERS,
                           shuffle=True, pin_memory=use_gpu)
val_loader_bd = DataLoader(vset_bd, batch_size=batch_size,
                           num_workers=NUM_WORKERS,
                           shuffle=True, pin_memory=use_gpu)


# %%
# print current status

print("Using {}...".format(device))
print("""Patch size: {}
Encoded space dimension: {}
Batch size: {}
Learning rate: {}
Rotation delta: {:.1f}°
Training set size: {}
Validation set size (good): {}
Validation set size (bad): {}""".format(patch_size, encoded_space_dim,
                                        batch_size, learning_rate,
                                        args.delta_r * 180.0 / np.pi,
                                        len(train_loader.dataset),
                                        len(val_loader_gd.dataset),
                                        len(val_loader_bd.dataset)))


# %%
# configure training parameters and train

loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.4 * num_epochs),
                                                       int(0.75 * num_epochs)],
                                           gamma=0.1)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
#                                           steps_per_epoch=len(train_loader),
#                                           epochs=num_epochs)

losses = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': []}
epoch = done_epochs
for epoch in range(done_epochs+1, done_epochs+num_epochs+1):
    train_loss = model.train_epoch(train_loader, loss_fn, optimizer)
    val_loss_gd = model.val_epoch(val_loader_gd, loss_fn)
    val_loss_bd = model.val_epoch(val_loader_bd, loss_fn)
    print('EPOCH {}/{}\ttrain loss {:.4f}\tval loss good {:.4f}'
          '\tval loss bad {:.4f}'.format(epoch, done_epochs + num_epochs,
                                         train_loss, val_loss_gd, val_loss_bd))
    losses['train_loss'].append(train_loss)
    losses['val_loss_gd'].append(val_loss_gd)
    losses['val_loss_bd'].append(val_loss_bd)

    if epoch % 20 == 0:
        model.plot_outputs(vset_gd, 5)
        model.plot_outputs(vset_bd, 5)

model.save_checkpoint(models_dir, batch_size, epoch,
                      splits_dir, split_id, FOLD_IDX)


# %%
# plot results

model.plot_outputs(vset_gd, 5)
model.plot_outputs(vset_bd, 5)

plt.figure()
plt.semilogy(losses['train_loss'], label='Train')
plt.semilogy(losses['val_loss_gd'], label='Valid_Good')
plt.semilogy(losses['val_loss_bd'], label='Valid_Bad')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid()
plt.legend()
plt.title('loss')
plt.show()
