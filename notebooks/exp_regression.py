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

from IPython import get_ipython

import sys
import os

sys.path.append(os.path.normpath(
                os.path.join(os.path.dirname(
                             os.path.abspath(__file__)), '..')))
from src.foe_mlp import FOEMLP                  # noqa: E402
from src.foe_autoencoder import FOEAutoencoder  # noqa: E402
from src.foe_utils import init_datasets         # noqa: E402

get_ipython().run_line_magic('matplotlib', 'widget')


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(0.1)


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
if args.mlp_model_name is not None:
    model_path = models_dir.joinpath(args.mlp_model_name)

ae_path = models_dir.joinpath(args.ae_model_name)
encoded_space_dim = args.encoded_space_dim
radius = args.radius
patch_size = args.patch_size
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
    model = FOEMLP(encoded_space_dim, device=device)
    done_epochs = 0
else:
    (model, batch_size, done_epochs, split_dir, split_id, FOLD_IDX,
     val_results) = FOEMLP.load_checkpoint(model_path, device, True)

    encoded_space_dim = model.input_dim
    print('Model is loaded from {}. Given split data, encoded space dimension,'
          ' number of folds and batch size will be ignored'.format(model_path))


# %%
# configure fingerprint patch datasets

(ae, *_) = FOEAutoencoder.load_checkpoint(ae_path, 'cpu', True)

tset_gd, vset_gd = init_datasets(dataset_dir, ['good'], splits_dir, split_id,
                                 radius=radius, patch_size=patch_size,
                                 n_folds=num_folds, fold_idx=FOLD_IDX)
tset_bd, vset_bd = init_datasets(dataset_dir, ['bad'], splits_dir, split_id,
                                 radius=radius, patch_size=patch_size,
                                 n_folds=num_folds, fold_idx=FOLD_IDX)
tset, _ = init_datasets(dataset_dir, ['good', 'bad'], splits_dir, split_id,
                        radius=radius, patch_size=patch_size,
                        n_folds=num_folds, fold_idx=FOLD_IDX)

if not args.no_hflip:
    tset.set_hflip(True)
if args.delta_r > 0.0:
    tset.set_delta_r(args.delta_r)

tset.set_autoencoder(ae)
tset_gd.set_autoencoder(ae)
vset_gd.set_autoencoder(ae)
tset_bd.set_autoencoder(ae)
vset_bd.set_autoencoder(ae)

train_loader = DataLoader(tset, batch_size=batch_size,
                          num_workers=NUM_WORKERS,
                          shuffle=True, pin_memory=use_gpu)
val_loader_gd = DataLoader(vset_gd, batch_size=batch_size,
                           num_workers=NUM_WORKERS,
                           shuffle=True, pin_memory=use_gpu)
val_loader_bd = DataLoader(vset_bd, batch_size=batch_size,
                           num_workers=NUM_WORKERS,
                           shuffle=True, pin_memory=use_gpu)

n_train = len(train_loader.dataset)
n_val_gd = len(val_loader_gd.dataset)
n_val_bd = len(val_loader_bd.dataset)

# %%
# print current status

print("Using {}...".format(device))
print("""Patch size: {}
Encoded space (input) dimension: {}
Batch size: {}
Learning rate: {}
Rotation delta: {:.1f}°
Training set size: {}
Validation set size (good): {}
Validation set size (bad): {}""".format(patch_size, encoded_space_dim,
                                        batch_size, learning_rate,
                                        args.delta_r * 180.0 / np.pi,
                                        n_train, n_val_gd, n_val_bd))

# %%
# configure training parameters and train


def tanloss(y, ye):
    return torch.mean((y[0]/y[1] - ye[0]/y[1])**2)


model = FOEMLP(encoded_space_dim, device)
# model.apply(init_weights)

loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.4 * num_epochs),
                                                       int(0.75 * num_epochs)],
                                           gamma=0.1)

metrics = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': [],
           'train_rmse': [], 'val_rmse_gd': [], 'val_rmse_bd': []}
epoch = done_epochs
for epoch in range(done_epochs+1, done_epochs+num_epochs+1):
    train_loss = model.train_epoch(train_loader, loss_fn, optimizer)
    val_loss_gd = model.val_epoch(val_loader_gd, loss_fn)
    val_loss_bd = model.val_epoch(val_loader_bd, loss_fn)
    scheduler.step()

    # train_rmse = np.sqrt(train_err_sqr / n_train) * 180.0 / np.pi  # in degrees
    # val_rmse_gd = np.sqrt(val_err_sqr_gd / n_val_gd) * 180.0 / np.pi
    # val_rmse_bd = np.sqrt(val_err_sqr_bd / n_val_bd) * 180.0 / np.pi

    # print('EPOCH {}/{}\ttrain loss / rmse {:.4f} / {:.1f}°\t'
    #       'val loss / rmse good {:.4f} / {:.1f}°\t'
    #       'val loss / rmse bad {:.4f} / {:.1f}°'
    #       .format(epoch + 1, done_epochs + num_epochs, train_loss, train_rmse,
    #               val_loss_gd, val_rmse_gd, val_loss_bd, val_rmse_bd))

    print('EPOCH {}/{}\ttrain loss {:.4f}\t'
          'val loss good{:.4f}\t'
          'val loss bad {:.4f}'
          .format(epoch, done_epochs + num_epochs, train_loss,
                  val_loss_gd, val_loss_bd))

    metrics['train_loss'].append(train_loss)
    metrics['val_loss_gd'].append(val_loss_gd)
    metrics['val_loss_bd'].append(val_loss_bd)

    if epoch % 20 == 0:
        model.plot_outputs(vset_gd, 5)
        model.plot_outputs(vset_bd, 5)

model.save_checkpoint(models_dir, batch_size, epoch,
                      splits_dir, split_id, FOLD_IDX)

# %%
# plot results

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
