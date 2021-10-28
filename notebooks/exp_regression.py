# %%
# import modules and packages
import yaml
import hashlib
from types import SimpleNamespace
from pathlib import Path
import random
import numpy as np

import torch
import torch.optim as optim

from foe_fingerprint_dataset import FOEFingerprintDataset
from foe_orientation import FOEOrientation
from foe_results import FOEResults
from functions import (create_dataloaders, train_autoencoder,
                       initialize_mlp, initialize_cnn, train, plot_loss)

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'widget')


def angle_loss(y, ye):
    delta_sqr = [torch.min(torch.abs(yi-yei), np.pi-torch.abs(yi-yei))
                 for (yi, yei) in zip(y, ye)]
    return torch.mean(torch.stack(delta_sqr))


# %%
# parse and configure the experiment parameters
args_file = 'args.yml'

with open(args_file) as f:
    args = yaml.safe_load(f)
    args = SimpleNamespace(**args)

file_id = hashlib.md5(open(args_file, 'rb').read()).hexdigest()

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

use_gpu = torch.cuda.is_available() and not args.use_cpu
device = 'cuda' if use_gpu else 'cpu'

print('Using {}...'.format(device))
print('Patch size: {}'.format(args.patch_size))
print('Batch size: {}'.format(args.batch_size))

if args.seed is not None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)

NUM_WORKERS = 4
results_all = FOEResults()
# %%
# configure fingerprint patch dataset in folds and run the experiments

fpd_gd = FOEFingerprintDataset(dataset_dir, 'good')
fpd_gd.set_split_indices(splits_dir, args.split_id, args.num_folds)

fpd_bd = FOEFingerprintDataset(dataset_dir, 'bad')
fpd_bd.set_split_indices(splits_dir, args.split_id, args.num_folds)

for fold_id in range(args.num_folds):
    print('* '*40)
    print('Experiments for fold {} starts...'.format(fold_id))

    (ae_train_loader, train_loader, val_loader_gd, val_loader_bd,
     val_loader) = create_dataloaders(fpd_gd, fpd_bd, fold_id, NUM_WORKERS,
                                      use_gpu, args)

    metrics = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': [],
               'train_rmse': [], 'val_rmse_gd': [], 'val_rmse_bd': []}

    if args.approach == 'ae_mlp':
        ae = train_autoencoder(models_dir, fold_id, device, ae_train_loader,
                               val_loader_gd, val_loader_bd, args)
        (model, done_epochs, num_epochs,
         model_path, results) = initialize_mlp(models_dir, fold_id,
                                               device, args)
        model._set_encoder(ae.encoder)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        # loss_fn = angle_loss
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler_array = [int(0.4 * args.num_epochs),
                           int(0.75 * args.num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_array,
                                                   gamma=0.1)
    elif args.approach == 'cnn':
        (model, done_epochs, num_epochs,
         model_path, results) = initialize_cnn(models_dir, fold_id,
                                               device, args)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler_array = [int(0.4 * args.num_epochs),
                           int(0.75 * args.num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_array,
                                                   gamma=0.1)
    else:
        print('The "approach" parameter should either be "ae_mlp" or "cnn"!')
        break

    estimator = None
    if args.num_classes > 1:
        estimator = FOEOrientation.radians_from_marginalization

    epoch = done_epochs
    for epoch in range(done_epochs+1, num_epochs+1):
        print('EPOCH {}/{} for {}:'.format(epoch, num_epochs, args.approach))
        results = FOEResults()
        results.set_fold(fold_id)

        losses = train(model, train_loader, val_loader_gd, val_loader_bd,
                       loss_fn, optimizer, scheduler, results, metrics)

        print('LOSS for train/good val/bad val: '
              '{:.4f} / {:.4f} / {:.4f}'.format(*losses))

        if epoch % 10 == 1 or epoch == num_epochs:
            rmse_values = results.compute_classification_rmse(fold_id,
                                                              estimator)
            rmse_values = [rmse / np.pi * 180 for rmse in rmse_values]
            print('RMSE for good train/bad train good val/bad val:\t'
                  '{:.1f}°/{:.1f}°/{:.1f}°/{:.1f}°'.format(*rmse_values))

            if args.num_classes > 1:
                acc_values = results.compute_classification_acc(fold_id)
                acc_values = [acc * 100 for acc in acc_values]
                print('ACC for good train/bad train good val/bad val:\t'
                      '{:.1f}%/{:.1f}%/{:.1f}%/{:.1f}%'.format(*acc_values))
    results_all.merge(results)
    if model_path:
        plot_loss(metrics)
        model.save_checkpoint(model_path)
    else:
        pass

(rmse_tra_gd, rmse_tra_bd, rmse_val_gd,
 rmse_val_bd) = results_all.compute_rmse_stats(estimator)

print('FINAL RESULTS:\n\t'
      'Train bad rmse: {:.1f}° ± {:.1f}°\n\t'
      'Train good rmse: {:.1f}° ± {:.1f}°\n\t'
      'Val bad rmse: {:.1f}° ± {:.1f}°\n\t'
      'Val good rmse: {:.1f}° ± {:.1f}°\n'
      .format(rmse_tra_gd['mean'].iloc[0] * 180 / np.pi,
              rmse_tra_gd['std'].iloc[0] * 180 / np.pi,
              rmse_tra_bd['mean'].iloc[0] * 180 / np.pi,
              rmse_tra_bd['std'].iloc[0] * 180 / np.pi,
              rmse_val_gd['mean'].iloc[0] * 180 / np.pi,
              rmse_val_gd['std'].iloc[0] * 180 / np.pi,
              rmse_val_bd['mean'].iloc[0] * 180 / np.pi,
              rmse_val_bd['std'].iloc[0] * 180 / np.pi))
