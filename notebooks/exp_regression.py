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
from functions import (create_dataloaders, init_model, train_epoch, val_epoch,
                       save_model)

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
# %%
# configure fingerprint patch dataset in folds and run the experiments

fpd_gd = FOEFingerprintDataset(dataset_dir, 'good')
fpd_gd.set_split_indices(splits_dir, args.split_id, args.num_folds)

fpd_bd = FOEFingerprintDataset(dataset_dir, 'bad')
fpd_bd.set_split_indices(splits_dir, args.split_id, args.num_folds)

RMSE = []
for fold_id in range(args.num_folds):
    print('* '*40)
    print('Experiments for fold {} starts...'.format(fold_id))

    (ae_train_loader, train_loader, val_loader_gd, val_loader_bd,
     val_loader) = create_dataloaders(fpd_gd, fpd_bd, fold_id, NUM_WORKERS,
                                      use_gpu, args)
    num_val_gd = len(val_loader_gd.dataset)
    num_val_bd = len(val_loader_bd.dataset)

    metrics = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': [],
               'train_rmse': [], 'val_rmse_gd': [], 'val_rmse_bd': []}
    encoder = None
    if args.approach == 'ae_mlp':
        ae, metrics = init_model('ae', models_dir, file_id, fold_id,
                                 device, args)
        num_epochs = args.ae_num_epochs - len(metrics.data['train_loss'])
        loss_fn = torch.nn.MSELoss()
        optimizer = optim.Adam(ae.parameters(), lr=args.ae_learning_rate)
        scheduler_array = [int(0.4 * args.ae_num_epochs),
                           int(0.7 * args.ae_num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   scheduler_array,
                                                   gamma=0.1)
        for e in range(1, num_epochs+1):
            train_loss = train_epoch(ae, train_loader, loss_fn, optimizer,
                                     is_ae=True)
            val_loss_gd = val_epoch(ae, val_loader_gd, loss_fn, is_ae=True)
            val_loss_bd = val_epoch(ae, val_loader_bd, loss_fn, is_ae=True)
            val_loss = ((num_val_gd * val_loss_gd + num_val_bd * val_loss_bd) /
                        (num_val_gd + num_val_bd))
            scheduler.step()

            print('EPOCH {}/{}\tLOSS for train/good val/bad val: '
                  '{:.3f} / {:.3f} / {:.3f} / {:.3f}'
                  .format(e, num_epochs, train_loss, val_loss,
                          val_loss_gd, val_loss_bd))

            metrics.append(train_loss, val_loss, val_loss_gd, val_loss_bd)
        metrics.plot()
        ae.plot_outputs(val_loader_gd.dataset, 5)
        ae.plot_outputs(val_loader_bd.dataset, 5)
        if num_epochs > 0:
            save_model(ae)
            metrics.save()
        encoder = ae.encoder

        model, metrics = init_model('mlp', models_dir, file_id, fold_id,
                                    device, args)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler_array = [int(0.4 * args.num_epochs),
                           int(0.75 * args.num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_array,
                                                   gamma=0.1)
    elif args.approach == 'cnn':
        model, metrics = init_model('cnn', models_dir, file_id, fold_id,
                                    device, args)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler_array = [int(0.4 * args.num_epochs),
                           int(0.75 * args.num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_array,
                                                   gamma=0.1)
    else:
        raise Exception('The "approach" parameter should'
                        ' either be "ae_mlp" or "cnn"!')

    estimator = None
    if args.num_classes > 1:
        estimator = FOEOrientation.radians_from_marginalization

    num_epochs = args.num_epochs - len(metrics.data['train_loss'])
    for e in range(1, num_epochs+1):
        results = FOEResults()
        results.set_fold(fold_id)

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer,
                                 encoder=encoder, results=results)
        val_loss_gd = val_epoch(model, val_loader_gd, loss_fn,
                                encoder=encoder, results=results)
        val_loss_bd = val_epoch(model, val_loader_bd, loss_fn,
                                encoder=encoder, results=results)
        val_loss = ((num_val_gd * val_loss_gd + num_val_bd * val_loss_bd) /
                    (num_val_gd + num_val_bd))

        metrics.append(train_loss, val_loss, val_loss_gd, val_loss_bd)
        scheduler.step()

        print('EPOCH {}/{}\tLOSS for train/good val/bad val:\t'
              '{:.3f} / {:.3f} / {:.3f} / {:.3f}'
              .format(e, num_epochs, train_loss, val_loss,
                      val_loss_gd, val_loss_bd))

        if e % 10 == 1 or e == args.num_epochs:
            rmse_val = results.compute_classification_rmse(fold_id, estimator)
            rmse_val = [rmse / np.pi * 180 for rmse in rmse_val]
            print('RMSE for good train/bad train good val/bad val:\t\t'
                  '{:3.1f}° / {:3.1f}° / {:3.1f}° / {:3.1f}°'
                  ''.format(*rmse_val))

            if args.num_classes > 1:
                acc_val = results.compute_classification_acc(fold_id)
                acc_val = [acc * 100 for acc in acc_val]
                print('ACC for good train/bad train good val/bad val:\t\t'
                      '{:3.1f}% / {:3.1f}% / {:3.1f}% / {:3.1f}%'
                      ''.format(*acc_val))
    metrics.plot()
    if num_epochs > 0:
        save_model(model)
        metrics.save()
    else:
        results = FOEResults()
        results.set_fold(fold_id)
        train_loss = val_epoch(model, train_loader, loss_fn,
                               encoder=encoder, results=results, is_tr=True)
        val_loss_gd = val_epoch(model, val_loader_gd, loss_fn,
                                encoder=encoder, results=results)
        val_loss_bd = val_epoch(model, val_loader_bd, loss_fn,
                                encoder=encoder, results=results)
        val_loss = ((num_val_gd * val_loss_gd + num_val_bd * val_loss_bd) /
                    (num_val_gd + num_val_bd))

        print('Final LOSS for train/good val/bad val:\t'
              '{:.3f} / {:.3f} / {:.3f} / {:.3f}'
              .format(train_loss, val_loss, val_loss_gd, val_loss_bd))

        rmse_val = results.compute_classification_rmse(fold_id, estimator)
        rmse_val = [rmse / np.pi * 180 for rmse in rmse_val]
        print('RMSE for good train/bad train good val/bad val:\t\t'
              '{:3.1f}° / {:3.1f}° / {:3.1f}° / {:3.1f}°'.format(*rmse_val))

        if args.num_classes > 1:
            acc_val = results.compute_classification_acc(fold_id)
            acc_val = [acc * 100 for acc in acc_val]
            print('ACC for good train/bad train good val/bad val:\t\t'
                  '{:3.1f}% / {:3.1f}% / {:3.1f}% / {:3.1f}%'.format(*acc_val))
    RMSE.append(rmse_val)

RMSE_mean = np.array(RMSE).mean(0)
RMSE_std = np.array(RMSE).std(0)

print('FINAL RESULTS:\n\t'
      'Train good rmse: {:.1f}° ± {:.1f}°\n\t'
      'Train bad rmse: {:.1f}° ± {:.1f}°\n\t'
      'Val good rmse: {:.1f}° ± {:.1f}°\n\t'
      'Val bad rmse: {:.1f}° ± {:.1f}°\n'
      .format(RMSE_mean[0], RMSE_std[0], RMSE_mean[1], RMSE_std[1],
              RMSE_mean[2], RMSE_std[2], RMSE_mean[3], RMSE_std[3]))
