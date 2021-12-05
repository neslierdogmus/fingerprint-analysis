# %%
# import modules and packages
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


def angle_loss(ye, y):
    delta_abs = [torch.min(torch.abs(yi-yei), np.pi-torch.abs(yi-yei))
                 for (yi, yei) in zip(y, ye)]
    return torch.mean(torch.stack(delta_abs))


def misclassification_cost_loss(ye, y):
    ye_lp = -1*torch.nn.functional.log_softmax(ye)
    # 18 => num_class
    costs = torch.vstack([torch.stack([torch.min(torch.abs(yi-yeii),
                                                 18-torch.abs(yi-yeii))/90
                                       for yeii in range(len(yei))])
                          for (yi, yei) in zip(y, ye)])
    return torch.sum(torch.mean(costs*ye_lp, 1))


def ordinal_loss(ye, y):
    # Create out modified target with [batch_size, num_labels] shape
    modified_y = torch.zeros_like(ye)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(y):
        if target < 9:
            modified_y[i, 0:target+1] = 1
        else:
            modified_y[i, 0:18-target] = 1
            modified_y[i, -1] = 1

    return torch.nn.MSELoss(reduction='none')(ye, modified_y).sum(axis=1).sum()


# %%
# parameters
dataset_dir = '../datasets/Finger/FOESamples'
output_dir = '../results'
use_cpu = False
seed = 0
split_id = 2
num_folds = 5
approach = 'cnn'
batch_size = 32

hflip = True
rotate = True
num_epochs = 500
learning_rate = 0.001

patch_size = 64
num_classes = 90

encoded_space_dim = 512
train_with_bad = True
ae_num_epochs = 20
ae_learning_rate = 0.00001
file_id = 'default_file_id'

# dataset_dir = ''
# output_dir = ''
# use_cpu = ''
# seed = ''
# split_id = ''
# num_folds = ''

# patch_size = ''
# batch_size = ''
# hflip = ''
# rotate = ''

# num_classes = ''
# num_epochs = ''
# learning_rate = ''
# approach = ''

# encoded_space_dim = ''
# train_with_bad = ''
# ae_num_epochs = ''
# ae_learning_rate = ''
# file_id = ''

# %%
# parse and configure the experiment parameters
args = dict(((k, eval(k)) for k in ('patch_size', 'batch_size', 'hflip',
                                    'rotate', 'num_classes', 'num_epochs',
                                    'learning_rate', 'approach',
                                    'encoded_space_dim', 'train_with_bad',
                                    'ae_num_epochs', 'ae_learning_rate')))
print(args)

dataset_dir = Path(dataset_dir)
output_dir = Path(output_dir)

models_dir = output_dir.joinpath('models')
splits_dir = output_dir.joinpath('splits')
logs_dir = output_dir.joinpath('logs')
results_dir = output_dir.joinpath('results')

output_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
splits_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'

print('Using {}...'.format(device))
print('Patch size: {}'.format(patch_size))
print('Batch size: {}'.format(batch_size))

if seed is not None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)

NUM_WORKERS = 4
# %%
# configure fingerprint patch dataset in folds and run the experiments

fpd_gd = FOEFingerprintDataset(dataset_dir, 'good')
fpd_gd.set_split_indices(splits_dir, split_id, num_folds)

fpd_bd = FOEFingerprintDataset(dataset_dir, 'bad')
fpd_bd.set_split_indices(splits_dir, split_id, num_folds)

fpd_sn = FOEFingerprintDataset(dataset_dir, 'synth')

RMSE = []
for fold_id in range(num_folds):
    print('* '*40)
    print('Experiments for fold {} starts...'.format(fold_id))

    ae_train_loader = train_loader = val_loader = None
    val_loader_gd = val_loader_bd = None

    (ae_train_loader, train_loader_gd, train_loader_bd, train_loader,
     val_loader_gd, val_loader_bd, val_loader) = create_dataloaders(fpd_gd, fpd_bd, fpd_sn, fold_id, NUM_WORKERS, use_gpu, args)
    num_val_gd = len(val_loader_gd.dataset)
    num_val_bd = len(val_loader_bd.dataset)

    encoder = None
    if approach == 'ae_mlp':
        ae, metrics = init_model('ae', models_dir, file_id, fold_id,
                                 device, args)
        n_epochs = ae_num_epochs - len(metrics.data['train_loss'])
        loss_fn = torch.nn.MSELoss(reduction='sum').to(device)
        optimizer = optim.Adam(ae.parameters(), lr=ae_learning_rate)
        # scheduler_array = [int(0.7 * ae_num_epochs),
        #                    int(0.9 * ae_num_epochs)]
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        #                                            scheduler_array,
        #                                            gamma=0.1)
        for e in range(1, n_epochs+1):
            train_loss = train_epoch(ae, ae_train_loader, loss_fn, optimizer,
                                     is_ae=True)
            val_loss_gd = val_epoch(ae, val_loader_gd, loss_fn, is_ae=True)
            val_loss_bd = val_epoch(ae, val_loader_bd, loss_fn, is_ae=True)
            val_loss = ((num_val_gd * val_loss_gd + num_val_bd * val_loss_bd) /
                        (num_val_gd + num_val_bd))
            # scheduler.step()

            print('EPOCH {}/{}\tLOSS for train/good val/bad val: '
                  '{:.3f} / {:.3f} / {:.3f} / {:.3f}'
                  .format(e, n_epochs, train_loss, val_loss,
                          val_loss_gd, val_loss_bd))

            metrics.append(train_loss, val_loss, val_loss_gd, val_loss_bd)
        metrics.plot()
        ae.plot_outputs(val_loader_gd.dataset, 5)
        ae.plot_outputs(val_loader_bd.dataset, 5)
        if n_epochs > 0:
            save_model(ae)
            metrics.save()
        encoder = ae.encoder

        model, metrics = init_model('mlp', models_dir, file_id, fold_id,
                                    device, args)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler_array = [int(0.8 * num_epochs),
                           int(0.9 * num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_array,
                                                   gamma=0.1)
    elif approach == 'cnn':
        model, metrics = init_model('cnn', models_dir, file_id, fold_id,
                                    device, args)
        # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        # loss_fn = ordinal_loss
        # loss_fn = torch.nn.MSELoss(reduction='sum')
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler_array = [int(0.8 * num_epochs),
                           int(0.9 * num_epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_array,
                                                   gamma=0.1)
    else:
        raise Exception('The "approach" parameter should'
                        ' either be "ae_mlp" or "cnn"!')

    estimator = None
    if num_classes > 1:
        # estimator = FOEOrientation.radians_from_marginalization
        codes = FOEOrientation.create_codes_list(num_classes)
        estimator = FOEOrientation.radians_from_ordinal_code(codes)

    n_epochs = num_epochs - len(metrics.data['train_loss'])
    for e in range(1, n_epochs+1):
        results = FOEResults()
        results.set_fold(fold_id)

        if e < 0:
            train_loss = train_epoch(model, train_loader_gd, loss_fn,
                                     optimizer, encoder=encoder, results=results)
        elif e < 0:
            train_loss = train_epoch(model, train_loader_bd, loss_fn,
                                     optimizer, encoder=encoder, results=results)
        else:
            train_loss = train_epoch(model, train_loader, loss_fn,
                                     optimizer, encoder=encoder, results=results)
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
              .format(e, n_epochs, train_loss, val_loss,
                      val_loss_gd, val_loss_bd))

        if e % 20 == 1 or e == n_epochs:
            rmse_val = results.compute_classification_rmse(fold_id, estimator)
            rmse_val = [rmse / np.pi * 180 for rmse in rmse_val]
            print('RMSE for good train/bad train good val/bad val:\t\t'
                  '{:3.1f}° / {:3.1f}° / {:3.1f}° / {:3.1f}°'
                  ''.format(*rmse_val))

            # if num_classes > 1:
            #     acc_val = results.compute_classification_acc(fold_id)
            #     acc_val = [acc * 100 for acc in acc_val]
            #     print('ACC for good train/bad train good val/bad val:\t\t'
            #           '{:3.1f}% / {:3.1f}% / {:3.1f}% / {:3.1f}%'
            #           ''.format(*acc_val))

    metrics.plot()
    if n_epochs > 0:
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

        results.analyze(fold_id, estimator)

        print('Final LOSS for train/good val/bad val:\t'
              '{:.3f} / {:.3f} / {:.3f} / {:.3f}'
              .format(train_loss, val_loss, val_loss_gd, val_loss_bd))

        rmse_val = results.compute_classification_rmse(fold_id, estimator)
        rmse_val = [rmse / np.pi * 180 for rmse in rmse_val]
        print('RMSE for good train/bad train good val/bad val:\t\t'
              '{:3.1f}° / {:3.1f}° / {:3.1f}° / {:3.1f}°'.format(*rmse_val))

        if num_classes > 1:
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
