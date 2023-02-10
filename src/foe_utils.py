#!/usr/bin/env python3
import sys
import random

import numpy as np

import torch

from .foe_conv_net import FOEConvNet
from .foe_patch import FOEOrientation
from .foe_fingerprint import FOEFingerprint
from .foe_patch_dataset import FOEPatchDataset

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def init_model_and_datasets(fp_paths, n_classes=36, patch_size=32,
                            n_folds=5, fold_idx=0, model_path=None, 
                            no_shuffle=False, verbose=True):
    fp_list = []
    for fp_path in fp_paths:
        fp_list.extend(FOEFingerprint.load_index_file(fp_path[0],
                                                      'index.txt',
                                                      fp_path[1]))

    if verbose:
        eprint('Loaded {} fingerprints.'.format(len(fp_list)))

    if model_path is None:
        if not no_shuffle:
            random.shuffle(fp_list)
            eprint('Randomized splits.')

            # create train/val split for fold 0
            tset, vset = FOEPatchDataset.trainval_for_cv(fp_list, n_classes,
                                                         patch_size,
                                                         n_folds, fold_idx)
            model = FOEConvNet(patch_size, n_classes)
    else:
        model, train_images = load_checkpoint(model_path, True)
        tset, vset = FOEPatchDataset.trainval_from_split(fp_list, n_classes,
                                                         patch_size,
                                                         train_images)
    return tset, vset, model


def train_model(model, train_loader, optimizer, criterion, device):
    train_loss = 0.0
    n_train_batches = 0
    for idx, (x, y, *_) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        ye = model(x)
        loss = criterion(ye, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_train_batches += 1

    return train_loss / n_train_batches


def validate_model(model, val_loader, criterion, device):
    with torch.no_grad():
        val_acc = 0.0
        val_loss = 0.0
        n_val_batches = 0
        val_results = []
        for x, y, gt_thetas, item_indices in val_loader:
            x, y = x.to(device), y.to(device)
            y_dist = model(x)
            _, class_idx = y_dist.max(1)
            val_acc += (class_idx == y).sum().item()
            val_loss += criterion(y_dist, y).item()

            val_results.append((gt_thetas.cpu(),
                                y_dist.cpu(),
                                item_indices.cpu()))
            n_val_batches += 1

        n_val = len(val_loader.dataset)
        val_loss /= n_val_batches
        val_acc /= n_val

        return val_loss, val_acc, val_results


def rmse_from_results(results, dataset):
    images = dataset.images
    n_patches = {image: 0 for image in images}
    rmse = {image: 0.0 for image in images}
    for gt_thetas, y_dist, item_indices in results:
        err_sqr = FOEOrientation.estimation_error_sqr(gt_thetas,
                                                      y_dist)
    for item_idx, e in zip(item_indices, err_sqr):
        image = dataset.patches[item_idx].filename
        n_patches[image] += 1
        rmse[image] += e.item()

    for image in images:
        rmse[image] = np.sqrt(rmse[image] / n_patches[image])

    rmse_all = np.asarray(list(rmse.values())).mean()
    rmse_good = np.asarray([r for image, r in rmse.items()
                            if dataset.is_image_good[image]]).mean()
    rmse_bad = np.asarray([r for image, r in rmse.items()
                           if not dataset.is_image_good[image]]).mean()
    return rmse_all, rmse_good, rmse_bad


def save_checkpoint(model_dir, model, batch_size, epochs,
                    train_images, val_results=None, verbose=True):
    chk_dict = {'mstate': model.state_dict(),
                'Nc': model.n_classes,
                'Np': model.patch_size,
                'Nb': batch_size,
                'Ne': epochs,
                'train-images': train_images,
                'val-results': val_results}
    model_path = model_dir.joinpath('foe_conv_net_c{}_'
                                    'b{}_e{:04d}.pt'.format(model.n_classes,
                                                            batch_size,
                                                            epochs))
    torch.save(chk_dict, model_path)
    if verbose:
        eprint('Saved model to {}'.format(model_path))
    return model_path


def load_checkpoint(model_path, verbose=True):
    chk_dict = torch.load(model_path)
    n_classes = chk_dict['Nc']
    patch_size = chk_dict['Np']
    train_images = chk_dict['train-images']
    mstate = chk_dict['mstate']

    model = FOEConvNet(patch_size, n_classes)
    model.load_state_dict(mstate)

    if verbose:
        eprint("""Loaded model from {}:
   # classes: {}
   patch size: {}""".format(model_path,
                            n_classes,
                            patch_size))

    return model, train_images
