#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from pathlib import Path
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from foe_fingerprint import FOEFingerprint
from foe_patch_dataset import FOEPatchDataset
from foe_conv_net import FOEConvNet
from foe_utils import *


if __name__ == '__main__':
    VALIDATION_BATCH_SIZE = 256
    NUM_WORKERS = 4
    FOLD_IDX = 0

    parser = ArgumentParser(description='FOE baseline experiments')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='display more information')
    parser.add_argument('-S', '--seed', dest='seed',
                        default=None, metavar='SEED',
                        help='process fingerprint patches of size SxS')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='/opt/data/FOESamples', metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=32, type=int, metavar='S',
                        help='process fingerprint patches of size SxS')
    parser.add_argument('-n', '--n-folds', dest='n_folds',
                        default=5, type=int, metavar='Nf',
                        help='number of folds')
    parser.add_argument('-N', '--n-classes', dest='n_classes',
                        default=8, type=int, metavar='Nc',
                        help='number of classes')
    parser.add_argument('--without-bad', action='store_true',
                        help='do not load bad fingerprints')
    parser.add_argument('--no-shuffle', dest='no_shuffle',
                        action='store_true',
                        help='do not shuffle fingerprints before'
                        ' fold computation')
    parser.add_argument('--no-hflip', dest='no_hflip',
                        action='store_true',
                        help='do not horizantally flip samples'
                        ' to augment the training set')
    parser.add_argument('--delta-r', dest='delta_r', type=float,
                        default=np.pi/12.0, metavar='Dr',
                        help='rotate sample from [-Dr, Dr] to augment'
                        ' the training set. Set Dr <= 0.0 to turn of.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        default='output', metavar='DIR',
                        help='output directory to store results')
    parser.add_argument('-e', '--n-epochs', dest='n_epochs', type=int,
                        default=50, metavar='E',
                        help='number of epochs')
    parser.add_argument('-B', '--batch-size', dest='batch_size', type=int,
                        default=256, metavar='B',
                        help='batch size for training and validation')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate',
                        type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--model', dest='model_path',
                        default=None, metavar='PTFILE',
                        help='initial learning rate')
    parser.add_argument('--cpu', action='store_true',
                        help='run on cpu')
    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    use_gpu = torch.cuda.is_available() and not args.cpu
    device = 'cuda' if use_gpu else 'cpu'

    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.set_deterministic(True)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_dir = output_path.joinpath('models')
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_path.joinpath('log/foe')
    tb_writer = SummaryWriter(log_dir=log_dir)

    # load fingerprint data
    good_path = (base_path.joinpath('Good'), True)
    fp_paths = [good_path]
    if not args.without_bad:
        bad_path = base_path.joinpath('Bad')
        fp_paths.append((bad_path, False))
        
    tset, vset, model = init_model_and_datasets(fp_paths, args.n_classes,
                                                args.patch_size,
                                                args.n_folds, FOLD_IDX,
                                                args.model_path,
                                                args.no_shuffle,
                                                args.verbose)

    # Data augmentation parameters
    if not args.no_hflip:
        tset.set_hflip(True)
    if args.delta_r > 0.0:
        tset.set_delta_r(args.delta_r)

    # initialize loader, model, and other training machinary
    train_loader = DataLoader(tset,
                              batch_size=batch_size,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              pin_memory=use_gpu)
    val_loader = DataLoader(vset,
                            batch_size=VALIDATION_BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=True,
                            pin_memory=use_gpu)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [int(0.4 * n_epochs),
                                                int(0.75 * n_epochs)],
                                               gamma=0.1)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print("""Number of classes: {}
Patch size: {}
Batch size: {}
Rotation delta: {:.1f}°
Training set size: {}
Validation set size: {}""".format(model.n_classes, model.patch_size,
                                  batch_size,
                                  args.delta_r * 180.0 / np.pi,
                                  n_train, n_val))

    # run training + validation for each epoch
    for e in range(n_epochs):
        train_loss = train_model(model, train_loader,
                                 optimizer, criterion, device)
        
        val_loss, val_acc, val_results = validate_model(model, val_loader,
                                                        criterion, device)
        scheduler.step()

        # report epoch results
        val_rmse = rmse_from_results(val_results, vset)
        val_rmse_all = val_rmse[0] * 180.0 / np.pi
        val_rmse_good = val_rmse[1] * 180.0 / np.pi
        val_rmse_bad = val_rmse[2] * 180.0 / np.pi
        tb_writer.add_scalar('train_loss', train_loss, e)
        tb_writer.add_scalar('val_loss', val_loss, e)
        tb_writer.add_scalar('val_acc', val_acc, e)
        tb_writer.add_scalar('val_rmse_good', val_rmse_good, e)
        tb_writer.add_scalar('val_rmse_bad', val_rmse_bad, e)
        eprint('Epoch {}/{}: train loss = {:.4f}'
               ' validation loss / acc / rmse (all, good, bad) = '
               '{:.4f} / {:.1f}% / '
               '({:.1f}°, {:.1f}°, {:.1f}°)'.format(e+1, n_epochs, train_loss,
                                                  val_loss,
                                                  val_acc * 100.0,
                                                  val_rmse_all,
                                                    val_rmse_good,
                                                    val_rmse_bad))

        if e % 100 == 99:
            save_checkpoint(model_dir, model, batch_size, e,
                            tset.images, val_results, args.verbose)

    # save model also at the end
    model_path = save_checkpoint(model_dir, model, batch_size, n_epochs,
                                 tset.images, val_results, verbose=True)
    
    tb_writer.close()
