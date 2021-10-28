from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from foe_ae import FOE_AE
from foe_mlp import FOE_MLP
from foe_cnn import FOE_CNN

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


def train_autoencoder(models_dir, fold_id, device, ae_train_loader,
                      val_loader_gd, val_loader_bd, args):
    if not args.ae_continue_training:
        ae_name = 'foe_ae_in{:03d}_out{:03d}_e{:04d}_s{}_f{}.pt'
        ae_name = ae_name.format(args.patch_size, args.encoded_space_dim,
                                 args.ae_num_epochs, args.split_id, fold_id)
        ae_path = models_dir.joinpath(ae_name)
        if ae_path.exists():
            ae, val_results = FOE_AE.load_checkpoint(ae_path, device)
            done_epochs = args.ae_num_epochs
        else:
            ae = FOE_AE(args.patch_size, args.encoded_space_dim, device)
            done_epochs = 0
    else:
        ae_name = 'foe_ae_in{:03d}_out{:03d}_e{}_s{}v_f{}.pt'
        ae_name = ae_name.format(args.patch_size, args.encoded_space_dim,
                                 '*', args.split_id, fold_id)
        file_list = list(models_dir.glob(ae_name))
        if file_list:
            ae_path = file_list[-1]
            ae, val_results = FOE_AE.load_checkpoint(ae_path, device)
            done_epochs = int(str(ae_path).split('_')[-3][1:])
            args.ae_num_epochs = args.ae_num_epochs + done_epochs
        else:
            ae = FOE_AE(args.patch_size, args.encoded_space_dim, device)
            done_epochs = 0

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=args.ae_learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [int(0.4 * args.ae_num_epochs),
                                                int(0.7 * args.ae_num_epochs)],
                                               gamma=0.1)
    ae_metrics = {'train_loss': [], 'val_loss_gd': [], 'val_loss_bd': []}
    epoch = done_epochs
    for epoch in range(done_epochs+1, args.ae_num_epochs+1):
        train_loss = ae.train_epoch(ae_train_loader, loss_fn, optimizer)
        val_loss_gd = ae.val_epoch(val_loader_gd, loss_fn)
        val_loss_bd = ae.val_epoch(val_loader_bd, loss_fn)
        scheduler.step()

        print('EPOCH {}/{}\t'
              'Losses for train/good val/bad val: {:.4f} / {:.4f} / {:.4f}'
              .format(epoch, args.ae_num_epochs, train_loss, val_loss_gd,
                      val_loss_bd))

        ae_metrics['train_loss'].append(train_loss)
        ae_metrics['val_loss_gd'].append(val_loss_gd)
        ae_metrics['val_loss_bd'].append(val_loss_bd)

        if epoch == args.ae_num_epochs:
            ae_name = 'foe_ae_in{:03d}_out{:03d}_e{:04d}_s{}_f{}.pt'
            ae_name = ae_name.format(args.patch_size, args.encoded_space_dim,
                                     epoch, args.split_id, fold_id)
            ae_path = models_dir.joinpath(ae_name)
            ae.save_checkpoint(ae_path)

            plot_loss(ae_metrics)
    ae.plot_outputs(val_loader_gd.dataset, 5)
    ae.plot_outputs(val_loader_bd.dataset, 5)

    return ae


def initialize_mlp(models_dir, fold_id, device, args):
    if not args.continue_training:
        mlp_name = 'foe_mlp_in{:03d}_out{:03d}_e{:04d}_s{}_f{}.pt'
        mlp_name = mlp_name.format(args.encoded_space_dim, args.num_classes,
                                   args.num_epochs, args.split_id, fold_id)
        mlp_path = models_dir.joinpath(mlp_name)

        if mlp_path.exists():
            mlp, results = FOE_MLP.load_checkpoint(mlp_path, device)
            done_epochs = args.num_epochs
            mlp_path = None
        else:
            mlp = FOE_MLP(args.encoded_space_dim, args.num_classes, device)
            done_epochs = 0
    else:
        mlp_name = 'foe_mlp_in{:03d}_out{:03d}_e{}_s{}_f{}.pt'
        mlp_name = mlp_name.format(args.encoded_space_dim, args.num_classes,
                                   '*', args.split_id, fold_id)
        file_list = list(models_dir.glob(mlp_name))
        if file_list:
            mlp_path = file_list[-1]
            mlp, results = FOE_MLP.load_checkpoint(mlp_path, device)
            done_epochs = int(str(mlp_path).split('_')[-3][1:])
            args.num_epochs = args.num_epochs + done_epochs
        else:
            mlp = FOE_MLP(args.encoded_space_dim, args.num_classes)
            done_epochs = 0

        mlp_name = 'foe_mlp_in{:03d}_out{:03d}_e{:04d}_s{}_f{}.pt'
        mlp_name = mlp_name.format(args.encoded_space_dim, args.num_classes,
                                   args.num_epochs, args.split_id, fold_id)
        mlp_path = models_dir.joinpath(mlp_name)

    return mlp, done_epochs, args.num_epochs, mlp_path, results


def initialize_cnn(models_dir, fold_id, device, args):
    if not args.continue_training:
        cnn_name = 'foe_cnn_in{:03d}_out{:03d}_e{:04d}_s{}_f{}.pt'
        cnn_name = cnn_name.format(args.patch_size, args.num_classes,
                                   args.num_epochs, args.split_id, fold_id)
        cnn_path = models_dir.joinpath(cnn_name)

        if cnn_path.exists():
            cnn, results = FOE_CNN.load_checkpoint(cnn_path, device)
            done_epochs = args.num_epochs
            cnn_path = None
        else:
            cnn = FOE_CNN(args.patch_size, args.num_classes, device)
            done_epochs = 0
    else:
        cnn_name = 'foe_cnn_in{:03d}_out{:03d}_e{}_s{}_f{}.pt'
        cnn_name = cnn_name.format(args.patch_size, args.num_classes,
                                   '*', args.split_id, fold_id)
        file_list = list(models_dir.glob(cnn_name))
        if file_list:
            cnn_path = file_list[-1]
            cnn, results = FOE_CNN.load_checkpoint(cnn_path, device)
            done_epochs = int(str(cnn_path).split('_')[-3][1:])
            args.num_epochs = args.num_epochs + done_epochs
        else:
            cnn = FOE_CNN(args.patch_size, args.num_classes, device)
            done_epochs = 0

        cnn_name = 'foe_cnn_in{:03d}_out{:03d}_e{:04d}_s{}_f{}.pt'
        cnn_name = cnn_name.format(args.patch_size,
                                   args.num_classes, args.num_epochs,
                                   args.split_id, fold_id)
        cnn_path = models_dir.joinpath(cnn_name)

    return cnn, done_epochs, args.num_epochs, cnn_path, results


def train(model, train_loader, val_loader_gd, val_loader_bd, loss_fn,
          optimizer, scheduler, results, metrics):
    train_loss = model.train_epoch(train_loader, loss_fn, optimizer)
    val_loss_gd = model.val_epoch(val_loader_gd, loss_fn, results)
    val_loss_bd = model.val_epoch(val_loader_bd, loss_fn, results)
    scheduler.step()

    metrics['train_loss'].append(train_loss)
    metrics['val_loss_gd'].append(val_loss_gd)
    metrics['val_loss_bd'].append(val_loss_bd)

    return train_loss, val_loss_gd, val_loss_bd
