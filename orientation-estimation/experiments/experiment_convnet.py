# %%
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from foe_fp_image_dataset import FOEFPImageDataset
from foe_model_convnet import FOEConvNet
import utils

num_folds = 5
use_cpu = False

num_epochs = 10000
batch_size = 2
num_workers = 4
num_synth = 0

learning_rate = 10**-2
gamma = 10**-4
power = 0.75
weight_decay = 5*10**-6
momentum = 0.5

n_class = 32    # 32/256 - 0 for sin/cos regression
n_disc = 8      # 8/1
disc_method = 'eq_len'
encoding_method = 'one_hot'

if n_class == 0:
    K = 2
elif encoding_method == 'one_hot':
    K = n_disc * n_class
elif encoding_method == 'ordinal':
    K = n_disc * (n_class-1)
elif encoding_method == 'cyclic':
    K = n_disc * n_class / 2

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'
print(device)

base_path_bad = '../../datasets/foe/Bad'
base_path_good = '../../datasets/foe/Good'
base_path_synth = '../../datasets/foe/Synth'

def split_database(base_path, num_folds):
    index_path = os.path.join(base_path, 'index.txt')
    with open(index_path, 'r') as fin:
        fp_ids = [line.split('.')[0] for line in fin.readlines()[1:]]
    random.shuffle(fp_ids)
    parts = np.array_split(fp_ids, num_folds)
    return parts

parts_bad = split_database(base_path_bad, num_folds)
parts_good = split_database(base_path_good, num_folds)
parts_synth = split_database(base_path_synth, 1)

# %%
def construct_lr_lambda(gamma, power):
    def lr_lambda(epoch):
        return (1 + gamma * epoch) ** (-power)
    return lr_lambda

for fold in range(num_folds):
    fp_ids_bad_val = parts_bad[fold]
    fp_ids_bad_tra = np.append(parts_bad[:fold], parts_bad[fold+1:])
    fp_ids_good_val = parts_good[fold]
    fp_ids_good_tra = np.append(parts_good[:fold], parts_good[fold+1:])
    fp_ids_synth_tra = parts_synth[0][0:num_synth]
    foe_img_ds_val = FOEFPImageDataset([base_path_bad, base_path_good],
                                       [fp_ids_bad_val, fp_ids_good_val])
    foe_img_ds_tra = FOEFPImageDataset([base_path_bad, base_path_good,
                                        base_path_synth],
                                       [fp_ids_bad_tra, fp_ids_good_tra,
                                        fp_ids_synth_tra])
    #foe_img_ds_val.set_hflip()
    #foe_img_ds_tra.set_hflip()
    #foe_img_ds_val.set_rotate()
    #foe_img_ds_tra.set_rotate()
    foe_img_dl_val = torch.utils.data.DataLoader(foe_img_ds_val,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True,
                                                 pin_memory=use_gpu)
    foe_img_dl_tra = torch.utils.data.DataLoader(foe_img_ds_tra,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True,
                                                 pin_memory=use_gpu)
    if n_class == 0:
        discs = [0]
    else:
        discs = utils.discretize_orientation(disc_method, n_class, n_disc,
                                             sample=None)
    
    model = FOEConvNet(out_len=K, final_relu=(n_class != 0))
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    lr_lambda = construct_lr_lambda(gamma, power)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lr_lambda)
    for e in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xi, orientations, mask, fp_type, index in foe_img_dl_tra:
            x = xi.to(device)
            if n_class == 0:
                yi = utils.angle_to_sincos(orientations)
            else:
                yi = None
                for disc in discs:
                    if yi is None:
                        yi = utils.encode_angle(orientations, encoding_method,
                                                disc)
                    else:
                        yi = np.append(yi, utils.encode_angle(orientations,
                                                              encoding_method,
                                                              disc), 1)
            y = torch.from_numpy(yi).to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            y_out = model(x)
            # loss = F.mse_loss(y_out, y, reduction='none')
            # loss = torch.sum(loss, dim=1) * mask
            loss = 0
            for i in range(n_disc):
                if encoding_method == 'one_hot':
                    loss += F.cross_entropy(y_out[:,i*n_class:(i+1)*n_class],
                                            y[:,i*n_class:(i+1)*n_class],
                                            reduction='none')
                elif encoding_method == 'ordinal':
                    pass
                elif encoding_method == 'cyclic':
                    pass
            loss = loss * mask
            loss = torch.sum(loss) / torch.sum(mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
        print(e, total_loss / len(foe_img_dl_tra), scheduler.get_last_lr())

        # if e % 20 == 0:
        #     if e % 100 == 0:
        #         path = '../results/model_'+str(e).zfill(3)+'.pt'
        #         torch.save({'mstate': model.state_dict()}, path)
        #         print('Saved model to {}'.format(path))
        #     with torch.no_grad():
        #         model.eval()
        #         total_loss = 0.0
        #         rmse_bad_tra = []
        #         rmse_good_tra = []
        #         for (xi, orientations, mask, fp_type,index) in foe_img_dl_tra:
        #             x = xi.to(device)
        #             y = yi.to(device)
        #             y_out = model(x)
        #             for fpt in fp_type:
        #                 if fpt == 'Bad':
        #                     rmse_bad_tra.append(utils.calc_rmse2(y, y_out, mask, n_class))
        #                 elif fpt == 'Good':
        #                     rmse_good_tra.append(utils.calc_rmse2(y, y_out, mask, n_class))
        #         print(np.mean(rmse_bad_tra), np.mean(rmse_good_tra))

        #         total_loss = 0.0
        #         rmse_bad_val = []
        #         rmse_good_val = []
        #         for (xi, orientations, mask, fp_type, index) in foe_img_dl_val:
        #             x = xi.to(device)
        #             if n_class == 0:
        #                 yi = utils.angle_to_sincos(orientations)
        #             else:
        #                 yi = None
        #                 for disc in discs:
        #                     if yi is None:
        #                         yi = utils.encode_angle(orientations, encoding_method,
        #                                                 disc)
        #                     else:
        #                         yi = np.append(yi, utils.encode_angle(orientations,
        #                                                             encoding_method,
        #                                                             disc), 1)
        #             y = torch.from_numpy(yi).to(device)
        #             y_out = model(x)
        #             yd = y.cpu().detach().numpy()
        #             y_outd = y_out.cpu().detach().numpy()
        #             maskd = mask.cpu().detach().numpy()
        #             degrees = np.arctan2(yd[:, 0], yd[:, 1]) / np.pi * 180
        #             degrees_out = np.arctan2(y_outd[:, 0],
        #                                     y_outd[:, 1]) / np.pi * 180
        #             degrees_se = np.sum(np.power(degrees - degrees_out, 2) *
        #                                 maskd)
        #             degrees_mse = degrees_se / np.sum(maskd)
        #             for fpt in fp_type:
        #                 if fpt == 'Bad':
        #                     # rmse_bad_val.append(calc_rmse(y, y_out, mask))
        #                     rmse_bad_val.append(utils.calc_rmse2(y, y_out, mask, n_classes))
        #                 elif fpt == 'Good':
        #                     # rmse_good_val.append(calc_rmse(y, y_out, mask))
        #                     rmse_good_val.append(utils.calc_rmse2(y, y_out, mask, n_classes))
        #         print(np.mean(rmse_bad_val), np.mean(rmse_good_val))