# %%
import os
import random

import numpy as np
import torch


from foe_fp_image_dataset import FOEFPImageDataset
from foe_convnet import FOEConvNet

num_folds = 5
use_cpu = False

n_classes = 1
num_epochs = 10000
batch_size = 1
num_workers = 4
num_synth = 0

learning_rate = 10**-2
gamma = 10**-3
power = 0.75
weight_decay = 5*10**-6
momentum = 0.5

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'

print(os.getcwd())

base_path_bad = './foe/dataset/Bad'
base_path_good = './foe/dataset/Good'
base_path_synth = './foe/dataset/Synth'


def split_database(base_path, num_folds):
    index_path = os.path.join(base_path, 'index.txt')
    with open(index_path, 'r') as fin:
        fp_ids = [line.split('.')[0] for line in fin.readlines()[1:]]
    random.shuffle(fp_ids)
    parts = np.array_split(fp_ids, num_folds)
    return parts


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output_exp, output_pre):
        # Compute the squared norm of the difference vector
        diff = output_pre - output_exp
        squared_norm = torch.sum(diff ** 2, dim=0)

        # Average the squared norms over the MxN dimensions
        loss = torch.mean(squared_norm)

        return loss


def construct_lr_lambda(gamma, power):
    def lr_lambda(epoch):
        return (1 + gamma * epoch) ** (-power)
    return lr_lambda


def delta_sqr(angle0_in_radians, angle1_in_radians):
    d = np.fabs(angle0_in_radians - angle1_in_radians)
    if d > np.pi/2.0:
        d = np.pi - d
    return d**2


def calc_rmse(output_exp, output_pre, mask):
    output_exp = output_exp.cpu().detach().numpy()
    radians_exp = np.arctan2(output_exp[:, 0], output_exp[:, 1]) / 2
    radians_exp = np.where(radians_exp < 0, radians_exp + np.pi, radians_exp)
    degrees_exp = radians_exp / np.pi * 180

    output_pre = output_pre.cpu().detach().numpy()
    radians_pre = np.arctan2(output_pre[:, 0], output_pre[:, 1]) / 2
    radians_pre = np.where(radians_pre < 0, radians_pre + np.pi, radians_pre)
    degrees_pre = radians_pre / np.pi * 180

    degrees_diff = np.abs(degrees_exp - degrees_pre)
    degrees_diff = np.where(degrees_diff > 90, 180-degrees_diff, degrees_diff)

    mask = mask.cpu().detach().numpy()
    degrees_se = np.sum(np.power(degrees_diff, 2) * mask)
    degrees_rmse = np.sqrt(degrees_se / np.sum(mask))

    return degrees_rmse


parts_bad = split_database(base_path_bad, num_folds)
parts_good = split_database(base_path_good, num_folds)
parts_synth = split_database(base_path_synth, 1)

print(device)

# %%
for fold in range(num_folds):
    model = FOEConvNet()
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    lr_lambda = construct_lr_lambda(gamma, power)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lr_lambda)

    fp_ids_bad_val = parts_bad[fold]
    fp_ids_bad_tra = np.append(parts_bad[:fold], parts_bad[fold+1:])
    fp_ids_good_val = parts_good[fold]
    fp_ids_good_tra = np.append(parts_good[:fold], parts_good[fold+1:])
    fp_ids_synth_tra = parts_synth[0][0:num_synth]
    foe_img_ds_val = FOEFPImageDataset([base_path_bad, base_path_good],
                                       [fp_ids_bad_val, fp_ids_good_val],
                                       n_classes)
    foe_img_ds_tra = FOEFPImageDataset([base_path_bad, base_path_good,
                                        base_path_synth],
                                       [fp_ids_bad_tra, fp_ids_good_tra,
                                        fp_ids_synth_tra],
                                       n_classes)
    foe_img_ds_val.set_hflip()
    foe_img_ds_tra.set_hflip()
    foe_img_ds_val.set_rotate()
    foe_img_ds_tra.set_rotate()
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

    for e in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xi, yi, mask, orientations, fp_type, index in foe_img_dl_tra:
            x = xi.to(device)
            y = yi.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            y_out = model(x)
            loss = torch.nn.functional.mse_loss(y, y_out, reduction='none')
            loss = torch.sum(loss, dim=1) * mask
            loss = torch.sum(loss) / torch.sum(mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
        print(e, total_loss / len(foe_img_dl_tra.dataset) * batch_size,
              scheduler.get_last_lr())

        if e % 20 == 0:
            if e % 100 == 0:
                path = './foe/results/model_'+str(e).zfill(3)+'.pt'
                torch.save({'mstate': model.state_dict()}, path)
                print('Saved model to {}'.format(path))
            with torch.no_grad():
                model.eval()
                total_loss = 0.0
                rmse_bad_tra = []
                rmse_good_tra = []
                for (xi, yi, mask, orientations, fp_type,
                     index) in foe_img_dl_tra:
                    x = xi.to(device)
                    y = yi.to(device)
                    y_out = model(x)
                    for fpt in fp_type:
                        if fpt == 'Bad':
                            rmse_bad_tra.append(calc_rmse(y, y_out, mask))
                        elif fpt == 'Good':
                            rmse_good_tra.append(calc_rmse(y, y_out, mask))
                print(np.mean(rmse_bad_tra), np.mean(rmse_good_tra))

                rmse_bad_val = []
                rmse_good_val = []
                for (xi, yi, mask, orientations, fp_type,
                     index) in foe_img_dl_val:
                    x = xi.to(device)
                    y = yi.to(device)
                    y_out = model(x)
                    yd = y.cpu().detach().numpy()
                    y_outd = y_out.cpu().detach().numpy()
                    maskd = mask.cpu().detach().numpy()
                    degrees = np.arctan2(yd[:, 0], yd[:, 1]) / np.pi * 180
                    degrees_out = np.arctan2(y_outd[:, 0],
                                             y_outd[:, 1]) / np.pi * 180
                    degrees_se = np.sum(np.power(degrees - degrees_out, 2) *
                                        maskd)
                    degrees_mse = degrees_se / np.sum(maskd)
                    for fpt in fp_type:
                        if fpt == 'Bad':
                            rmse_bad_val.append(calc_rmse(y, y_out, mask))
                        elif fpt == 'Good':
                            rmse_good_val.append(calc_rmse(y, y_out, mask))
                print(np.mean(rmse_bad_val), np.mean(rmse_good_val))

# %%
