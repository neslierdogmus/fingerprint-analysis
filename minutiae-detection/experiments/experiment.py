# %%
import random

import numpy as np
import torch
from torchvision.utils import save_image

from fmd_fp_image_dataset import FMDFPImageDataset
from fmd_model_convnet import FMDConvNet

num_folds = 5
use_cpu = False

n_classes = 1
num_epochs = 21
batch_size = 4
num_workers = 4
num_synth = 0

learning_rate = 2 * 10**-4
gamma = 10**-3
power = 0.75
weight_decay = 0
momentum = 0

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'

base_path = '../../datasets/fmd'

fp_ids = list(range(400))
random.shuffle(fp_ids)
splits = np.array(np.array_split(fp_ids, num_folds))


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


print(device)

# %%
for fold in range(1):
    model = FMDConvNet()
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #                             momentum=momentum, weight_decay=weight_decay)
    # lr_lambda = construct_lr_lambda(gamma, power)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                               lr_lambda=lr_lambda)

    fp_ids_val = splits[fold]
    fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

    fmd_img_ds_val = FMDFPImageDataset(base_path, fp_ids_val, n_classes)
    fmd_img_ds_tra = FMDFPImageDataset(base_path, fp_ids_tra, n_classes)

    fmd_img_ds_val.set_resize()
    fmd_img_ds_tra.set_hflip()
    # fmd_img_ds_tra.set_rotate()
    fmd_img_ds_tra.set_resize()

    fmd_img_dl_val = torch.utils.data.DataLoader(fmd_img_ds_val,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True,
                                                 pin_memory=use_gpu)
    fmd_img_dl_tra = torch.utils.data.DataLoader(fmd_img_ds_tra,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True,
                                                 pin_memory=use_gpu)

    for e in range(num_epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for xi, yi, quality, index in fmd_img_dl_tra:
            count += 1
            x = xi.to(device)
            y = yi.to(device)
            mask = (y > 0).float().to(device)
            optimizer.zero_grad()
            y_out = model(x)
            loss = torch.nn.functional.binary_cross_entropy(y, y_out)
            # loss = torch.nn.functional.mse_loss(y, y_out, reduction='none')
            # loss = loss * mask
            # loss = torch.sum(loss) / torch.sum(mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # scheduler.step()
        # print(e, total_loss / len(fmd_img_dl_tra), scheduler.get_last_lr())
        print(e, total_loss / len(fmd_img_dl_tra))
        save_image(x[0], 'x.png')
        save_image(y[0], 'y.png')
        save_image(y_out[0], 'y_out.png')

        if e % 5 == 0:
            if e % 10 == 0:
                path = '../../fmd/results/model_'+str(e).zfill(3)+'.pt'
                torch.save({'mstate': model.state_dict()}, path)
                print('Saved model to {}'.format(path))
            # with torch.no_grad():
            #     model.eval()
            #     total_loss = 0.0
            #     rmse_bad_tra = []
            #     rmse_good_tra = []
            #     for xi, yi, quality, index in fmd_img_dl_tra:
            #         x = xi.to(device)
            #         y = yi.to(device)
            #         y_out = model(x)
            #         loss = torch.nn.functional.mse_loss(y, y_out)
            #         total_loss += loss.item()
            #         # for fpt in fp_type:
            #         #     if fpt == 'Bad':
            #         #         rmse_bad_tra.append(calc_rmse(y, y_out, mask))
            #         #     elif fpt == 'Good':
            #         #         rmse_good_tra.append(calc_rmse(y, y_out, mask))
            #     # print(np.mean(rmse_bad_tra), np.mean(rmse_good_tra))
            #     print(e, total_loss / len(fmd_img_dl_tra))

            #     total_loss = 0.0
            #     rmse_bad_val = []
            #     rmse_good_val = []
            #     for xi, yi, quality, index in fmd_img_dl_val:
            #         x = xi.to(device)
            #         y = yi.to(device)
            #         y_out = model(x)
            #         loss = torch.nn.functional.mse_loss(y, y_out)
            #         total_loss += loss.item()
            #         # for fpt in fp_type:
            #         #     if fpt == 'Bad':
            #         #         rmse_bad_val.append(calc_rmse(y, y_out, mask))
            #         #     elif fpt == 'Good':
            #         #         rmse_good_val.append(calc_rmse(y, y_out, mask))
            #     # print(np.mean(rmse_bad_val), np.mean(rmse_good_val))
            #     print(e, total_loss / len(fmd_img_dl_val))

# %%
