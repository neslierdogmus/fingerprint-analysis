# %%
import random
import numpy as np
from skimage.feature import peak_local_max

import torch
from torchvision.utils import save_image

from fmd_fp_image_dataset import FMDFPImageDataset
from fmd_model_convnet import FMDConvNet

num_folds = 5
use_cpu = False

n_classes = 1
num_epochs = 101
batch_size = 4
num_workers = 2
num_synth = 0

learning_rate = 10**-2
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


def calc_accuracy(y_out, gt_min_yx, Td):
    Recall = []
    Precision = []
    F1 = []
    for i in range(y_out.shape[0]):
        yi_out = y_out[i, 0, :, :]
        output = yi_out.cpu().detach().numpy()
        output[output < 0] = 0
        est_yx = peak_local_max(output, min_distance=3)
        gt_m_yx = np.array(gt_min_yx)[:, i, :].transpose()
        gt_m_yx = gt_m_yx[gt_m_yx[:, 0] > 0, :]
        M = len(est_yx)
        N = len(gt_m_yx)
        dist = np.zeros((M, N))
        for m in range(M):
            for n in range(N):
                dist[m, n] = np.linalg.norm(est_yx[m] - gt_m_yx[n])
        dist_tmp = np.array(dist)
        match = np.zeros_like(dist_tmp)
        while np.sum(match) < M and np.sum(match) < N:
            index = np.unravel_index(dist_tmp.argmin(), dist_tmp.shape)
            if dist_tmp[index[0], index[1]] > Td:
                break
            if (np.sum(match[index[0], :]) == 0 and
               np.sum(match[:, index[1]]) == 0):
                match[index[0], index[1]] = 1
            dist_tmp[index[0], index[1]] = 9999
        TP = np.sum(match)
        Recall.append(TP / N)
        if M > 0:
            Precision.append(TP / M)
        else:
            Precision.append(0)
        F1.append(2 * TP / (M + N))
    return Recall, Precision, F1


print(device)

# %%
for fold in range(1):
    model = FMDConvNet(8)
    # chk_dict = torch.load('../results/model_100.pt')
    # mstate = chk_dict['mstate']
    # model.load_state_dict(mstate)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #                             momentum=momentum, weight_decay=weight_decay)
    # lr_lambda = construct_lr_lambda(gamma, power)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                               lr_lambda=lr_lambda)
    loss_fnc = torch.nn.functional.binary_cross_entropy_with_logits

    fp_ids_val = splits[fold]
    fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

    fmd_img_ds_val = FMDFPImageDataset(base_path, fp_ids_val, n_classes)
    fmd_img_ds_tra = FMDFPImageDataset(base_path, fp_ids_tra, n_classes)

    # fmd_img_ds_val.set_resize()
    fmd_img_ds_tra.set_hflip()
    fmd_img_ds_tra.set_rotate()
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
    torch.autograd.set_detect_anomaly(True)
    for e in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xi, yi, gt_yx, quality, index in fmd_img_dl_tra:
            mask = torch.logical_or((yi > 0), (torch.rand(yi.shape) > 0.2))
            mask = mask.float().to(device)
            xi = xi.to(device)
            yi = yi.to(device)
            optimizer.zero_grad()
            y_out = model(xi)
            loss = loss_fnc(y_out, yi, reduction='none')
            loss = loss * mask
            loss = torch.sum(loss) / torch.sum(mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # scheduler.step()
        # print(e, total_loss / len(fmd_img_dl_tra), scheduler.get_last_lr())
        print(e, total_loss / len(fmd_img_dl_tra))

        save_image(xi[0], '{}_xi.png'.format(str(e).zfill(2)))
        save_image(yi[0], '{}_yi.png'.format(str(e).zfill(2)))
        save_image(y_out[0], '{}_yo.png'.format(str(e).zfill(2)))

        if e % 10 == 0:
            path = '../../fmd/results/model_'+str(e).zfill(3)+'.pt'
            torch.save({'mstate': model.state_dict()}, path)
            print('Saved model to {}'.format(path))
            if e > 0:
                with torch.no_grad():
                    model.eval()
                    total_loss = 0.0
                    Recall_tra = []
                    Precision_tra = []
                    F1_tra = []
                    Quality_tra = []
                    for xi, yi, gt_yx, quality, index in fmd_img_dl_val:
                        xi = xi.to(device)
                        y_out = model(xi)
                        Recall, Precision, F1 = calc_accuracy(y_out, gt_yx, 15)
                        Recall_tra.extend(Recall)
                        Precision_tra.extend(Precision)
                        F1_tra.extend(F1)
                        quality_np = quality.cpu().detach().numpy()
                        Quality_tra.extend(list(quality_np))
                    print(np.mean(Recall_tra), np.mean(Precision_tra),
                          np.mean(F1_tra))

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
