# %%
import numpy as np
import torch
import torch.nn.functional as F

from foe_fp_image_dataset import FOEFPImageDataset
from foe_model_convnet import FOEConvNet
import utils

num_folds = 5
use_cpu = False

num_epochs = 101
batch_size = 1
num_workers = 4
num_synth = 0

learning_rate = 10**-3
gamma = 10**-4
power = 0.75
weight_decay = 5*10**-6
momentum = 0.5

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'
print(device)

base_path_bad = './datasets/foe/Bad'
base_path_good = './datasets/foe/Good'
base_path_synth = './datasets/foe/Synth'

parts_bad = utils.split_database(base_path_bad, num_folds)
parts_good = utils.split_database(base_path_good, num_folds)
parts_synth = utils.split_database(base_path_synth, 1)

# %%
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
    # foe_img_ds_val.set_hflip()
    # foe_img_ds_tra.set_hflip()
    # foe_img_ds_val.set_rotate()
    # foe_img_ds_tra.set_rotate()

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

    discs_all = [[]]
    disc_methods = ['']
    for params in [['eq_len', 256, 1], ['eq_len', 128, 2], ['eq_len', 64, 4],
                   ['eq_len', 32, 8], ['eq_len', 16, 16], ['eq_prob', 128, 2],
                   ['eq_prob', 64, 4], ['eq_prob', 32, 8], ['eq_prob', 16, 16],
                   ['k_means', 32, 8], ['k_means', 16, 16]]:
        discs_all.append(utils.discretize_orientation(params[0], params[1],
                                                      params[2],
                                                      foe_img_ds_tra))
        disc_methods.append(params[0])

    for discs, disc_method in zip(discs_all, disc_methods):
        for encod_met in ['one_hot', 'ordinal', 'circular']:
            if not discs:
                K = 2
                loss_fnc = F.mse_loss
                n_disc = 1
                code_len = 2
            else:
                n_disc = len(discs)
                n_class = len(discs[0]) - 1
                if encod_met == 'one_hot':
                    loss_fnc = F.cross_entropy
                    def prob_fnc(probs): return F.softmax(probs, dim=1)
                    code_len = n_class
                else:
                    loss_fnc = F.binary_cross_entropy_with_logits
                    prob_fnc = F.sigmoid
                    if encod_met == 'ordinal':
                        code_len = n_class-1
                    elif encod_met == 'circular':
                        code_len = n_class // 2
                K = n_disc * code_len

            print('-' * 20)
            if discs:
                print('disc method', disc_method)
                print('n_class:', n_class)
                print('n_disc:', n_disc)
                print('encod method', encod_met)

            model = FOEConvNet(out_len=K)
            model = model.to(device)
            print(sum(p.numel() for p in model.parameters()))
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
            lr_lambda = utils.construct_lr_lambda(gamma, power)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            for e in range(num_epochs):
                model.train()
                total_loss = 0.0
                for x, oris, mask, fp_type, ind in foe_img_dl_tra:
                    x = x.to(device)
                    if not discs:
                        yt = utils.angle_to_sincos(oris)
                    else:
                        yt = utils.encode_angle(oris, encod_met, discs)
                    yt = yt.to(device)

                    mask = mask.to(device)
                    optimizer.zero_grad()
                    yo = model(x)
                    loss = 0
                    for i in range(n_disc):
                        loss += loss_fnc(yo[:, i*code_len:(i+1)*code_len],
                                         yt[:, i*code_len:(i+1)*code_len],
                                         reduction='none')
                    if len(loss.shape) == 4:
                        loss = torch.mean(loss, dim=1)
                    loss = loss * mask
                    loss = torch.sum(loss) / torch.sum(mask)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    scheduler.step()

                if e % 20 == 0:
                    print(e, total_loss / len(foe_img_dl_tra),
                          scheduler.get_last_lr(), end=' ')
                else:
                    print(e, total_loss / len(foe_img_dl_tra),
                          scheduler.get_last_lr())

                if e % 20 == 0:
                    # if e % 100 == 0:
                    #     path = '../results/model_'+str(e).zfill(3)+'.pt'
                    #     torch.save({'mstate': model.state_dict()}, path)
                    #     print('Saved model to {}'.format(path))
                    with torch.no_grad():
                        model.eval()
                        dls = [foe_img_dl_tra, foe_img_dl_val]
                        total_loss = 0.0
                        rmse_g = [[], []]
                        rmse_g2 = [[], []]
                        rmse_b = [[], []]
                        rmse_b2 = [[], []]
                        macc_g = [[], []]
                        macc_b = [[], []]
                        for d in range(len(dls)):
                            for x, oris, mask, fp_type, ind in dls[d]:
                                x = x.to(device)
                                yo = model(x)
                                mask_np = mask.cpu().detach().numpy()
                                ests2 = None
                                if not discs:
                                    new_macc = [0] * batch_size
                                    ests = utils.sincos_to_angle(yo)
                                else:
                                    yt = utils.encode_angle(oris,
                                                            encod_met,
                                                            discs)
                                    yt = yt.to(device)
                                    new_macc = utils.calc_class_acc(yo, yt,
                                                                    mask_np,
                                                                    code_len,
                                                                    n_disc,
                                                                    encod_met)
                                    ests = utils.decode_angle(yo, encod_met,
                                                              discs, prob_fnc,
                                                              regr='max')
                                    if encod_met == "one_hot":
                                        ests2 = utils.decode_angle(yo,
                                                                   encod_met,
                                                                   discs,
                                                                   prob_fnc,
                                                                   regr='exp')
                                new_rmse = utils.calc_rmse(oris, ests, mask_np)
                                for er, ac, t in zip(new_rmse, new_macc,
                                                     fp_type):
                                    if t == 'Good':
                                        rmse_g[d] = np.append(rmse_g[d], [er])
                                        macc_g[d] = np.append(macc_g[d], [ac])
                                    elif t == 'Bad':
                                        rmse_b[d] = np.append(rmse_b[d], [er])
                                        macc_b[d] = np.append(macc_b[d], [ac])
                                if ests2 is not None:
                                    new_rmse2 = utils.calc_rmse(oris, ests2,
                                                                mask_np)
                                    for er, t in zip(new_rmse2, fp_type):
                                        if t == 'Good':
                                            rmse_g2[d] = np.append(rmse_g[d],
                                                                   [er])
                                        elif t == 'Bad':
                                            rmse_b2[d] = np.append(rmse_g[d],
                                                                   [er])
                            print(np.mean(rmse_g[d]), np.mean(rmse_b[d]),
                                  end=' ')
                            print(np.mean(macc_g[d]), np.mean(macc_b[d]),
                                  end=' ')
                            if len(rmse_g2[0]) > 0:
                                print(np.mean(rmse_g2[d]), np.mean(rmse_b2[d]),
                                      end=' ')
                    print()
            if not discs:
                break
