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
eval_step = 10
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

base_path_bad = '../../datasets/foe/Bad'
base_path_good = '../../datasets/foe/Good'
base_path_synth = '../../datasets/foe/Synth'

parts_bad = utils.split_database(base_path_bad, num_folds)
parts_good = utils.split_database(base_path_good, num_folds)
parts_synth = utils.split_database(base_path_synth, 1)

# %%
all_results = []
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
    disc_names = ['sin_cos']
    for params in [['eq_len', 256, 1], ['eq_len', 128, 2], ['eq_len', 64, 4],
                   ['eq_len', 32, 8], ['eq_len', 16, 16], ['eq_prob', 128, 2],
                   ['eq_prob', 64, 4], ['eq_prob', 32, 8], ['eq_prob', 16, 16],
                   ['k_means', 32, 8], ['k_means', 16, 16]]:
        discs = utils.discretize_orientation(params[0], params[1], params[2],
                                             foe_img_ds_tra)
        discs_all.append(discs)
        img_name = "f"+str(fold)+"_"+"_".join([str(p) for p in params])
        disc_names.append(img_name)
        utils.view_discs(discs, img_name, foe_img_ds_tra)
        utils.view_codes(discs, img_name)

    fold_results = []
    for discs, disc_name in zip(discs_all, disc_names):
        for encod_met in ['one_hot', 'ordinal', 'circular']:
            if not discs:
                K = 2
                loss_fnc = F.mse_loss
                n_disc = 1
                code_len = 2
                exp_name = disc_name
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
                exp_name = disc_name + "_" + encod_met

            print('>>>>> Experiment:', exp_name)

            model = FOEConvNet(out_len=K)
            model = model.to(device)
            print(sum(p.numel() for p in model.parameters()))
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
            lr_lambda = utils.construct_lr_lambda(gamma, power)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            loss_list = []
            result_list = []
            for e in range(num_epochs):
                model.train()
                e_loss = []
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
                    e_loss.append(loss.item())
                    scheduler.step()

                loss_list.append(np.mean(e_loss))
                end = ' ' if e % eval_step == 0 else '\n'
                print(e, loss_list[-1], scheduler.get_last_lr(), end=end)

                if e % eval_step == 0:
                    # if e % 100 == 0:
                    #     path = '../results/model_'+str(e).zfill(3)+'.pt'
                    #     torch.save({'mstate': model.state_dict()}, path)
                    #     print('Saved model to {}'.format(path))
                    with torch.no_grad():
                        model.eval()
                        dls = [foe_img_dl_tra, foe_img_dl_val]
                        total_loss = 0.0
                        rmse_g = [[], []]
                        rmse_b = [[], []]
                        rmse2_g = [[], []]
                        rmse2_b = [[], []]
                        macc_g = [[], []]
                        macc_b = [[], []]
                        for d in range(len(dls)):
                            for x, oris, mask, fp_type, ind in dls[d]:
                                x = x.to(device)
                                yo = model(x)
                                mask_np = mask.cpu().detach().numpy()
                                new_macc = [0] * batch_size
                                new_rmse2 = [0] * batch_size
                                if not discs:
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
                                        new_rmse2 = utils.calc_rmse(oris,
                                                                    ests2,
                                                                    mask_np)
                                new_rmse = utils.calc_rmse(oris, ests, mask_np)
                                for er, er2, ac, t in zip(new_rmse, new_rmse2,
                                                          new_macc, fp_type):
                                    if t == 'Good':
                                        rmse_g[d] = np.append(rmse_g[d], [er])
                                        rmse2_g[d] = np.append(rmse2_g[d],
                                                               [er2])
                                        macc_g[d] = np.append(macc_g[d], [ac])
                                    elif t == 'Bad':
                                        rmse_b[d] = np.append(rmse_b[d], [er])
                                        rmse2_b[d] = np.append(rmse2_g[d],
                                                               [er2])
                                        macc_b[d] = np.append(macc_b[d], [ac])
                    res_ls = [rmse_g, rmse_b, rmse2_g, rmse2_b, macc_g, macc_b]
                    result_list.append([np.mean(res[d]) for d in range(2)
                                       for res in res_ls])
                    print(result_list[-1], end=' ')
                    print()
            fold_results.append(result_list)
            if not discs:
                break
    all_results.append(fold_results)
#plot all results