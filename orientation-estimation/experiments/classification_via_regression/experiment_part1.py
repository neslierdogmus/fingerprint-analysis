# %%
import numpy as np
import torch
import torch.nn.functional as F

from foe_fp_image_dataset import FOEFPImageDataset
from foe_model_convnet import FOEConvNet
import utils

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

num_folds = 5
use_cpu = False

num_epochs = 201
eval_step = 10
batch_size = 1
num_workers = 4
num_synth = 0

lr = 10**-3
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

np.save('parts_bad.npy', parts_bad)
np.save('parts_good.npy', parts_good)
np.save('parts_synth.npy', parts_synth)

lr_coeff = [10, 10, 30, 30, 10, 30, 30, 3, 10, 10, 3, 10, 10, 1, 3, 3,
            10, 30, 30, 3, 10, 10, 3, 10, 10, 1, 3, 3, 1, 3, 3, 1, 3, 3]

# %%
all_loss = []
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

    fold_loss = []
    fold_results = []
    disc_names = ['f'+str(fold)+'_'+'sin_cos']
    lrc = iter(lr_coeff)
    for params in [['eq_len', 256, 1], ['eq_len', 128, 2], ['eq_len', 64, 4],
                   ['eq_len', 32, 8], ['eq_len', 16, 16], ['eq_prob', 128, 2],
                   ['eq_prob', 64, 4], ['eq_prob', 32, 8], ['eq_prob', 16, 16],
                   ['k_means', 32, 8], ['k_means', 16, 16]]:
        disc_name = 'f'+str(fold)+'_'+'_'.join([str(p) for p in params])
        disc_names.append(disc_name)
        discs = utils.discretize_orientation(params[0], params[1], params[2],
                                             foe_img_ds_tra)
        utils.view_discs(discs, disc_name, foe_img_ds_tra)
        utils.view_codes(discs, disc_name)
        np.save(disc_name+'.npy', discs)

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
                exp_name = disc_name + '_' + encod_met

            print('>>>>> Experiment:', exp_name)

            model = FOEConvNet(out_len=K)
            model = model.to(device)
            print(sum(p.numel() for p in model.parameters()))
            optimizer = torch.optim.SGD(model.parameters(), lr=lr*next(lrc),
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
                                    if encod_met == 'one_hot':
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
                                        rmse2_b[d] = np.append(rmse2_b[d],
                                                               [er2])
                                        macc_b[d] = np.append(macc_b[d], [ac])
                    res_ls = [rmse_g, rmse_b, rmse2_g, rmse2_b, macc_g, macc_b]
                    result_list.append([np.mean(res[d]) for d in range(2)
                                       for res in res_ls])
                    print(result_list[-1], end=' ')
                    print()
            fold_loss.append(loss_list)
            fold_results.append(result_list)
            ind_str = str(ind[-1].item())
            rmse_str = str(new_rmse[-1])
            print('Fingerprint {} with  RMSE {}'.format(ind_str, rmse_str))
            img_name = exp_name + '_' + ind_str + '_' + rmse_str + '.png'
            utils.view_ests(x[-1], oris[-1], ests[-1], mask[-1], img_name)
            if not discs:
                break
    all_loss.append(fold_loss)
    all_results.append(fold_results)
# %%
# plot all results
# al = np.array(all_loss)
# ar = np.array(all_results)
al = np.load('/home/finperprint-analysis/foe/results/part-1/best_10_401/al.npy')
ar = np.load('/home/finperprint-analysis/foe/results/part-1/best_10_401/ar.npy')
num_epochs = 401
eval_step = 10

# np.save('al.npy', al)
# np.save('ar.npy', ar)

al_mean = np.mean(al, axis=0)
ar_mean = np.mean(ar, axis=0)
al_std = np.std(al, axis=0)
ar_std = np.std(ar, axis=0)

x_data = np.arange(0, num_epochs, eval_step)+1

color = iter(cm.rainbow(np.linspace(0, 1, 11)))
c = next(color)
ind = -1


def plot_rmse(num, x_data, ar_mean, ar_std, ind, c, par=''):
    lines = ['--', '-', '-.', ':']

    plt.plot(x_data, ar_mean[ind, :, 2*num], lines[2*num], c=c, label='g '+par)
    plt.fill_between(x_data, ar_mean[ind, :, 2*num]-ar_std[ind, :, 2*num],
                     ar_mean[ind, :, 2*num]+ar_std[ind, :, 2*num], color=c,
                     alpha=.15)
    plt.plot(x_data, ar_mean[ind, :, 2*num+1], lines[2*num+1], c=c,
             label='b '+par)
    plt.fill_between(x_data, ar_mean[ind, :, 2*num+1]-ar_std[ind, :, 2*num+1],
                     ar_mean[ind, :, 2*num+1]+ar_std[ind, :, 2*num+1], color=c,
                     alpha=.15)
    plt.legend(bbox_to_anchor=(1.1, 1))


ar_mean = np.mean(ar, axis=0)
ar_std = np.std(ar, axis=0)
# NaN values for one fold of one experiment. Remove for the future runs!
ar_mean[22, :, :] = np.mean(ar[[0, 1, 2, 4], 22, :, :], axis=0)
ar_std[22, :, :] = np.std(ar[[0, 1, 2, 4], 22, :, :], axis=0)
ar_mean_min = np.min(ar_mean[:, :, 7], axis=1)
ar_mean_min_sincos = np.mean(ar_mean_min[0])
ar_mean_min_el = np.mean(ar_mean_min[1:16])
ar_mean_min_ep = np.mean(ar_mean_min[16:28])
ar_mean_min_km = np.mean(ar_mean_min[28:])
ar_mean_min_oh = np.mean(ar_mean_min[np.arange(1, 34, 3)])
ar_mean_min_oh_exp = np.mean(np.min(ar_mean[[1, 4, 7, 10, 13, 16, 19, 25, 28,
                                             31], :, 9], axis=1))
ar_mean_min_or = np.mean(ar_mean_min[np.arange(2, 34, 3)])
ar_mean_min_cr = np.mean(ar_mean_min[np.arange(3, 34, 3)])
ar_mean_100 = ar_mean[:, 11, 7]

fig, ax = plt.subplots(1, 1)
plt.ylabel('Mean RMSE')
plt.xlabel('Discretization Methods')
_ = ax.bar(np.arange(1, 7, 2), [ar_mean_min_el, ar_mean_min_ep,
                                ar_mean_min_km])
ax.set_xticks(np.arange(1, 7, 2))
ax.set_xticklabels(['eq-len', 'eq-prob', 'k-means'])
plt.ylim(10, 12)

fig, ax = plt.subplots(1, 1)
plt.ylabel('RMSE')
plt.xlabel('Encoding Methods')
_ = ax.bar(np.arange(1, 9, 2), [ar_mean_min_oh, ar_mean_min_oh_exp,
                                ar_mean_min_or, ar_mean_min_cr])
ax.set_xticks(np.arange(1, 9, 2))
ax.set_xticklabels(['one-hot-max', 'one-hot-exp', 'ordinal', 'circular'])
plt.ylim(10, 12)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.ylabel('RMSE')
plt.xlabel('Experiments')
_ = ax.bar(np.arange(1, 34), np.sort(ar_mean_100[1:]))
ax.set_xticks(np.arange(1, 34))
ax.set_xticklabels(np.argsort(ar_mean_100[1:])+1)
plt.ylim(9.5, 13)
plt.xlim(0, 35)

for disc_name in disc_names:
    for encod_met in ['one_hot', 'ordinal', 'circular']:
        ind += 1
        fig_name = '_'.join(disc_name.split('_')[1:3] + [encod_met])
        plt.figure(fig_name)
        if 'sin_cos' in disc_name:
            plt.title('Sin-Cos')
            plot_rmse(0, x_data, ar_mean, ar_std, ind, c)
            break
        else:
            plt.title(' '.join(disc_name.split('_')[1:3] + [encod_met]))
            par = ' '.join(disc_name.split('_')[-2:])
            if 'one_hot' in encod_met:
                plot_rmse(0, x_data, ar_mean, ar_std, ind, c, 'max ' + par)
                try:
                    c = next(color)
                except StopIteration:
                    color = iter(cm.rainbow(np.linspace(0, 1, 11)))
                    c = next(color)
                plot_rmse(1, x_data, ar_mean, ar_std, ind, c, 'exp ' + par)
            else:
                plot_rmse(0, x_data, ar_mean, ar_std, ind, c, par)
            try:
                c = next(color)
            except StopIteration:
                color = iter(cm.rainbow(np.linspace(0, 1, 11)))
                c = next(color)
plt.show()

# %% Plots for the journal paper

num_epochs = 101
num_measurements = 11
ar = np.load('/home/finperprint-analysis/foe/results/part-1/best_10_401/ar.npy')
ar_mean = np.mean(ar, axis=0)
ar_std = np.std(ar, axis=0)
ar_mean[22, :, :] = np.mean(ar[[0, 1, 2, 4], 22, :, :], axis=0)
ar_std[22, :, :] = np.std(ar[[0, 1, 2, 4], 22, :, :], axis=0)

ar_mean = ar_mean[:,:num_measurements,:]
ar_std = ar_std[:,:num_measurements,:]

ar_mean_2 = np.zeros((44,num_measurements,4))
ar_mean_2[0,:,:] = ar_mean[1,:,[0,1,6,7]].T
ar_mean_2[1,:,:] = ar_mean[1,:,[2,3,8,9]].T
ar_mean_2[2:4,:,:] = ar_mean[2:4,:,[0,1,6,7]]
ar_mean_2[4,:,:] = ar_mean[4,:,[0,1,6,7]].T
ar_mean_2[5,:,:] = ar_mean[4,:,[2,3,8,9]].T
ar_mean_2[6:8,:,:] = ar_mean[5:7,:,[0,1,6,7]]
ar_mean_2[8,:,:] = ar_mean[7,:,[0,1,6,7]].T
ar_mean_2[9,:,:] = ar_mean[7,:,[2,3,8,9]].T
ar_mean_2[10:12,:,:] = ar_mean[8:10,:,[0,1,6,7]]
ar_mean_2[12,:,:] = ar_mean[10,:,[0,1,6,7]].T
ar_mean_2[13,:,:] = ar_mean[10,:,[2,3,8,9]].T
ar_mean_2[14:16,:,:] = ar_mean[11:13,:,[0,1,6,7]]
ar_mean_2[16,:,:] = ar_mean[13,:,[0,1,6,7]].T
ar_mean_2[17,:,:] = ar_mean[13,:,[2,3,8,9]].T
ar_mean_2[18:20,:,:] = ar_mean[14:16,:,[0,1,6,7]]
ar_mean_2[20,:,:] = ar_mean[16,:,[0,1,6,7]].T
ar_mean_2[21,:,:] = ar_mean[16,:,[2,3,8,9]].T
ar_mean_2[22:24,:,:] = ar_mean[17:19,:,[0,1,6,7]]
ar_mean_2[24,:,:] = ar_mean[19,:,[0,1,6,7]].T
ar_mean_2[25,:,:] = ar_mean[19,:,[2,3,8,9]].T
ar_mean_2[26:28,:,:] = ar_mean[20:22,:,[0,1,6,7]]
ar_mean_2[28,:,:] = ar_mean[22,:,[0,1,6,7]].T
ar_mean_2[29,:,:] = ar_mean[22,:,[2,3,8,9]].T
ar_mean_2[30:32,:,:] = ar_mean[23:25,:,[0,1,6,7]]
ar_mean_2[32,:,:] = ar_mean[25,:,[0,1,6,7]].T
ar_mean_2[33,:,:] = ar_mean[25,:,[2,3,8,9]].T
ar_mean_2[34:36,:,:] = ar_mean[26:28,:,[0,1,6,7]]
ar_mean_2[36,:,:] = ar_mean[28,:,[0,1,6,7]].T
ar_mean_2[37,:,:] = ar_mean[28,:,[2,3,8,9]].T
ar_mean_2[38:40,:,:] = ar_mean[29:31,:,[0,1,6,7]]
ar_mean_2[40,:,:] = ar_mean[31,:,[0,1,6,7]].T
ar_mean_2[41,:,:] = ar_mean[31,:,[2,3,8,9]].T
ar_mean_2[42:,:,:] = ar_mean[32:,:,[0,1,6,7]]

ar_std_2 = np.zeros((44,num_measurements,4))
ar_std_2[0,:,:] = ar_std[1,:,[0,1,6,7]].T
ar_std_2[1,:,:] = ar_std[1,:,[2,3,8,9]].T
ar_std_2[2:4,:,:] = ar_std[2:4,:,[0,1,6,7]]
ar_std_2[4,:,:] = ar_std[4,:,[0,1,6,7]].T
ar_std_2[5,:,:] = ar_std[4,:,[2,3,8,9]].T
ar_std_2[6:8,:,:] = ar_std[5:7,:,[0,1,6,7]]
ar_std_2[8,:,:] = ar_std[7,:,[0,1,6,7]].T
ar_std_2[9,:,:] = ar_std[7,:,[2,3,8,9]].T
ar_std_2[10:12,:,:] = ar_std[8:10,:,[0,1,6,7]]
ar_std_2[12,:,:] = ar_std[10,:,[0,1,6,7]].T
ar_std_2[13,:,:] = ar_std[10,:,[2,3,8,9]].T
ar_std_2[14:16,:,:] = ar_std[11:13,:,[0,1,6,7]]
ar_std_2[16,:,:] = ar_std[13,:,[0,1,6,7]].T
ar_std_2[17,:,:] = ar_std[13,:,[2,3,8,9]].T
ar_std_2[18:20,:,:] = ar_std[14:16,:,[0,1,6,7]]
ar_std_2[20,:,:] = ar_std[16,:,[0,1,6,7]].T
ar_std_2[21,:,:] = ar_std[16,:,[2,3,8,9]].T
ar_std_2[22:24,:,:] = ar_std[17:19,:,[0,1,6,7]]
ar_std_2[24,:,:] = ar_std[19,:,[0,1,6,7]].T
ar_std_2[25,:,:] = ar_std[19,:,[2,3,8,9]].T
ar_std_2[26:28,:,:] = ar_std[20:22,:,[0,1,6,7]]
ar_std_2[28,:,:] = ar_std[22,:,[0,1,6,7]].T
ar_std_2[29,:,:] = ar_std[22,:,[2,3,8,9]].T
ar_std_2[30:32,:,:] = ar_std[23:25,:,[0,1,6,7]]
ar_std_2[32,:,:] = ar_std[25,:,[0,1,6,7]].T
ar_std_2[33,:,:] = ar_std[25,:,[2,3,8,9]].T
ar_std_2[34:36,:,:] = ar_std[26:28,:,[0,1,6,7]]
ar_std_2[36,:,:] = ar_std[28,:,[0,1,6,7]].T
ar_std_2[37,:,:] = ar_std[28,:,[2,3,8,9]].T
ar_std_2[38:40,:,:] = ar_std[29:31,:,[0,1,6,7]]
ar_std_2[40,:,:] = ar_std[31,:,[0,1,6,7]].T
ar_std_2[41,:,:] = ar_std[31,:,[2,3,8,9]].T
ar_std_2[42:,:,:] = ar_std[32:,:,[0,1,6,7]]

color = iter(cm.rainbow(np.linspace(0, 1, 44)))

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(16, 4))
ax1.set_ylabel('RMSE', fontsize=12)
# ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_title('(a)', fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
# ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_title('(b)', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
# ax3.set_xlabel('Epoch', fontsize=12)
# ax3.set_ylabel('RMSE', fontsize=12)
ax3.set_title('(c)', fontsize=12)
ax3.tick_params(axis='both', which='major', labelsize=12)
color = iter(cm.rainbow(np.linspace(0, 1, 20)))
for i in range(0,20):
    c = next(color)
    ax1.plot(np.arange(0,num_epochs,10), ar_mean_2[i, :, 0], '-', c=c, label='Exp.'+str(i+1).zfill(2))
    ax1.fill_between(np.arange(0,num_epochs,10), ar_mean_2[i, :, 0]-ar_std_2[i, :, 0],
                     ar_mean_2[i, :, 0]+ar_std_2[i, :, 0], color=c, alpha=.15)
    ax1.plot(np.arange(0,num_epochs,10), ar_mean_2[i, :, 1], ':', c=c)
    ax1.fill_between(np.arange(0,num_epochs,10), ar_mean_2[i, :, 1]-ar_std_2[i, :, 1],
                     ar_mean_2[i, :, 1]+ar_std_2[i, :, 1], color=c, alpha=.15)
ax1.legend(fontsize=10, ncols=2)

color = iter(cm.rainbow(np.linspace(0, 1, 16)))
for i in range(20,36):
    c = next(color)
    ax2.plot(np.arange(0,num_epochs,10), ar_mean_2[i, :, 0], '-', c=c, label='Exp.'+str(i+1))
    ax2.fill_between(np.arange(0,num_epochs,10), ar_mean_2[i, :, 0]-ar_std_2[i, :, 0],
                     ar_mean_2[i, :, 0]+ar_std_2[i, :, 0], color=c, alpha=.15)
    ax2.plot(np.arange(0,num_epochs,10), ar_mean_2[i, :, 1], ':', c=c)
    ax2.fill_between(np.arange(0,num_epochs,10), ar_mean_2[i, :, 1]-ar_std_2[i, :, 1],
                     ar_mean_2[i, :, 1]+ar_std_2[i, :, 1], color=c, alpha=.15)
ax2.legend(fontsize=10, ncols=2)

color = iter(cm.rainbow(np.linspace(0, 1, 8)))
for i in range(36,44):
    c = next(color)
    ax3.plot(np.arange(0,num_epochs,10), ar_mean_2[i, :, 0], '-', c=c, label='Exp.'+str(i+1))
    ax3.fill_between(np.arange(0,num_epochs,10), ar_mean_2[i, :, 0]-ar_std_2[i, :, 0],
                     ar_mean_2[i, :, 0]+ar_std_2[i, :, 0], color=c, alpha=.15)
    ax3.plot(np.arange(0,num_epochs,10), ar_mean_2[i, :, 1], ':', c=c)
    ax3.fill_between(np.arange(0,num_epochs,10), ar_mean_2[i, :, 1]-ar_std_2[i, :, 1],
                     ar_mean_2[i, :, 1]+ar_std_2[i, :, 1], color=c, alpha=.15)
ax3.legend(fontsize=10, ncols=2)

# %%
ar = np.load('/home/finperprint-analysis/foe/results/part-1/best_10_401/ar.npy')
num_epochs = 401
eval_step = 10
ar_mean = np.mean(ar, axis=0)
ar_mean[22, :, :] = np.mean(ar[[0, 1, 2, 4], 22, :, :], axis=0)
ar_mean_min = np.min(ar_mean[:, :, 7], axis=1)
ar_mean_min_el = ar_mean_min[1:16]
ar_mean_min_ep = ar_mean_min[16:28]
ar_mean_min_km = ar_mean_min[28:]
ar_mean_min_oh = ar_mean_min[np.arange(1, 34, 3)]
ar_mean_min_oh_exp = np.min(ar_mean[[1, 4, 7, 10, 13, 16, 19, 25, 28, 31], :, 9], axis=1)
ar_mean_min_or = ar_mean_min[np.arange(2, 34, 3)]
ar_mean_min_cr = ar_mean_min[np.arange(3, 34, 3)]
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.set_ylabel('Mean RMSE')
ax.set_xlabel('Discretization and Encoding Methods')
ax.yaxis.grid(True)
bplot = ax.boxplot([ar_mean_min_el, ar_mean_min_ep, ar_mean_min_km, ar_mean_min_oh, ar_mean_min_oh_exp, ar_mean_min_or, ar_mean_min_cr],labels=['equal\nlength', 'equal\nprobability', 'random', 'one-hot\nmax', 'one-hot\nexp', 'ordinal', 'circular'], showmeans=True, widths=0.4, positions=[1,2,3,5,6,7,8], patch_artist=True)
for patch, color in zip(bplot['boxes'], ['cornsilk', 'cornsilk', 'cornsilk', 'azure', 'azure', 'azure', 'azure', 'azure']):
    patch.set_facecolor(color)
# %%
ar = np.load('/home/finperprint-analysis/foe/results/part-1/best_10_401/ar.npy')
num_epochs = 401
eval_step = 10
ar_mean = np.mean(ar, axis=0)
ar_mean[22, :, :] = np.mean(ar[[0, 1, 2, 4], 22, :, :], axis=0)
ar_mean_min = np.min(ar_mean[:, :, 7], axis=1)
ar_mean_min_el = ar_mean_min[1:16]
ar_mean_min_ep = ar_mean_min[16:28]
ar_mean_min_km = ar_mean_min[28:]
ar_mean_min_oh = ar_mean_min[np.arange(1, 34, 3)]
ar_mean_min_oh_exp = np.min(ar_mean[[1, 4, 7, 10, 13, 16, 19, 25, 28, 31], :, 9], axis=1)
ar_mean_min_or = ar_mean_min[np.arange(2, 34, 3)]
ar_mean_min_cr = ar_mean_min[np.arange(3, 34, 3)]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xlabel('Discretization and Encoding Methods', fontsize=16)
ax.yaxis.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
bplot = ax.boxplot([ar_mean_min_el, ar_mean_min_ep, ar_mean_min_km, ar_mean_min_oh, ar_mean_min_oh_exp, ar_mean_min_or, ar_mean_min_cr],labels=['equal\nlength', 'equal\nprobability', 'random', 'one-hot\nmax', 'one-hot\nexp', 'ordinal', 'cyclic'], showmeans=True, widths=0.4, positions=[1,2,3,5,6,7,8], patch_artist=True)
for patch, color in zip(bplot['boxes'], ['cornsilk', 'cornsilk', 'cornsilk', 'azure', 'azure', 'azure', 'azure', 'azure']):
    patch.set_facecolor(color)