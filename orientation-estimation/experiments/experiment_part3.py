# %%
import numpy as np
import torch
import torch.nn.functional as F

from foe_fp_image_dataset import FOEFPImageDataset
from foe_model_convnet import FOEConvNet
import utils

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

num_folds = 10
use_cpu = False

num_epochs = 401
eval_step = 5
batch_size = 1
num_workers = 4

lr = 10**-3
gamma = 10**-4
power = 0.75
weight_decay = 5*10**-5
momentum = 0.7

lr_coeff = [30, 30, 10]
encod_met = 'circular'
hflip = True
rot_lim = 5
num_synth = 50
gamma_correct = True
add_noise = True

use_gpu = torch.cuda.is_available() and not use_cpu
device = 'cuda' if use_gpu else 'cpu'
print('Device:', device)

base_path_bad = '../../datasets/foe/Bad'
base_path_good = '../../datasets/foe/Good'
base_path_synth = '../../datasets/foe/Synth'

try:
    parts_bad = np.load('parts_bad.npy')
    parts_good = np.load('parts_good.npy')
    print('Fold splits loaded.')
except FileNotFoundError:
    parts_bad = utils.split_database(base_path_bad, num_folds)
    parts_good = utils.split_database(base_path_good, num_folds)

    np.save('parts_bad.npy', parts_bad)
    np.save('parts_good.npy', parts_good)
# %%
all_loss = []
all_results = []
for fold in range(num_folds):
    fp_ids_bad_val = parts_bad[fold]
    fp_ids_bad_tra = np.append(parts_bad[:fold], parts_bad[fold+1:])
    fp_ids_good_val = parts_good[fold]
    fp_ids_good_tra = np.append(parts_good[:fold], parts_good[fold+1:])
    fp_ids_synth = ['10649', '9327', '9628', '7424', '3376', '10117', '12249',
                    '7444', '6332', '7421', '9477', '5247', '7134', '10481',
                    '5824', '10747', '10091', '7235', '11594', '5023', '10082',
                    '6332', '12247', '10927', '11534', '6931', '12469', '6591',
                    '11733', '8650', '12105', '11057', '11859', '3480', '4498',
                    '10092', '10408', '6578', '7422', '2325', '12238', '4947',
                    '10666', '10516', '7051', '2771', '12040', '11052', '7884',
                    '11227', '5634', '4008', '10023', '10024', '11626', '2782',
                    '10254', '8155', '6783', '10569', '10906', '3886', '5138',
                    '12058', '11896', '4261', '3340', '4538', '11094', '12130']
    import random
    random.shuffle(fp_ids_synth)
    fp_ids_synth_tra = fp_ids_synth[:num_synth]
    fp_ids_synth_val = fp_ids_synth[num_synth:]
    foe_img_ds_val = FOEFPImageDataset([base_path_bad, base_path_good,
                                        base_path_synth],
                                       [fp_ids_bad_val, fp_ids_good_val,
                                        fp_ids_synth_val])
    foe_img_ds_tra = FOEFPImageDataset([base_path_bad, base_path_good,
                                        base_path_synth],
                                       [fp_ids_bad_tra, fp_ids_good_tra,
                                        fp_ids_synth_tra])
    foe_img_ds_tra.set_hflip(hflip)
    foe_img_ds_tra.set_rotlim(rot_lim)
    foe_img_ds_tra.set_gamma_correct(gamma_correct)
    foe_img_ds_tra.set_add_noise(add_noise)
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
    disc_names = []
    lrc = iter(lr_coeff)
    for params in [['eq_len', 256, 1], ['eq_len', 128, 2], ['eq_len', 64, 4]]:
        disc_name = 'f'+str(fold)+'_'+'_'.join([str(p) for p in params])
        disc_names.append(disc_name)
        discs = utils.discretize_orientation(params[0], params[1], params[2],
                                             foe_img_ds_tra)
        # utils.view_discs(discs, disc_name, foe_img_ds_tra)
        # utils.view_codes(discs, disc_name)
        # np.save(disc_name+'.npy', discs)

        n_disc = len(discs)
        n_class = len(discs[0]) - 1
        loss_fnc = F.binary_cross_entropy_with_logits
        prob_fnc = F.sigmoid
        code_len = n_class // 2
        K = n_disc * code_len
        exp_name = disc_name + '_' + encod_met

        print('>>>>> Experiment:', exp_name)

        model = FOEConvNet(out_len=K)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr*next(lrc),
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        # lr_lambda = utils.construct_lr_lambda(gamma, power)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.003, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=120, power=1.0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.0003)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
        
        model = model.to(device)
        print(sum(p.numel() for p in model.parameters()))
        loss_list = []
        result_list = []
        for e in range(num_epochs):
            model.train()
            e_loss = []
            print("%4d %7.5f" % (e, scheduler.get_last_lr()[0]), end=' ')
            for x, oris, mask, fp_type, ind in foe_img_dl_tra:
                x = x.to(device)
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
                e_loss.append(loss.item())
                optimizer.step()
            if e < 100:
                scheduler.step()

            loss_list.append(np.mean(e_loss))
            end = ' ' if e % eval_step == 0 else '\n'
            print("%7.5f" % (loss_list[-1]), end=end)
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
                    rmse_s = [[], []]
                    for d in range(len(dls)):
                        for x, oris, mask, fp_type, ind in dls[d]:
                            x = x.to(device)
                            yo = model(x)
                            mask_np = mask.cpu().detach().numpy()
                            yt = utils.encode_angle(oris, encod_met, discs)
                            yt = yt.to(device)
                            ests = utils.decode_angle(yo, encod_met, discs,
                                                      prob_fnc)
                            new_rmse = utils.calc_rmse(oris, ests, mask_np)
                            for er, t in zip(new_rmse, fp_type):
                                if t == 'Good':
                                    rmse_g[d] = np.append(rmse_g[d], [er])
                                elif t == 'Bad':
                                    rmse_b[d] = np.append(rmse_b[d], [er])
                                elif t == 'Synth':
                                    rmse_s[d] = np.append(rmse_s[d], [er])
                res_ls = [rmse_g, rmse_b, rmse_s]
                result_list.append([np.mean(res[d]) for d in range(2)
                                    for res in res_ls])
                print("%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f" % tuple(result_list[-1]))
        fold_loss.append(loss_list)
        fold_results.append(result_list)
        # ind_str = str(ind[-1].item())
        # rmse_str = str(new_rmse[-1])
        # print('Fingerprint {} with  RMSE {}'.format(ind_str, rmse_str))
        # img_name = exp_name + '_' + ind_str + '_' + rmse_str + '.png'
        # utils.view_ests(x[-1], oris[-1], ests[-1], mask[-1], img_name)
    all_loss.append(fold_loss)
    all_results.append(fold_results)
# %%
# plot all results
al = np.array(all_loss)
ar = np.array(all_results)

np.save('al.npy', al)
np.save('ar.npy', ar)

al_mean = np.mean(al, axis=0)
ar_mean = np.mean(ar, axis=0)
al_std = np.std(al, axis=0)
ar_std = np.std(ar, axis=0)

x_data = np.arange(0, num_epochs, eval_step)+1


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
ar_mean_min = np.min(ar_mean[:, :, 3], axis=1)
ar_mean_100 = ar_mean_min

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.ylabel('RMSE')
plt.xlabel('Experiments')
_ = ax.bar(np.arange(1, 6), np.sort(ar_mean_100))
ax.set_xticks(np.arange(1, 6))
ax.set_xticklabels(np.argsort(ar_mean_100)+1)
plt.ylim(8, 11)
plt.xlim(0, 6)

color = iter(cm.rainbow(np.linspace(0, 1, 5)))
ind = -1
plt.figure()
for disc_name in disc_names:
    ind += 1
    c = next(color)
    par = ' '.join(disc_name.split('_')[-2:])
    plot_rmse(0, x_data, ar_mean, ar_std, ind, c, par)
plt.show()

# %%
# folders = ['ConvNet', 'ConvNet_20', 'ConvNet_70', 'ConvNet_Synth_48',
#            'ConvNet_Synth_144', 'ConvNet_Synth_144_20']
# labels = ['Tam evrişimli', 'Tam evrişimli_20', 'Tam evrişimli_70',
#           'Tam evrişimli_Sentetik_48', 'Tam evrişimli_Sentetik_144',
#           'Tam evrişimli_Sentetik_144_20', ]
# num = len(folders)
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# shift = np.arange(-0.4, 0.4, 0.8/num)
# xticks = np.array([])
# xtick_labels = np.array([])
# for i in range(num):
#     folder = folders[i]
#     sh = shift[i]
#     ar = np.load('../results/part-2/'+folder+'/ar.npy')
#     if ar.shape[-1] > 4:
#         ar = ar[:, :, :, [0, 1, 6, 7]]
#     ar_mean = np.mean(ar, axis=0)
#     ar_std = np.std(ar, axis=0)
#     ar_mean_min = np.min(ar_mean[:, :, 2], axis=1)
#     ar_std_min = ar_std[[0, 1, 2, 3, 4], np.argmin(ar_mean[:, :, 2], axis=1),
#                         2]

#     plt.ylabel('Ortalama Karesel Hata')
#     plt.xlabel('Deneyler')
#     _ = ax.bar(np.arange(1, 6)+sh, ar_mean_min, width=0.1, label=labels[i])
#     ax.errorbar(np.arange(1, 6)+sh, ar_mean_min, ar_std_min, fmt='.',
#                 color='Black', elinewidth=2, capthick=10, errorevery=1,
#                 alpha=0.5, ms=2, capsize=0)
# ax.set_xticks(np.arange(1, 6))
# ax.set_xticklabels(disc_names)
# plt.ylim(6, 13)
# plt.xlim(0, 6)
# plt.legend()
