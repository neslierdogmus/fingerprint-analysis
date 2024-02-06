import os
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import ClusterError
import torch


def split_database(base_path, num_folds):
    index_path = os.path.join(base_path, 'index.txt')
    with open(index_path, 'r') as fin:
        fp_ids = [line.split('.')[0] for line in fin.readlines()[1:]]
    random.shuffle(fp_ids)
    parts = np.array_split(fp_ids, num_folds)
    return parts


def angle_to_sincos(oris):
    sincos = np.stack([np.sin(2*oris), np.cos(2*oris)], axis=1)

    return torch.from_numpy(sincos.astype(np.single))


def sincos_to_angle(outputs):
    out_np = outputs.cpu().detach().numpy()
    oris = np.arctan2(out_np[:, 0], out_np[:, 1]) / 2
    oris = np.where(oris < 0, oris+np.pi, oris)

    return oris


def construct_lr_lambda(gamma, power):
    def lr_lambda(epoch):
        return (1 + gamma * epoch) ** (-power)
    return lr_lambda


def discretize_orientation(method, num_class=32, num_disc=8, sample=None):
    valid_methods = ["eq_len", "eq_prob", "k_means"]
    assert method in valid_methods, "not a valid discretization method"

    if method != "eq_len":
        assert sample is not None, "For the given method, a sample\
                                    should be provided."

    discs = []
    for i in range(num_disc):
        if method == "eq_len":
            start = i * np.pi / (num_class*num_disc)
            disc = np.linspace(start, start+np.pi, num_class+1)
        else:
            oris = [ori for fp in sample.fps
                    for (oris, mask) in zip(fp.gt.orientations, fp.gt.mask)
                    for (ori, m) in zip(oris, mask) if m > 0]
            oris = np.array(oris)
            if method == "eq_prob":
                assert num_class <= 143, "The number of classes should be\
                                          smaller than the ratio of number of\
                                          all orientations to the size of the\
                                          largest orientation set."
                oris.sort()
                shift = sum(oris >= np.unique(oris)[i*int(180/num_class)])
                oris = np.roll(oris, shift)
                samples_per_bin = int(len(oris)/num_class)
                edge_samples = np.arange(num_class) * samples_per_bin
                disc = np.sort(oris[edge_samples])
                disc = np.append(disc, [disc[0]+np.pi])
            else:
                complete = False
                while not complete:
                    try:
                        shift = np.random.random() * np.pi / 2
                        oris_shifted = oris + shift
                        oris_shifted[oris_shifted > np.pi] -= np.pi
                        _, clusters = kmeans2(oris_shifted, num_class,
                                              minit='random', missing='raise')
                        complete = True
                    except ClusterError:
                        complete = False

                disc = [oris[clusters == i].min() for i in range(num_class)]
                disc = np.array(disc) - shift
                disc[disc < 0] += np.pi
                disc.sort()
                disc = np.append(disc, [disc[0]+np.pi])
        discs.append(disc)
    return discs


def view_discs(discs, img_name, sample=None):
    fig, ax = plt.subplots(1, 1)
    M = len(discs)
    width = 0.5 / M
    for i in range(M):
        disc = discs[i]
        sizes = [(disc[i+1]-disc[i])/np.pi*100 for i in range(len(disc)-1)]
        ax.pie(sizes, radius=1-i*width, startangle=disc[0]/np.pi*180,
               wedgeprops=dict(width=width, edgecolor='w'))
    ax.plot([-1.05, 1.05], [0, 0], linewidth=1, color='k', clip_on=False)
    ax.plot([0, 0], [-1.05, 1.05], linewidth=1, color='k', clip_on=False)
    ax.text(1.1, -0.025, r'0$^\circ$/ $\pi$')
    ax.text(-0.2, 1.1, r'45$^\circ$/ $\pi/4$')
    ax.text(-1.5, -0.025, r'90$^\circ$/ $\pi/2$')
    ax.text(-0.25, -1.15, r'135$^\circ$/ $3\pi/4$')

    fig.savefig(img_name+'_dist.png')

    if sample is not None:
        if M > 1:
            fig, axes = plt.subplots(M//2, 2,
                                     subplot_kw={'projection': 'polar'})
        else:
            fig, axes = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
        fig.set_figheight(30)
        fig.set_figwidth(10)
        oris = [ori for fp in sample.fps
                for (oris, mask) in zip(fp.gt.orientations, fp.gt.mask)
                for (ori, m) in zip(oris, mask) if m > 0]
        oris = np.array(oris)
        for i in range(M):
            disc = discs[i]
            count, _ = np.histogram(oris, np.append([0], disc))
            count[-1] += count[0]
            count = count[1:]
            xticks = []

            w = []
            for j in range(len(disc)-1):
                xticks.append((disc[j]+disc[j+1])/2)
                w.append(disc[j+1]-disc[j])
            if xticks[-1] > np.pi:
                xticks[-1] -= np.pi
            w[-1] += disc[0]

            if M > 2:
                ax2 = axes[i//2, i % 2]
            elif M > 1:
                ax2 = axes[i % 2]
            else:
                ax2 = axes
            mc = max(count)
            step = int(np.ceil(mc / 300) * 100)
            _ = ax2.bar(2*np.array(xticks), count, width=2*np.array(w),
                        bottom=step)
            ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax2.set_xticklabels([r'0 / $\pi$', r'$\pi/4$',
                                 r'$\pi/2$', r'$3\pi/4$'])
            ax2.set_yticks([step*2, step*3, step*4])
            ax2.set_yticklabels([step, step*2, step*3])
        fig.savefig(img_name+'_hist.png')
    plt.close('all')


def encode_angle(oris, method, discs):
    valid_methods = ["one_hot", "ordinal", "circular"]
    assert method in valid_methods, "not a valid encoding method"

    codes_dim = list(oris.size())
    codes_dim.insert(1, 0)
    all_codes = torch.empty(codes_dim)
    for disc in discs:
        L = len(disc) - 1
        if method == 'circular':
            code_len = L//2
            codes_dim[1] = code_len
            codes = np.zeros(codes_dim)
            for j in range(code_len):
                low, up = disc[j], disc[j+code_len]
                codes[:, j] = torch.logical_and(torch.gt(oris, low),
                                                torch.le(oris, up)).long()
        else:
            bin_num = np.digitize(oris, disc) - int(disc[0] == 0)
            bin_num[bin_num == L] = 0
            if method == 'one_hot':
                codes = np.identity(L)[bin_num]
            elif method == 'ordinal':
                code_array = np.array([np.pad(np.ones(i), (0, L-1-i))
                                       for i in range(L)])
                codes = code_array[bin_num]
            codes = np.moveaxis(codes, 3, 1)
        all_codes = torch.cat((all_codes, torch.from_numpy(codes)), dim=1)

    return all_codes


def view_codes(discs, img_name):
    methods = ["one_hot", "ordinal", "circular"]
    M = len(discs)
    N = len(methods)
    L = len(discs[0]) - 1
    code_len = [L, L-1, L//2]

    oris = np.expand_dims(np.arange(256) * np.pi / 256, (0, 1))
    oris = torch.from_numpy(oris)
    dpi = 100
    pixel_per_bar = 10
    pixel_per_ori = 2
    fig_size = (L * pixel_per_bar * N / dpi, 256 * pixel_per_ori * M / dpi)
    fig, axes = plt.subplots(M, N, figsize=fig_size, dpi=dpi)
    for j in range(N):
        codes = encode_angle(oris, methods[j], discs)
        c = code_len[j]
        for i in range(M):
            if M > 1:
                ax = axes[i][j]
            else:
                ax = axes[j]
            ax.imshow((1-codes[0, i*c:(i+1)*c, 0]).T, cmap='binary',
                      aspect='auto', interpolation='nearest')
    fig.savefig(img_name+'_codes.png')
    plt.close('all')


def decode_angle(outputs, method, discs, prob_fnc, regr='max'):
    valid_methods = ["one_hot", "ordinal", "circular"]
    assert method in valid_methods, "not a valid encoding method"

    dim = list(outputs.size())
    n_disc = len(discs)
    code_len = dim[1] // n_disc
    dim[1] = n_disc
    ests = np.zeros(dim)

    for d in range(n_disc):
        disc = discs[d]
        probs = prob_fnc(outputs[:, d*code_len:(d+1)*code_len])
        out_np = probs.cpu().detach().numpy()

        bin_cen = np.array([(disc[j+1]+disc[j])/2 for j in range(len(disc)-1)])
        bin_cen[bin_cen > np.pi] -= np.pi
        bin_cen.sort()
        if method == 'one_hot':
            val_regress = ['max', 'exp']
            assert regr in val_regress, "not a valid one-hot regression method"
            if regr == 'max':
                labels = np.argmax(out_np, axis=1)
                oris = bin_cen[labels]
            else:
                # circular expected value
                bin_cen_exp = bin_cen[None, :, None, None]
                exp_sin = np.sum(out_np * np.sin(bin_cen_exp * 2), axis=1)
                exp_cos = np.sum(out_np * np.cos(bin_cen_exp * 2), axis=1)
                oris = np.arctan2(exp_sin, exp_cos)
                oris = np.where(oris < 0, oris+2*np.pi, oris) / 2
        elif method == 'ordinal':
            labels = np.sum(out_np > 0.5, axis=1)
            oris = bin_cen[labels]
        else:
            L = len(disc)-1
            weights = np.ones((L//2, L)) * -1
            for i in range(L):
                if i <= L//2:
                    weights[0:i, i] = 1
                else:
                    weights[i-L//2:, i] = 1
            weighted_sums = np.dot(np.moveaxis(out_np, 1, 3), weights)
            plus_ones = np.append(np.arange(L//2, 0, -1), np.arange(L//2))
            weighted_sums += plus_ones
            labels = weighted_sums.argmax(axis=-1)
            oris = bin_cen[labels]
        ests[:, d] = oris

    # circular mean along discs
    ests = np.arctan2(np.mean(np.sin(ests*2), axis=1),
                      np.mean(np.cos(ests*2), axis=1))
    ests = np.where(ests < 0, ests+2*np.pi, ests) / 2

    return ests


def calc_rmse(oris, ests, mask_np):
    oris_np = oris.numpy()
    oris_np_degrees = oris_np / np.pi * 180
    ests = ests * mask_np
    ests_degrees = ests / np.pi * 180
    diff_degrees = np.abs(oris_np_degrees - ests_degrees)
    diff_degrees = np.where(diff_degrees > 90, 180-diff_degrees, diff_degrees)
    se_degrees = np.sum(np.power(diff_degrees, 2), axis=(1, 2))
    rmse_degrees = np.sqrt(se_degrees / np.sum(mask_np, axis=(1, 2)))

    return rmse_degrees


def calc_class_acc(yo, yt, mask_np, code_len, n_disc, method):
    valid_methods = ["one_hot", "ordinal", "circular"]
    assert method in valid_methods, "not a valid encoding method"

    yo_np = yo.cpu().detach().numpy()
    yt_np = yt.cpu().detach().numpy()
    acc = []
    for i in range(n_disc):
        yoi = yo_np[:, i*code_len:(i+1)*code_len]
        yti = yt_np[:, i*code_len:(i+1)*code_len]
        if method == "one_hot":
            check = yoi.argmax(axis=1) == yti.argmax(axis=1)
            num_total = np.sum(mask_np, axis=(1, 2))
        else:
            check = np.sum(np.equal(yoi > 0.5, yti > 0.5), axis=1)
            num_total = np.sum(mask_np, axis=(1, 2)) * code_len
        num_corr = np.sum(check * mask_np, axis=(1, 2))
        acc.append(num_corr / num_total)

    return np.mean(acc, axis=0)
