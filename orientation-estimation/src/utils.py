import torch
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import ClusterError
from matplotlib import pyplot as plt


def angle_to_sincos(oris):
    sincos = np.append(np.sin(2*oris), np.cos(2*oris), axis=1)

    return torch.from_numpy(sincos.astype(np.single))


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
                shift = sum(oris >= np.unique(oris)[i*int(360/num_class)])
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


def view_discretization(discs, sample=None):
    _, ax = plt.subplots(1, 1)
    width = 0.06
    M = len(discs)
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

    if sample is not None:
        fig, axes = plt.subplots(M//2, 2, subplot_kw={'projection': 'polar'})
        fig.set_figheight(20)
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

            ax2 = axes[i//2, i % 2]
            _ = ax2.bar(2*np.array(xticks), count, width=2*np.array(w),
                        bottom=1000)
            ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax2.set_xticklabels([r'0 / $\pi$', r'$\pi/4$',
                                 r'$\pi/2$', r'$3\pi/4$'])
            ax2.set_yticks([2000, 3000, 4000])
            ax2.set_yticklabels([1000, 2000, 3000])


def encode_angle(oris, method, disc):
    valid_methods = ["one_hot", "ordinal", "cyclic"]
    assert method in valid_methods, "not a valid encoding method"

    L = len(disc) - 1
    if method == 'cyclic':
        code_len = L//2
        codes_dim = list(oris.size())
        codes_dim.insert(1, code_len)
        codes = np.zeros(codes_dim)
        for i in range(codes_dim[0]):
            for j in range(codes_dim[1]):
                for k in range(codes_dim[2]):
                    for m in range(codes_dim[3]):
                        c = int(disc[j] < oris[i, k, m] <= disc[j+code_len])
                        codes[i, j, k, m] = c
    else:
        bin_num = np.digitize(oris, np.append([0], disc)) - 1
        bin_num[bin_num == bin_num.max()] = 0
        if method == 'one_hot':
            codes = np.identity(L)[bin_num]
        elif method == 'ordinal':
            code_array = np.array([np.pad(np.ones(i), (0, L-1-i))
                                   for i in range(L)])
            codes = code_array[bin_num]
        codes = np.moveaxis(codes, 3, 1)

    return torch.from_numpy(codes)


def view_codes(discs):
    methods = ["one_hot", "ordinal", "cyclic"]
    M = len(discs)
    N = len(methods)
    L = len(discs[0]) - 1
    oris = np.expand_dims(np.arange(256) * np.pi / 256, (0, 1))
    oris = torch.from_numpy(oris)
    dpi = 100
    pixel_per_bar = 10
    pixel_per_ori = 2
    fig_size = (L * pixel_per_bar * N / dpi, 256 * pixel_per_ori * M / dpi)
    fig, axes = plt.subplots(M, N, figsize=fig_size, dpi=dpi)
    for i in range(M):
        for j in range(N):
            codes = encode_angle(oris, methods[j], discs[i])
            ax = axes[i][j]
            ax.imshow(1-codes[0, 0], cmap='binary', aspect='auto',
                      interpolation='nearest')


def decode_angle(outputs, method, disc, regression='max'):
    valid_methods = ["one_hot", "ordinal", "cyclic"]
    assert method in valid_methods, "not a valid encoding method"

    out_np = outputs.cpu().detach().numpy()

    bin_centers = np.array([(disc[j+1]+disc[j])/2
                            for j in range(len(disc) - 1)])
    bin_centers[bin_centers > np.pi] -= np.pi
    if method == 'one_hot':
        val_regress = ['max', 'exp']
        assert regression in val_regress, "not a valid regression method"
        if regression == 'max':
            labels = np.argmax(out_np, axis=1)
            oris = bin_centers[labels]
        else:
            oris = np.sum(out_np * bin_centers[None, :, None, None], axis=1)
    elif method == 'ordinal':
        pass
    else:
        pass

    return oris


def calc_rmse(oris, ests, mask_np):

    oris_np = oris.numpy()
    oris_np_degrees = oris_np / np.pi * 180
    ests_degrees = ests / np.pi * 180
    diff_degrees = np.abs(oris_np_degrees - ests_degrees)
    diff_degrees = np.where(diff_degrees > 90, 180-diff_degrees, diff_degrees)
    se_degrees = np.sum(np.power(diff_degrees, 2), axis=(1, 2))
    rmse_degrees = np.sqrt(se_degrees / np.sum(mask_np))

    return rmse_degrees


def calc_rmse2(output_exp, output_pre, mask, n_classes):
    output_exp = output_exp.cpu().detach().numpy()
    radians_exp = np.argmax(output_exp, axis=1)/n_classes*np.pi
    degrees_exp = radians_exp / np.pi * 180

    output_pre = output_pre.cpu().detach().numpy()
    # radians_pre = np.arctan2(output_pre[:, 0], output_pre[:, 1]) / 2
    # radians_pre = np.where(radians_pre < 0, radians_pre + np.pi, radians_pre)
    radians_pre = np.argmax(output_pre, axis=1)/n_classes*np.pi
    degrees_pre = radians_pre / np.pi * 180

    degrees_diff = np.abs(degrees_exp - degrees_pre)
    degrees_diff = np.where(degrees_diff > 90, 180-degrees_diff, degrees_diff)

    mask = mask.cpu().detach().numpy()
    degrees_se = np.sum(np.power(degrees_diff, 2) * mask)
    degrees_rmse = np.sqrt(degrees_se / np.sum(mask))

    return degrees_rmse
