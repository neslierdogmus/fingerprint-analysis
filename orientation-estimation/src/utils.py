import numpy as np
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt

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
                    for (ori,m) in zip(oris, mask) if m>0]
            oris = np.array(oris)
            if method == "eq_prob":
                assert num_class <= 143, "The number of classes should be\
                                          smaller than the ratio of number of\
                                          all orientations to the size of the\
                                          largest orientation set."
                oris.sort()
                shift = sum(oris>=np.unique(oris)[i*int(360/num_class)])
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
                        oris_shifted[oris_shifted>np.pi] -= np.pi
                        _, clusters = kmeans2(oris_shifted, num_class,
                                                      minit='random',
                                                      missing='raise')                        
                        complete = True
                    except:
                        complete = False
                        
                disc = [oris[clusters==i].min() for i in range(num_class)]
                disc = np.array(disc) - shift
                disc[disc < 0] += np.pi
                disc.sort()
                disc = np.append(disc, [disc[0]+np.pi])
        discs.append(disc)
    return discs

def view_discretization(discs, sample=None):
    _, ax = plt.subplots(1,1)
    width = 0.06
    M = len(discs)
    for i in range(M):
        disc = discs[i]
        sizes = [(disc[i+1]-disc[i])/np.pi*100 for i in range(len(disc)-1)]
        ax.pie(sizes, radius=1-i*width, startangle=disc[0]/np.pi*180,
                wedgeprops=dict(width=width, edgecolor='w'))
    ax.plot([-1.05,1.05], [0, 0], linewidth=1, color='k', clip_on=False)
    ax.plot([0, 0], [-1.05,1.05], linewidth=1, color='k', clip_on=False)
    ax.text(1.1,-0.025,'0$^\circ$/ $\pi$')
    ax.text(-0.2, 1.1,'45$^\circ$/ $\pi/4$')
    ax.text(-1.5,-0.025,'90$^\circ$/ $\pi/2$')
    ax.text(-0.25, -1.15,'135$^\circ$/ $3\pi/4$')
    
    if sample is not None:
        fig, axes = plt.subplots(M//2, 2, subplot_kw={'projection': 'polar'})
        fig.set_figheight(20)
        fig.set_figwidth(10)
        oris = [ori for fp in sample.fps
                    for (oris, mask) in zip(fp.gt.orientations, fp.gt.mask)
                    for (ori,m) in zip(oris, mask) if m>0]
        oris = np.array(oris)
        for i in range(M):
            disc = discs[i]
            N = len(disc) - 1
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
            
            ax2 = axes[i//2, i%2]
            bars = ax2.bar(2*np.array(xticks), count, width=2*np.array(w),
                           bottom=1000)
            ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax2.set_xticklabels(['0 / $\pi$', '$\pi/4$',
                                 '$\pi/2$', '$3\pi/4$'])
            ax2.set_yticks([2000,3000,4000])
            ax2.set_yticklabels([1000,2000,3000])
            
def encode_angle(oris, method, disc):
    valid_methods = ["one_hot", "ordinal", "cyclic"]
    assert method in valid_methods, "not a valid encoding method"

    K = len(disc) - 1
    if method == 'cyclic':
        codes = []
        for ori in oris:
            ori_coded = []
            for i in range(0, K//2):
                ori_coded.append(int(disc[i] <= ori < disc[i+K//2]))
            codes.append(ori_coded)
        codes = np.array(codes)
    else:
        bin_num = np.digitize(oris, np.append([0], disc)) - 1
        bin_num[bin_num == bin_num.max()] = 0            
        if method =='one_hot':
            codes = np.zeros((len(oris), K))
            codes[np.arange(len(bin_num)), bin_num] = 1
        elif method == 'ordinal':
            codes = np.array([np.pad(np.ones(b),(0, K-1-b)) for b in bin_num])
    
    return codes

def view_codes(discs):
    methods = ["one_hot", "ordinal", "cyclic"]
    M = len(discs)
    N = len(methods)
    c = len(discs[0]) - 1
    oris = np.arange(256) * np.pi / 256
    dpi = 100
    pixel_per_bar = 10
    pixel_per_ori = 2
    fig_size = (c * pixel_per_bar * N / dpi, 256 * pixel_per_ori * M / dpi)
    fig, axes = plt.subplots(M, N, figsize=fig_size, dpi=dpi)
    for i in range(M):
        for j in range(N):
            codes = encode_angle(oris, methods[j], discs[i])
            ax = axes[i][j]
            ax.imshow(1-codes, cmap='binary', aspect='auto', 
                        interpolation='nearest')