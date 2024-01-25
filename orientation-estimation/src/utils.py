import numpy as np
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt

def discretize_orientation(method, num_class=36, num_disc=5, sample=None):
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

def view_discretization(discs):
    fig, ax = plt.subplots()
    width = 0.1
    for i in range(len(discs)):
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