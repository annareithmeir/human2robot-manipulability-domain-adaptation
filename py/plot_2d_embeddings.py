import numpy as np
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rpa.diffusion_map import get_diffusionEmbedding
from pyriemann.utils.distance import distance_riemann
from rpa.helpers.transfer_learning.utils import mean_riemann

plt.rcParams['text.usetex'] = True

#
# Diffusion maps
#

def plot_diffusion_embedding(source, target, ax1, idx_s=None, idx_t=None, pairwise=False):
    covs = np.concatenate([source, target, [mean_riemann(source)], [mean_riemann(target)]])
    sess = np.array([1] * len(source) + [2] * len(target) + [3] * 1 + [4] * 1)

    uorg, l = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    #np.savetxt(path+'/diffusion_embedding.csv', uorg, delimiter=',')

    colors = {1: 'b', 2: 'r',3: 'b', 4: 'r'}
    markers = {1:'o', 2:'o', 3:'x', 4:'x'}
    cmapstuff=['orange', 'darkolivegreen', 'darkviolet', 'dodgerblue','deeppink', 'saddlebrown', 'dimgrey','cornflowerblue','plum']
    tmp=0



    for ui, si in zip(uorg, sess):
        if idx_s is not None:
            if((si==1 and tmp in idx_s) or (si==2 and tmp-len(source) in idx_t)):
                if pairwise:
                    if (si==1):
                        itmp = np.where(idx_s == tmp)[0][0]
                    if (si==2):
                        itmp = np.where(idx_t == tmp-len(source))[0][0]
                    ax1.scatter(ui[1], ui[2], facecolor=cmapstuff[itmp % len(cmapstuff)], edgecolor='none',alpha=0.9, marker=markers[si])
                else:
                    ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none',alpha=0.9, marker=markers[si])
            else:
                ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.1, marker=markers[si])
        else:
            ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.8, marker=markers[si])
        tmp+=1


    ax1.scatter([], [], facecolor=colors[1], label='\\textit{source (student)}')
    ax1.scatter([], [], facecolor=colors[2], label='\\textit{target (teacher)}')

    #ax1.legend(loc='lower center')
    return ax1


def plot_diffusion_embedding_target_new(source, target_new, ax1):
    covs = np.concatenate([source, target_new])
    sess = np.array([1] * len(source) + [2] * len(target_new))

    colors = {1: 'blue', 2: 'red'}

    uorg, l = get_diffusionEmbedding(points=covs, distance=distance_riemann)

    for ui, si in zip(uorg, sess):
        ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.7)
    ax1.scatter([], [], facecolor=colors[1], label='\\textit{ground truth (student)}')
    ax1.scatter([], [], facecolor=colors[2], label='\\textit{target new (teacher)}')
    #ax1.legend(loc='lower right')
    return ax1


def plot_diffusion_embedding_target_new_and_naive(groundtruth, target_icp, target_naive, ax1):
    covs = np.concatenate([groundtruth, target_icp])
    covs = np.concatenate([covs, target_naive])
    sess = np.array([1] * len(groundtruth) + [2] * len(target_icp) + [3] * len(target_naive))

    colors = {1: 'blue', 2: 'purple', 3: 'orange'}

    uorg, l = get_diffusionEmbedding(points=covs, distance=distance_riemann)

    for ui, si in zip(uorg, sess):
        ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.7)
    ax1.scatter([], [], facecolor=colors[1], label='\\textit{ground truth}')
    ax1.scatter([], [], facecolor=colors[2], label='\\textit{target mapped ICP}')
    ax1.scatter([], [], facecolor=colors[3], label='\\textit{target mapped naive}')
    #ax1.legend(loc='lower right')
    return ax1