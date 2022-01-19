import numpy as np
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rpa.diffusion_map import get_diffusionEmbedding
from pyriemann.utils.distance import distance_riemann
from rpa.helpers.transfer_learning.utils import mean_riemann


#
# TSNE embedding
#

def embed_2d(source, target, ax):
    if (source.ndim == 3 and target.ndim == 3):
        source = source.reshape((source.shape[0], source.shape[1] * source.shape[1]))
        target = target.reshape((target.shape[0], target.shape[1] * target.shape[1]))

    covs = np.concatenate([source, target])
    sess = np.array([1] * len(source) + [2] * len(target))

    X_embedded = TSNE(n_components=2, perplexity=50, n_iter=5000,
                      init='random').fit_transform(covs)

    #np.savetxt(path+'/tsne.csv', X_embedded, delimiter=',')

    colors = {1: 'b', 2: 'r'}
    for ui, si in zip(X_embedded, sess):
        ax.scatter(ui[0], ui[1], facecolor=colors[si], edgecolor='none')
    return ax


def embed_2d_target_new(source, target, target_new, ax):
    if (source.ndim == 3):
        source = source.reshape((source.shape[0], source.shape[1] * source.shape[1]))
    if (target.ndim == 3):
        target = target.reshape((target.shape[0], target.shape[1] * target.shape[1]))
    if (target_new.ndim == 3):
        target_new = target_new.reshape((target_new.shape[0], target_new.shape[1] * target_new.shape[1]))

    covs = np.concatenate([source, target])
    covs = np.concatenate([covs, target_new])
    sess = np.array([1] * len(source) + [2] * len(target) + [3] * len(target_new))
    colors = {1: 'blue', 2: 'green', 3: 'red'}

    X_embedded = TSNE(n_components=2, perplexity=50, n_iter=5000,
                      init='random').fit_transform(covs)

    for ui, si in zip(X_embedded, sess):
        ax.scatter(ui[0], ui[1], facecolor=colors[si], edgecolor='none')
    ax.scatter([], [], facecolor=colors[1], label='source (robot)')
    ax.scatter([], [], facecolor=colors[2], label='target (human)')
    ax.scatter([], [], facecolor=colors[3], label='target new (human)')
    ax.legend(loc='lower right')
    return ax


def embed_2d_target_new_and_naive(source, target, target_new, target_naive, ax):
    if (source.ndim == 3):
        source = source.reshape((source.shape[0], source.shape[1] * source.shape[1]))
    if (target.ndim == 3):
        target = target.reshape((target.shape[0], target.shape[1] * target.shape[1]))
    if (target_new.ndim == 3):
        target_new = target_new.reshape((target_new.shape[0], target_new.shape[1] * target_new.shape[1]))
    if (target_naive.ndim == 3):
        target_naive = target_naive.reshape((target_naive.shape[0], target_naive.shape[1] * target_naive.shape[1]))

    covs = np.concatenate([source, target])
    covs = np.concatenate([covs, target_new])
    covs = np.concatenate([covs, target_naive])
    sess = np.array([1] * len(source) + [2] * len(target) + [3] * len(target_new) + [4] * len(target_naive))
    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange'}

    X_embedded = TSNE(n_components=2, perplexity=50, n_iter=5000,
                      init='random').fit_transform(covs)

    for ui, si in zip(X_embedded, sess):
        ax.scatter(ui[0], ui[1], facecolor=colors[si], edgecolor='none')
    ax.scatter([], [], facecolor=colors[1], label='source (robot)')
    ax.scatter([], [], facecolor=colors[2], label='target (human)')
    ax.scatter([], [], facecolor=colors[3], label='target mapped ICP (human)')
    ax.scatter([], [], facecolor=colors[4], label='target mapped naive (human)')
    ax.legend(loc='lower right')
    return ax


#
# Diffusion maps
#

def plot_diffusion_embedding(source, target, ax1, idx_s=None, idx_t=None):
    covs = np.concatenate([source, target, [mean_riemann(source)], [mean_riemann(target)]])
    sess = np.array([1] * len(source) + [2] * len(target) + [3] * 1 + [4] * 1)

    uorg, l = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    #np.savetxt(path+'/diffusion_embedding.csv', uorg, delimiter=',')

    colors = {1: 'b', 2: 'r',3: 'b', 4: 'r'}
    markers = {1:'o', 2:'o', 3:'x', 4:'x'}
    tmp=0

    for ui, si in zip(uorg, sess):
        if idx_s is not None:
            if((si==1 and tmp in idx_s) or (si==2 and tmp-len(source) in idx_t)):
                ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.7, marker=markers[si])
            else:
                ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.2, marker=markers[si])
        else:
            ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none', alpha=0.8, marker=markers[si])
        tmp+=1


    ax1.scatter([], [], facecolor=colors[1], label='source (robot)')
    ax1.scatter([], [], facecolor=colors[2], label='target (human)')

    ax1.legend(loc='lower center')
    return ax1


def plot_diffusion_embedding_target_new(source, target, target_new, ax1):
    covs = np.concatenate([source, target])
    covs = np.concatenate([covs, target_new])
    sess = np.array([1] * len(source) + [2] * len(target) + [3] * len(target_new))

    colors = {1: 'blue', 2: 'green', 3: 'red'}

    uorg, l = get_diffusionEmbedding(points=covs, distance=distance_riemann)

    for ui, si in zip(uorg, sess):
        ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none')
    ax1.scatter([], [], facecolor=colors[1], label='source (robot)')
    ax1.scatter([], [], facecolor=colors[2], label='target (human)')
    ax1.scatter([], [], facecolor=colors[3], label='target new (human)')
    ax1.legend(loc='lower right')
    return ax1


def plot_diffusion_embedding_target_new_and_naive(source, target, target_new, target_naive, ax1):
    covs = np.concatenate([source, target])
    covs = np.concatenate([covs, target_new])
    covs = np.concatenate([covs, target_naive])
    sess = np.array([1] * len(source) + [2] * len(target) + [3] * len(target_new) + [4] * len(target_naive))

    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange'}

    uorg, l = get_diffusionEmbedding(points=covs, distance=distance_riemann)

    for ui, si in zip(uorg, sess):
        ax1.scatter(ui[1], ui[2], facecolor=colors[si], edgecolor='none')
    ax1.scatter([], [], facecolor=colors[1], label='source (robot)')
    ax1.scatter([], [], facecolor=colors[2], label='target (human)')
    ax1.scatter([], [], facecolor=colors[3], label='target mapped ICP (human)')
    ax1.scatter([], [], facecolor=colors[4], label='target mapped naive (human)')
    ax1.legend(loc='lower right')
    return ax1