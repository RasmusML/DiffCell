import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import seaborn as sns

import dataset

def plot_image(image: torch.Tensor, path=None):
    plt.imshow(_plot_permute(image))
    
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_channels(image: torch.Tensor, path=None):
    titles = ["dapi", "tubulin", "actin"]
    
    plt_view = _plot_permute(image)
    fig, axs = plt.subplots(1, 3, figsize=(16,6))
    for i in range(3):
        ax = axs[i]
        c = torch.zeros_like(plt_view)
        c[:,:,i] = plt_view[:,:,i]
        ax.set_axis_off()
        ax.imshow(c)
        ax.set_title(titles[i])
    
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_MOA_distribution(metadata: pd.DataFrame, path=None):
    moa_count = {}
    
    moa_types = dataset.get_all_MOA_types()
    for moa in moa_types:
        moa_count[moa] = 0
    
    some_moa_types, some_counts = np.unique(metadata["moa"].to_numpy(), return_counts=True)
    for i, moa in enumerate(some_moa_types):
        moa_count[moa] = some_counts[i]
    
    counts = [moa_count[moa] for moa in moa_types]
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    if not isinstance(ax, plt.Axes):
        raise Exception("ax is not a plt.Axes")
    
    ax.set_ylabel("Percentage (%)")
    ax.grid()
    ax.set_title("Mechanism of Action (MOA) distribution")
    ax.bar(moa_types, counts / np.sum(counts) * 100)
    for tick in ax.get_xticklabels(): # type: ignore
        tick.set_rotation(-90)
    
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_treatment_heatmap(metadata: pd.DataFrame, path=None):
    import itertools
    heatmap = {}
    
    moa_types = dataset.get_all_MOA_types()
    concentration_types = dataset.get_all_concentration_types()
    
    for moa, concentration in itertools.product(moa_types, concentration_types):
        heatmap[(moa, concentration)] = 0
    
    for i, row in metadata.iterrows():
        heatmap[(row["moa"], row["Image_Metadata_Concentration"])] += 1
        
    heatmap_matrix = np.empty((moa_types.shape[0], concentration_types.shape[0]))

    for i, moa in enumerate(moa_types):
        for j, concentration in enumerate(concentration_types):
            heatmap_matrix[i][j] = heatmap[(moa, concentration)]
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    if not isinstance(ax, plt.Axes):
        raise Exception("ax is not a plt.Axes")
    
    sns.heatmap(np.log10(heatmap_matrix + 1), linewidth=.5, cmap="crest", annot=True)
    
    ax.set_xticklabels(concentration_types) # type: ignore
    for tick in ax.get_xticklabels(): # type: ignore
        tick.set_rotation(90)
    
    ax.set_yticklabels(moa_types) # type: ignore
    for tick in ax.get_yticklabels(): # type:ignore
        tick.set_rotation(0)
    
    ax.set_title("Treatment count - log10 scale")
    ax.set_xlabel("Concentration (μmol/L)")
    ax.set_ylabel("Mechanism of Action (MOA)")
    
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_image_diff(image1: torch.Tensor, image2: torch.Tensor, path=None):
    diff = torch.abs(image1 - image2)
    plot_image(diff, path)


def _plot_permute(image: torch.Tensor) -> torch.Tensor:
    """ 3x68x68 -> 68x68x3 """
    assert image.shape[0] == 3
    return image.permute(1, 2, 0)


