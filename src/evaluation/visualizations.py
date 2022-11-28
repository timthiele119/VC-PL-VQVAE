from matplotlib import pyplot as plt
import numpy as np
from src.models.vqvae import VQVAE
import torch
from torch.utils.data.dataset import Dataset


def plot_original_reconstruction_matrix(grid_x: int, grid_y: int, model: VQVAE, dataset: Dataset, device: str = "cpu"):
    """
    Randomly selects grid_x * grid_y examples from the dataset and plots the examples against their corresponding
    VQ-VAE reconstruction
    :param grid_x: Number of examples in the horizontal direction
    :param grid_y: Number of examples in the vertical direction
    :param model: VQ-VAE model
    :param dataset: Dataset
    :param device: Cuda / CPU device
    :return:
    """
    with torch.no_grad():
        model = model.to(device)
        n_imgs = grid_x * grid_y
        indices = list(np.random.choice(len(dataset), size=n_imgs, replace=False))
        originals = torch.cat([dataset[idx] for idx in indices], dim=0).unsqueeze(dim=1).to(device)
        out = model(originals)
        reconstructions = out["reconstructions"]
        originals_reconstructions = torch.cat([originals, reconstructions], dim=3)
        originals_reconstructions = torch.clip(originals_reconstructions, 0.0, 1.0)
        originals_reconstructions = torch.permute(originals_reconstructions, (0, 2, 3, 1))
        originals_reconstructions = originals_reconstructions.to("cpu")

        fig, axes = plt.subplots(figsize=(grid_x*4, grid_y*2), nrows=grid_x, ncols=grid_y, sharex=True, sharey=True)
        for x in range(grid_x):
            for y in range(grid_y):
                axes[x, y].imshow(originals_reconstructions[x*5+y], cmap="gray")

        return fig, axes
