import matplotlib.pyplot as plt
import torch

def plot_image(image: torch.Tensor):
    plt.imshow(_plot_view(image))

def plot_channels(image: torch.Tensor):
    titles = ["dapi", "tubulin", "actin"]
    
    plt_view = _plot_view(image)
    fig, axs = plt.subplots(1, 3, figsize=(16,6))
    for i in range(3):
        ax = axs[i]
        c = torch.zeros_like(plt_view)
        c[:,:,i] = plt_view[:,:,i]
        ax.set_axis_off()
        ax.imshow(c)
        ax.set_title(titles[i])

def _plot_view(image: torch.Tensor) -> torch.Tensor:
    """
        3x68x68 -> 68x68x3
    """
    return image.permute(1, 2, 0)


