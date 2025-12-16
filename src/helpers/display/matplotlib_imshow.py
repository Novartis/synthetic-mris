import matplotlib.pyplot as plt
import numpy as np
import torch


def matplotlib_imshow(
    img: torch.Tensor,
    save_path: str = None,
    one_channel: bool = True,
    vmin: int = 0,
    vmax: int = 1,
    cmap: str = "Greys_r",
    figsize: tuple = (15, 15),
):
    """Converts a pytorch tensor into a numpy array and displays or saves it as an image.

    Args:
        img (torch.tensor): Input tensor containing the image data in CHW format.
        save_path (str, optional): Save path for the created image. If None is supplied, the image is displayed instead. Defaults to None.
        one_channel (bool, optional): If True, axis 0 is averaged. Defaults to True.
        vmin (int, optional): Bottom end of the display range. Defaults to 0.
        vmax (int, optional): Top end of the display range. Defaults to 1.
        cmap (str, optional): Color-map for the image. Any matplotlib-compatible string is accepted. Defaults to 'Greys_r'.
        figsize (tuple, optional): Desired image size. Defaults to (15, 15).
    """
    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    if one_channel:
        # average over first dimension
        img = img.mean(dim=0)

    # convert image to numpy array
    npimg = img.cpu().numpy()

    if one_channel:
        ax.imshow(npimg, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(
            np.transpose(npimg, (1, 2, 0)),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

    plt.axis("off")

    # save image if save_path is supplied, otherwise display image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        return None
    else:
        return fig
