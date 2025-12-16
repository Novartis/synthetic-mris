import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from PIL import Image

# import imageio


def plot_overlay(
    bg_image_file: str,
    overlay_image_file: str,
    alpha: int = 0.25,
    save_file: str = "",
    figsize: tuple = (50, 200),
    plane: str = "transverse",
    slices_to_plot: list = [],
    num_slices: int = 5,
    delta: int = 70,
):
    fig, ax = plt.subplots(1, figsize=figsize)
    bg_image = nib.load(bg_image_file).get_fdata()
    overlay_image = nib.load(overlay_image_file).get_fdata()
    this_img_shape = bg_image.shape

    if plane == "sagittal":
        if not len(slices_to_plot) > 0:
            slices_to_plot = np.linspace(delta, this_img_shape[0] - delta, num=num_slices)
        ax.imshow(
            np.hstack([np.rot90(bg_image[int(this), :, :], k=1, axes=(0, 1)) for this in slices_to_plot]),
            cmap="gray",
        )
        ax.imshow(
            np.hstack([np.rot90(overlay_image[int(this), :, :], k=1, axes=(0, 1)) for this in slices_to_plot]),
            cmap="jet",
            alpha=alpha,
        )
    elif plane == "coronal":
        if not len(slices_to_plot) > 0:
            slices_to_plot = np.linspace(delta, this_img_shape[1] - delta, num=num_slices)

        ax.imshow(
            np.hstack([np.rot90(bg_image[:, int(this), :], k=1, axes=(0, 1)) for this in slices_to_plot]),
            cmap="gray",
        )
        ax.imshow(
            np.hstack([np.rot90(overlay_image[:, int(this), :], k=1, axes=(0, 1)) for this in slices_to_plot]),
            cmap="jet",
            alpha=alpha,
        )

    elif plane == "transverse":
        if not len(slices_to_plot) > 0:
            slices_to_plot = np.linspace(delta, this_img_shape[2] - delta, num=num_slices)
        ax.imshow(
            np.hstack([np.rot90(bg_image[:, :, int(this)], k=1, axes=(0, 1)) for this in slices_to_plot]),
            cmap="gray",
        )
        ax.imshow(
            np.hstack([np.rot90(overlay_image[:, :, int(this)], k=1, axes=(0, 1)) for this in slices_to_plot]),
            cmap="jet",
            alpha=alpha,
        )
    else:
        print("plane not implemented")

    plt.style.use("dark_background")
    plt.subplots_adjust(left=0.1, bottom=0.001, right=0.9, top=0.09, wspace=0.04, hspace=0.04)
    fig.show()
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        return None


def plot_more_slices_transverse_plane(
    input_file_list: list,
    save_file: str = None,
    figsize: tuple = (50, 200),
    slices_to_plot: list = [],
    num_slices: int = 5,
    delta: int = 70,
) -> None:
    """plots several slices in top-view of the images

    Args:
        input_file (str): mri file to plot
        save_file (str, optional): full path to save the image. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (50, 200).
        slices_to_plot (list, optional): list of slices to be plotted, if not defined it is
            computed using num_slices & delta
        num_slices (int, optional): number of slices to plot. Defaults to 5.
        delta (int, optional): delta between slices. Defaults to 70.

        NOTE: num_slices and delta are used to compute the slices to plot
            slices_to_plot = np.linspace(delta, this_img_shape[i_dim]-delta, num=num_slices)

    Returns:
        None: saved figure if save_file provided
    """

    # input_file_np = ants.image_read(fixed_file).numpy()

    fig, ax = plt.subplots(len(input_file_list), figsize=figsize)
    # fig = plt.figure(figsize=figsize, dpi= 80)

    for idx, input_file in enumerate(input_file_list):
        input_file_np = nib.load(input_file).get_fdata()

        this_img_shape = input_file_np.shape
        if not len(slices_to_plot) > 0:
            slices_to_plot = np.linspace(delta, this_img_shape[2] - delta, num=num_slices)

        third_view = np.hstack([np.rot90(input_file_np[:, :, int(this)], k=1, axes=(0, 1)) for this in slices_to_plot])
        if len(input_file_list) == 1:
            ax.imshow(third_view, cmap="gray")
        else:
            ax[idx].imshow(third_view, cmap="gray")

    plt.style.use("dark_background")
    # fig.tight_layout() # to set the proper space in subplots
    # setting space manually:
    plt.subplots_adjust(left=0.1, bottom=0.001, right=0.9, top=0.09, wspace=0.04, hspace=0.04)

    fig.show()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        return None


def plot_more_slices_coronal_plane(
    input_file_list: list,
    save_file: str = None,
    figsize=(50, 200),
    slices_to_plot: list = [],
    num_slices: int = 5,
    delta: int = 70,
) -> None:
    """plots several slices in coronal_plane

    Args:
        input_file (str): mri file to plot
        save_file (str, optional): full path to save the image. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (50, 200).
        slices_to_plot (list, optional): list of slices to be plotted, if not defined it is
            computed using num_slices & delta
        num_slices (int, optional): number of slices to plot. Defaults to 5.
        delta (int, optional): delta between slices. Defaults to 70.

        NOTE: num_slices and delta are used to compute the slices to plot
            slices_to_plot = np.linspace(delta, this_img_shape[i_dim]-delta, num=num_slices)

    Returns:
        None: saved figure if save_file provided
    """

    # input_file_np = ants.image_read(fixed_file).numpy()

    fig, ax = plt.subplots(len(input_file_list), figsize=figsize)
    # fig = plt.figure(figsize=figsize, dpi= 80)

    for idx, input_file in enumerate(input_file_list):
        input_file_np = nib.load(input_file).get_fdata()

        this_img_shape = input_file_np.shape
        if not len(slices_to_plot) > 0:
            slices_to_plot = np.linspace(delta, this_img_shape[1] - delta, num=num_slices)

        second_view = np.hstack([np.rot90(input_file_np[:, int(this), :], k=1, axes=(0, 1)) for this in slices_to_plot])
        ax[idx].imshow(second_view, cmap="gray")

    plt.style.use("dark_background")
    # fig.tight_layout() # to set the proper space in subplots
    # setting space manually:
    plt.subplots_adjust(left=0.1, bottom=0.001, right=0.9, top=0.09, wspace=0.04, hspace=0.04)

    fig.show()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        return None


def plot_more_slices_sagittal_plane(
    input_file_list: list,
    save_file: str = None,
    figsize=(50, 200),
    slices_to_plot: list = [],
    num_slices: int = 5,
    delta: int = 70,
) -> None:
    """plots several slices in saggital plane

    Args:
        input_file (str): mri file to plot
        save_file (str, optional): full path to save the image. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (50, 200).
        slices_to_plot (list, optional): list of slices to be plotted, if not defined it is
            computed using num_slices & delta
        num_slices (int, optional): number of slices to plot. Defaults to 5.
        delta (int, optional): delta between slices. Defaults to 70.

        NOTE: num_slices and delta are used to compute the slices to plot
            slices_to_plot = np.linspace(delta, this_img_shape[i_dim]-delta, num=num_slices)

    Returns:
        None: saved figure if save_file provided
    """

    # input_file_np = ants.image_read(fixed_file).numpy()

    fig, ax = plt.subplots(len(input_file_list), figsize=figsize)
    # fig = plt.figure(figsize=figsize, dpi= 80)

    for idx, input_file in enumerate(input_file_list):
        input_file_np = nib.load(input_file).get_fdata()

        this_img_shape = input_file_np.shape
        if not len(slices_to_plot) > 0:
            slices_to_plot = np.linspace(delta, this_img_shape[0] - delta, num=num_slices)

        second_view = np.hstack([np.rot90(input_file_np[int(this), :, :], k=1, axes=(0, 1)) for this in slices_to_plot])
        ax[idx].imshow(second_view, cmap="gray")

    plt.style.use("dark_background")
    # fig.tight_layout() # to set the proper space in subplots
    # setting space manually:
    plt.subplots_adjust(left=0.1, bottom=0.001, right=0.9, top=0.09, wspace=0.04, hspace=0.04)

    fig.show()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        return None


def combine_png(png_list: list, output_file: str, mode="") -> None:
    """combines vertically the list of images in the output file

    Args:
        images_list (list of string): list of images to combibe
        output (str): path to the combine output figure
    """

    # check if input files exists:
    assert all([os.path.isfile(this) for this in png_list])

    if mode == "vertical":
        combine_png_verical(png_list, output_file)
    else:
        combine_png_horizontal(png_list, output_file)

    return


def combine_png_horizontal(plot_ls: list, destin_path: str) -> None:
    """given a list of images in plot_ls combine them in a unique png

    Args:
        plot_ls (_type_): list of paths of images to combine
        destin_path (_type_): destination path/image.png where to save the combined plot
    """
    sns.set_style("white")
    plt.figure(figsize=(60, 45), dpi=80)
    images = [Image.open(x) for x in plot_ls]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(destin_path)


def combine_png_verical(images_list: list, output: str):
    """combines vertically the list of images in the output file

    Args:
        images_list (list of string): list of images to combibe
        output (str): path to the combine output figure
    """

    imgs = [Image.open(i) for i in images_list]

    # If you're using an older version of Pillow, you might have to use .size[0] instead of .width
    # and later on, .size[1] instead of .height
    min_img_width = min(i.width for i in imgs)

    total_height = 0
    for i, img in enumerate(imgs):
        # If the image is larger than the minimum width, resize it
        if img.width > min_img_width:
            imgs[i] = img.resize(
                (min_img_width, int(img.height / img.width * min_img_width)),
                Image.ANTIALIAS,
            )
        total_height += imgs[i].height

    # I have picked the mode of the first image to be generic. You may have other ideas
    # Now that we know the total height of all of the resized images, we know the height of our final image
    img_merge = Image.new(imgs[0].mode, (min_img_width, total_height))
    y = 0
    for img in imgs:
        img_merge.paste(img, (0, y))
        y += img.height

    os.makedirs(os.path.dirname(output), exist_ok=True)
    img_merge.save(output)
