import os
import warnings
from typing import List

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid

from helpers.display.matplotlib_imshow import matplotlib_imshow


def normalize(v):
    """Normalizes input array to a range of [0, 1]

    Args:
        v (np.array): input array

    Returns:
        np.array: normalized array
    """
    vmin = v.min()
    vmax = v.max()
    v = (v - vmin) / (vmax - vmin)
    return v


def save_gifs(
    image: np.array,
    save_dir: str,
    ds_factor: int = 2,
    fps: int = 24,
    flip_image: bool = True,
    overwrite: bool = True,
):
    """Creates and saves three orthogonally progressing gifs from a three-dimensional input tensor.

    Args:
        image (np.array): Three-dimensional input image tensor.
        save_dir (str): Directory to save the images in.
        ds_factor (int, optional): Downsampling factor along the progression axis.
            This can be used to slim down and/or speed up the resulting gifs. Defaults to 2.
        fps (int, optional): Frames per second of the resulting gifs. Defaults to 24.
        flip_image (bool, optional): If set to True, images are flipped along vertical axis. Defaults to True.
        overwrite (bool, optional): If set to True, potentially existing images in save_dir are overwritten. Defaults to True.
    """
    # check for correct dimensionality and edge cases in fps and ds_factor
    if len(image.shape) != 3:
        raise ValueError(f"Input tensor should be of type DHW but found tensor of type {type(image)} and of shape {image.shape}.")

    if ds_factor < 1:
        raise ValueError(f"Downsampling factor cannot be smaller than 1 but received {ds_factor}.")

    if fps < 1:
        raise ValueError(f"Cannot set less than one frame per second but received {fps}.")

    # create directory if needed
    os.makedirs(save_dir, exist_ok=overwrite)

    # normalize image tensor to [0,1] and convert to uint8
    full_img_norm = np.uint8(normalize(image) * 255)

    # convert tensor to list of 2D arrays along corresponding axis and save as gif
    img_list = [full_img_norm[i] for i in range(len(full_img_norm))]
    imageio.mimsave(
        os.path.join(save_dir, "transverse.gif"),
        img_list[::ds_factor],
        fps=fps,
        subrectangles=True,
    )

    # invert vertical axis if flip_images is set to true
    if flip_image:
        img_list = [full_img_norm[::-1, i] for i in range(len(full_img_norm))]
    else:
        img_list = [full_img_norm[:, i] for i in range(len(full_img_norm))]
    imageio.mimsave(
        os.path.join(save_dir, "frontal.gif"),
        img_list[::ds_factor],
        fps=24,
        subrectangles=True,
    )

    if flip_image:
        img_list = [full_img_norm[::-1, :, i] for i in range(len(full_img_norm))]
    else:
        img_list = [full_img_norm[:, :, i] for i in range(len(full_img_norm))]
    imageio.mimsave(
        os.path.join(save_dir, "sagittal.gif"),
        img_list[::ds_factor],
        fps=24,
        subrectangles=True,
    )


def model_generate_images(model: torch.nn.Module, device: str, num_images: int, n_classes: int = None):
    """Takes a pytorch model generating 3d images.

    Args:
        model (torch.nn.Module): Model used for the image generation. Expects the model to output image tensors of shape BCDHW.
            Requires any of the following methods:
            - "generate"
            - "generate_low_res"
            If generate and generate_low_res are both available, a tuple of generated outputs is returned.
            If none of the above are available a "forward" call that can be accessed via self(z) is required.
            Additionally, a latent_dim member variable is expected, defining the shape of the latent vector input.
        device (str): device on which to perform the inference. Should either be 'cuda' or 'cpu'.
        num_images (int, optional): Number of images to create for the grids. Defaults to 16.
        n_classes (int, optional): Sets the number of classes to be displayed for conditional models. Defaults to None.


    Returns:
        torch.Tensor: Image tensor of shape BCDHW.
    """
    # check for faulty configuration:
    if num_images < 1:
        raise ValueError(f"Number of images must be at least 1, but received {num_images}")

    try:
        latent_dim = model.latent_dim
    except AttributeError:
        raise AttributeError('This function expects the model to have a "latent_dim" member variable, which was not found.')

    if latent_dim < 1:
        raise ValueError(f"A latent dimension of at least 1 is required, but found {latent_dim}")

    # create noise vectors
    z = torch.randn(num_images, latent_dim, device=device)

    if n_classes is not None:
        # make sure the number of images is a multiple of the number of classes
        if num_images % n_classes != 0:
            raise ValueError(f"{n_classes} conditions defined, but {num_images} images were set. This must be a multiple of the number of conditions.")

        # create condition vector
        if hasattr(model, "classes_per_condition"):
            c = torch.stack(
                [torch.randint(0, num_classes, (int(num_images / n_classes),)) for num_classes in model.classes_per_condition],
                1,
            ).to(device)
        else:
            c = torch.arange(0, n_classes, 1, device=device).unsqueeze(-1).repeat(1, int(num_images / n_classes)).view(num_images, 1)

        # create tuple out of noise and condition as model input
        z = (z, c)

    # generate images
    generated_imgs = None

    # check for generate method in the model and use that instead of the forward call if available.
    if hasattr(model, "generate"):
        # instantiate list for hr-images.
        generated_high_res_imgs = []

        # High-res processing is more memory-intensive and needs to be done one by one. Generate image and append to list.
        for i in range(num_images):
            with torch.no_grad():
                z_single = torch.randn(1, latent_dim, device=device)
                if n_classes is not None:
                    z_single = (z_single, c[i : i + 1])
                img = model.generate(z_single)
                generated_high_res_imgs.append(img)

        # Stack images to tensor.
        generated_imgs = torch.cat(generated_high_res_imgs, dim=0)

    # check for generate_low_res method and generate images if applicable
    if hasattr(model, "generate_low_res"):
        with torch.no_grad():
            generated_imgs_lr = model.generate_low_res(z)

        # check if generated_imgs has already been populated by model.generate. In that case create tuple, else populate variable
        if generated_imgs is not None:
            generated_imgs = (generated_imgs, generated_imgs_lr)
        else:
            generated_imgs = generated_imgs_lr

    # if neither of the above methods were available, use the forward call of the model to generate images
    if generated_imgs is None:
        with torch.no_grad():
            generated_imgs = model(z)

    return generated_imgs


# TODO: align n_per_row positional argument with save_images_from_model


def save_image_grid(
    images: torch.Tensor,
    n_per_row: int,
    save_dir: str = None,
    file_name: str = "slices_grid.png",
    add_transposed: bool = True,
    transposed_file_name_prefixes: "tuple[str, str]" = ["frontal_", "sag_"],
    flip_images: bool = False,
    overwrite: bool = True,
    vmin: int = 0,
    vmax: int = 1,
    cmap: str = "Greys_r",
    slice_offset: int = 0,
    **kwargs,
):
    """Takes a pytorch tensors containing 3d images of shape BCDHW and saves grids of center slices.

    Args:
        images (torch.Tensor): Input tensor in the shape BCDHW.
        n_per_row (int, optional): Images per row of the grid. Defaults to 4.
        save_dir (str): Directory in which to save the images.
        file_name (str, optional): Name for the saved file. Defaults to 'slices_grid.png'.
        add_transposed (bool, optional): Additionally saves grids of center slices along the remaining two axes. Defaults to True.
        transposed_file_name_prefixes: (tuple(str, str), optional): File name prefixes for the transposed images. Defaults to ["frontal_", "sagittal_"].
        flip_images (bool, optional): If True, transposed images are flipped along the vertical axis. Defaults to False.
        overwrite (bool, optional): If set to True, potentially existing images in save_dir are overwritten. Defaults to True.
        vmin (int, optional): Bottom end of the display range. Defaults to 0.
        vmax (int, optional): Top end of the display range. Defaults to 1.
        cmap (str, optional): Color-map for the image. Any matplotlib-compatible string is accepted. Defaults to 'Greys_r'.
        slice_offset (int, optional): offset from the center slice to be applied to the images. Can be global or a list of length len(images). Defaults to 0
        kwargs (any, optional): Any keyword arguments than can be passed to matplotlib.pyplot.imshow.

    """
    # TODO: currently, rgb images are detected automatically by channel count. What is with images containing 3 channels that are not RGB?
    # Perhaps this should be set explicitly.

    num_images = images.shape[0]

    # check for faulty configuration:
    if n_per_row < 1:
        raise ValueError(f"Number of images per row must be at least 1, but received {num_images}")

    if num_images < n_per_row:
        raise ValueError(
            f"Number of images per row must be smaller than total number of images, but received {n_per_row} images per row for a total of {num_images}."
        )

    # make sure the number of images evenly fits in the desired number of rows
    if not ((num_images / n_per_row) % 1) == 0:
        raise ValueError(f"Cannot divide {num_images} into {n_per_row}.")

    # create folder if necessary
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=overwrite)

    # save images
    # check if rgb or greyscale
    is_rgb = images.shape[1] == 3

    # check if slice offset needs to be processed:
    if isinstance(slice_offset, int):
        grid_images = images[:, :, (images.shape[2] // 2) + slice_offset, :, :]
    elif isinstance(slice_offset, list):
        assert len(slice_offset) == len(images), f"Number of slice offsets does not match number of images. Received {len(slice_offset)} and {len(images)}."
        grid_images = torch.vstack(
            [images[image_idx : image_idx + 1, :, (images.shape[2] // 2) + offset, :, :] for image_idx, offset in enumerate(slice_offset)]
        )

    # create image grid
    grid_h = make_grid(grid_images, nrow=n_per_row, normalize=True, scale_each=True)

    if not is_rgb:
        grid_h = grid_h.sum(0, keepdim=True) / 3

    one_chan = not is_rgb

    # convert from torch to numpy/matplotlib and save
    matplotlib_imshow(
        grid_h,
        one_channel=one_chan,
        save_path=os.path.join(save_dir, file_name) if save_dir is not None else None,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )

    if add_transposed:
        # transpose images for frontal and sagittal slices
        axes = [[3, 4], [4, 3]]
        for axis, prefix in zip(axes, transposed_file_name_prefixes):
            transposed_imgs = torch.permute(images, (0, 1, axis[0], 2, axis[1]))
            if flip_images:
                transposed_imgs = torch.flip(transposed_imgs, (3,))
            if axis == [3, 4]:
                grid_transposed = make_grid(
                    transposed_imgs[:, :, transposed_imgs.shape[2] // 2, :, :],
                    nrow=n_per_row,
                    normalize=True,
                    scale_each=True,
                )
            else:  # slice through lung instead of spine
                grid_transposed = make_grid(
                    transposed_imgs[:, :, transposed_imgs.shape[2] // 2, :, :],
                    nrow=n_per_row,
                    normalize=True,
                    scale_each=True,
                )
            if not is_rgb:
                # TODO: check if using just one of the three channels works as well.
                grid_transposed = grid_transposed.sum(0, keepdim=True) / 3
            this_file_name = prefix + file_name
            matplotlib_imshow(
                grid_transposed,
                one_channel=one_chan,
                save_path=os.path.join(save_dir, this_file_name) if save_dir is not None else None,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                **kwargs,
            )


def save_images_from_model(
    model: torch.nn.Module,
    save_dir: str,
    file_name_list: List[str] = ["slices_grid.png"],
    add_transposed: bool = True,
    transposed_file_name_prefixes: "tuple[str, str]" = ["frontal_", "sagittal_"],
    num_images: int = 16,
    n_per_row: int = 4,
    flip_images: bool = False,
    overwrite: bool = True,
    n_classes: int = None,
    vmin: int = 0,
    vmax: int = 1,
    cmap: str = "Greys_r",
    **kwargs,
):
    """Takes a pytorch model generating 3d images and saves grids of center slices along the three axes.
    Args:
        model (torch.nn.Module): Model used for the image generation. Expects the model to output either BCDHW or (BCDHW, BCDHW for dual-path models.)
        save_dir (str): Directory in which to save the images.
        file_name_list (str, optional): List of names for the saved files. Defaults to ['slices_grid.png'].
            The number of names must match the number of outputs of the model.
        transposed_file_name_prefixes: (tuple(str, str), optional): File name prefixes for the transposed images. Defaults to ["frontal_", "sagittal_"].
        num_images (int, optional): Number of images to create for the grids. Defaults to 16.
        n_per_row (int, optional): Images per row of the grid. Defaults to 4.
        flip_images (bool, optional): If True, images are flipped along the vertical axis. Defaults to True.
        overwrite (bool, optional): If set to True, potentially existing images in save_dir are overwritten. Defaults to True.
        n_classes (int, optional): Sets the number of classes to be displayed for conditional models. Should equal number of rows in grid. Defaults to None.
        vmin (int, optional): Bottom end of the display range. Defaults to 0.
        vmax (int, optional): Top end of the display range. Defaults to 1.
        cmap (str, optional): Color-map for the image. Any matplotlib-compatible string is accepted. Defaults to 'Greys_r'.
        kwargs (any, optional): Any keyword arguments than can be passed to matplotlib.pyplot.imshow.
    """
    # TODO: This could be expanded to handle more input types. Such as BDHW or BCHW.

    # make sure the number of images evenly fits in the desired number of rows
    if n_per_row == 0:
        raise ValueError(f"Number of images per row cannot be zero, but found {n_per_row}.")
    if not ((num_images / n_per_row) % 1) == 0:
        raise ValueError(f"Cannot divide {num_images} into {n_per_row}.")

    # see if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # generate images
    generated_images = model_generate_images(model, device, num_images, n_classes)

    # check if model outputs single tensor or list/tuple of tensors.
    if isinstance(generated_images, torch.Tensor):
        generated_images = [generated_images]

    if len(file_name_list) != len(generated_images):
        raise ValueError(f"Number of supplied file names does not match the {len(generated_images)} outputs returned by the model. Received {file_name_list}.")

    # iterate over all generated image tensors
    for images, file_name in zip(generated_images, file_name_list):
        # save images
        save_image_grid(
            images,
            n_per_row,
            save_dir,
            file_name,
            add_transposed,
            transposed_file_name_prefixes,
            flip_images,
            overwrite,
            vmin,
            vmax,
            cmap,
            **kwargs,
        )


def make_horiz_and_vert_slices(sample_imgs: torch.Tensor, ignore_anisotropy: bool = False):
    """Extracts center slices from 3d pytorch tensor.

    Args:
        sample_imgs (torch.Tensor): input tensor of shape BCDHW
        ignore_anisotropy (bool, optional): By default, subvolumes of a few slices are not displayed in the horizontal axis.
        If set to True, this behavior is overridden. Defaults to False.

    Returns:
        torch.Tensor: Returns center slices of images. Either B vertical center slices or 2*B images,
            the first half of which are vertical slices and the second half are horizontal slices.
    """

    # extract vertical slices.
    vert_slices = sample_imgs[:, :, sample_imgs.shape[2] // 2]

    # if tensor is a cube or ignore anisotropy is True: extract horizontal center slices and concatenate with vertical slices
    if sample_imgs.shape[2] == sample_imgs.shape[3] or ignore_anisotropy:
        horz_images = torch.transpose(sample_imgs, 2, 3)
        horiz_slices = horz_images[:, :, sample_imgs.shape[2] // 2]
        out = torch.cat([vert_slices, horiz_slices], dim=0)

    # otherwise: output warning and return only vertical center slices.
    else:
        warnings.warn("Anisotropy detected in input volume. Only returning center slice. Check input shape or set ignore_anisotropy=True.")
        out = vert_slices

    return out
