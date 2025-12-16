import glob
import itertools
import os
import pickle
import re
from typing import List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import data
from helpers.clustering.dim_reduction import dim_reduction
from helpers.display import plot_mri as plot_utils
from helpers.display.visualization_utils import save_image_grid


def _parse_checkpoint_filename(filename: str) -> Tuple[int, int]:
    """Parse a checkpoint filename"""
    m = re.match(r"epoch\=(\d+)-step=(\d+).ckpt", os.path.basename(filename))
    if m:
        return filename, int(m.group(1)), int(m.group(2))
    else:
        return filename, None, None


def list_checkpoints(path: str) -> List[Tuple[str, int, int]]:
    """List checkpoints in a folder, together with their epoch and step numbers

    Args:
        path (str): Path to directory containing checkpoints.

    Returns:
        tuple: tuple containing
            - str: checkpoint filename
            - int: model epoch
            - int: model step
    """

    if os.path.isfile(path):
        return [_parse_checkpoint_filename(path)]
    elif os.path.isdir(path):
        checkpoints_paths = glob.glob(path + "/**/epoch*.ckpt")
        return [_parse_checkpoint_filename(p) for p in checkpoints_paths if p[1] is not None and p[2] is not None]
    else:
        raise ValueError(f"Checkpoint path {path} cannot be used")


def get_checkpoint_file(main_path: str, epoch_num: int = None):
    checkpoints = list_checkpoints(main_path)
    if epoch_num:
        this_checkpoint = [this for this in checkpoints if this[1] == epoch_num]
        assert len(this_checkpoint) == 1, "more than one checkpoint selected: check epoch number"

    else:
        # get the last checkpoint
        max_epoch = max(p[1] for p in checkpoints)
        this_checkpoint = [this for this in checkpoints if this[1] == max_epoch]

    this_checkpoint_path = this_checkpoint[0][0]
    return this_checkpoint_path


def _upsample_low_res(low_res_img, upsample_factor: int):
    """upsample low resolution image

    Args:
        low_res_img (_type_): low resolution map
        upsample_factor (int): factor for upsample

    Returns:
        _type_: upsampled map

    for more details look at F.interpolate functions
    """
    with torch.no_grad():
        gen_images = (
            F.interpolate(
                low_res_img,
                scale_factor=upsample_factor,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    return gen_images


def generate_high_res_from_low_res_data(
    cfg: DictConfig,
    output_dir: str,
    num_samples: int,
    batch_size: int = 2,
    file_prefix: str = "",
) -> list:
    """Upsample low-resolution real data and generate high-resolution images.

    Args:
        cfg (DictConfig): Hydra configuration object containing settings and parameters.
        output_dir (str): Directory where upsampled data will be saved.
        num_samples (int): Number of samples to generate.
        batch_size (int, optional): Batch size used during processing. Defaults to 2.
        file_prefix (str, optional): Prefix to be added to the synthetic data filename. Defaults to ''.

    Returns:
        list: List of file paths to the generated high-resolution images.
    """

    # create directory if required
    os.makedirs(output_dir, exist_ok=True)
    # check if already generated:
    up_sampled_low_res_files = os.listdir(output_dir)

    if len(up_sampled_low_res_files) >= num_samples:
        logger.info(f"there are already {len(up_sampled_low_res_files)} synthetic data in folder:\n{output_dir}")
        # files_full = [os.path.join(synthetic_data_dir, i_file) for i_file in files ]
        # files_full_path = [i_file for i_file in files_full if os.path.isfile(i_file)]
        output_files_full_path = [i_file for i_file in [os.path.join(output_dir, i_file) for i_file in up_sampled_low_res_files] if os.path.isfile(i_file)]
        return output_files_full_path

    files = []  # set to default value for keeping values in the loop

    # loop over real data:
    # get real data:
    dataset_dir = cfg.data.paths.training_data
    real_train_dataset = data.BaseImageDataset(dataset_dir)

    train_loader = torch.utils.data.DataLoader(dataset=real_train_dataset, batch_size=batch_size, shuffle=False)
    current_file_count = 0
    for current_batch in tqdm(train_loader, desc="upsample low resolution from real data"):
        downsampled_batch = current_batch[:, ::4, ::4, ::4]
        if len(downsampled_batch.shape) == 4:
            downsampled_batch = torch.unsqueeze(downsampled_batch, 0)
        gen_images = _upsample_low_res(downsampled_batch, upsample_factor=4)
        file_prefix = "from_real_data_sample"

        # pickle and save
        for batch_index in range(batch_size):
            sample = {"image": gen_images[batch_index]}

            out_file = os.path.join(
                output_dir,
                file_prefix + f"_{(current_file_count + batch_index):05}" + "_" + ".pickle",
            )
            pickle.dump(
                sample,
                open(out_file, "wb"),
            )
            files.append(out_file)

        current_file_count += batch_size

    logger.success(f"Stored {num_samples} images in directory {output_dir}.")
    return files


def _process_high_res(z, model):
    with torch.no_grad():
        return model.generate(z)[:, 0].cpu().numpy()


def _process_low_res(z, upsample_factor, model):
    with torch.no_grad():
        gen_images = model.generate_low_res(z)
        gen_images = (
            F.interpolate(
                gen_images,
                scale_factor=upsample_factor,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(dim=1)
            .cpu()
            .numpy()
        )

        # gen_images = (
        #     F.interpolate(
        #         gen_images,
        #         scale_factor=upsample_factor,
        #         mode="trilinear",
        #         align_corners=False,
        #     )
        #     .squeeze()
        #     .cpu()
        #     .numpy()
        # )
    return gen_images


def generate_synthetic_data(
    checkpoint_path: str,
    synthetic_data_dir: str,
    num_samples: int,
    cfg: DictConfig,
    batch_size: int = 2,
    upsample_factor: int = 4,
    file_prefix: str = "",
    add_low_res_images: bool = False,
    **kwargs,
):
    """generates synthetic dataset using the pretrained generator model (saved in checkpoint_path)
        note: for the conditional case, here it uses random conditions.
        If you want to generate data with same conditions are real data, use function generate_synthetic_data_with_real_data_conditions
        synthetic data are saved as pickle in the synthetic_data_dir

    Args:
        checkpoint_path (str): path to the checkpoint to use for the trained model
        synthetic_data_dir (str): directory where synthesized data are saved
        cfg (DictConfig): hydra config file
        batch_size (int, optional): Batch size used during processing. Defaults to 1.
        upsample_factor (int, optional): Upsampling used for models not capable of directly
            synthesizing high-resolution images. Defaults to 4.
        file_prefix (str, optional): Prefix to be added to the synthetic data filename. Defaults to ''.
        add_low_res_images (bool, optional): Some high-resolution models can additionally output
            low-resolution images. This can be used to check if the low-resolution output is turned into
            a dataset as well. Defaults to False.
        update:bool=False, set to true if you want to remove previous generated data and create new once

    Returns:
        files (list): list of all synthetic data files (saves as pickle)
        low_res_files (list): list of all low resoltion synthetic data files (saves as pickle)
    """
    # check if GPU instance is being used and assign correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create directory if required
    os.makedirs(synthetic_data_dir, exist_ok=True)
    # add directory for low-resolution images if needed
    if add_low_res_images:
        low_res_directory = synthetic_data_dir + "_low_res"  # os.path.join(synthetic_data_dir, "low_res")
        os.makedirs(low_res_directory, exist_ok=True)

    # check if already generated:
    low_res_files = []
    if add_low_res_images:
        low_res_files = os.listdir(low_res_directory)

    files = os.listdir(synthetic_data_dir)
    if len(files) >= num_samples:
        logger.info(f"there are already {len(files)} synthetic data in folder:\n{synthetic_data_dir}")
        # files_full = [os.path.join(synthetic_data_dir, i_file) for i_file in files ]
        # files_full_path = [i_file for i_file in files_full if os.path.isfile(i_file)]
        files_full_path = [i_file for i_file in [os.path.join(synthetic_data_dir, i_file) for i_file in files] if os.path.isfile(i_file)]
        low_res_files_full_path = [i_file for i_file in [os.path.join(low_res_directory, i_file) for i_file in low_res_files] if os.path.isfile(i_file)]

        return files_full_path, low_res_files_full_path
    files = []  # set to default value for keeping values in the loop

    # load trained model from defined checkpoint:
    assert os.path.isfile(checkpoint_path), f"not existing checkpoint: {checkpoint_path}"
    model_class = hydra.utils.get_class(cfg.model.net._target_)

    # load model from checkpoint with given model class instance
    model = model_class.load_from_checkpoint(checkpoint_path, cfg=cfg)
    conditional_model = not (model.num_embeddings == 0)

    logger.info(f"Model loaded. Using model checkpoint stored at {checkpoint_path}. Performing processing on device {device}.")

    # set model to evaluation mode and send to appropriate device
    model.eval()
    model.to(device)

    # initialize
    gen_images_low_res = None

    # loop until number of samples is reached
    current_file_count = 0
    for current_batch in tqdm(range((num_samples // batch_size) + 1), desc="generate synthetic data"):
        # check if unfull last batch exists and set batch_size accordingly for last run
        if current_batch == num_samples // batch_size:
            if num_samples % batch_size == 0:
                continue
            else:
                batch_size = num_samples % batch_size

        # create noise matrix for batch
        z = torch.randn(batch_size, model.latent_dim, device=device)

        # run synthesis according to model class
        if conditional_model:
            # c = torch.randint(model.num_class, size=(batch_size, 1)).to(device)
            c = torch.cat(
                [torch.randint(c_dim, (batch_size, 1)) for c_dim in model.num_embeddings],
                dim=1,
            ).to(device)
            gen_images = _process_high_res((z, c), model)
            if add_low_res_images:
                gen_images_low_res = _process_low_res((z, c), upsample_factor, model)

        else:
            gen_images = _process_high_res(z, model)
            if add_low_res_images:
                gen_images_low_res = _process_low_res(z, upsample_factor, model)

        # pickle and save
        for batch_index in range(batch_size):
            sample = {"image": gen_images[batch_index]}
            if conditional_model:
                condit_values = [condit + "" + str(value.item()) for (condit, value) in zip(list(model.conditions), c[batch_index].flatten())]
                suffix = "_".join(condit_values)
            else:
                suffix = ""

            out_file = os.path.join(
                synthetic_data_dir,
                file_prefix + f"_{(current_file_count + batch_index):05}" + "_" + suffix + ".pickle",
            )
            pickle.dump(
                sample,
                open(out_file, "wb"),
            )
            files.append(out_file)

            if add_low_res_images:
                sample_lr = {"image": gen_images_low_res[batch_index]}
                out_file_low_res = os.path.join(
                    low_res_directory,
                    file_prefix + f"_{(current_file_count + batch_index):05}" + "_" + suffix + ".pickle",
                )
                pickle.dump(
                    sample_lr,
                    open(out_file_low_res, "wb"),
                )
                low_res_files.append(out_file_low_res)

        current_file_count += batch_size
    logger.success(f"Stored {num_samples} images in directory {synthetic_data_dir}.")
    return files, low_res_files


@logger.catch
def generate_synthetic_data_with_real_data_conditions(
    checkpoint_path: str,
    synthetic_data_dir: str,
    cfg: DictConfig,
    batch_size: int = 2,
    upsample_factor: int = 4,
    file_prefix: str = "",
    add_low_res_images: bool = False,
    update: bool = False,
    **kwargs,
):
    """generates synthetic data based on real data conditions.
    The function loads the pretrained generator model (saved in checkpoint_path)
    and generates the same amount (and conditions, for the conditional case) as the real dataset
    synthetic data are saved as pickle in the synthetic_data_dir

    Args:
        checkpoint_path (str): path to the checkpoint to use for the trained model
        synthetic_data_dir (str): directory where synthesized data are saved
        cfg (DictConfig): hydra config file
        batch_size (int, optional): Batch size used during processing. Defaults to 1.
        upsample_factor (int, optional): Upsampling used for models not capable of directly
            synthesizing high-resolution images. Defaults to 4.
        file_prefix (str, optional): Prefix to be added to the synthetic data filename. Defaults to ''.
        add_low_res_images (bool, optional): Some high-resolution models can additionally output
            low-resolution images. This can be used to check if the low-resolution output is turned into
            a dataset as well. Defaults to False.
        update:bool=False, set to true if you want to remove previous generated data and create new once

    Returns:
        files (list): list of all synthetic data files (saves as pickle)
        low_res_files (list): list of all low resoltion synthetic data files (saves as pickle)
    """
    # check if GPU instance is being used and assign correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create directory if required
    os.makedirs(synthetic_data_dir, exist_ok=True)
    # add directory for low-resolution images if needed
    if add_low_res_images:
        low_res_directory = synthetic_data_dir + "_low_res"  # os.path.join(synthetic_data_dir, "low_res")
        os.makedirs(low_res_directory, exist_ok=True)

    # check if already generated:
    low_res_files = []
    if add_low_res_images:
        low_res_files = os.listdir(low_res_directory)

    # this function generates the same number of real data:
    conditional_model = len(cfg.model.conditions) > 0
    if conditional_model:
        dataset_object = data.ConditionalDataset
        real_train_dataset = dataset_object(
            dataset_dir=cfg.data.paths.training_data,
            clinical_data_dir=cfg.data.paths.clinical_data,
            clinical_data_processing=cfg.data.clinical_data_processing,
            conditions_list=cfg.model.conditions,
        )
    else:
        real_train_dataset = data.BaseImageDataset(dataset_dir=cfg.data.paths.training_data)

    num_samples = len(real_train_dataset)

    files = os.listdir(synthetic_data_dir)
    if len(files) >= num_samples:
        logger.info(f"there are already {len(files)} synthetic data in folder:\n{synthetic_data_dir}")
        # files_full = [os.path.join(synthetic_data_dir, i_file) for i_file in files ]
        # files_full_path = [i_file for i_file in files_full if os.path.isfile(i_file)]
        files_full_path = [i_file for i_file in [os.path.join(synthetic_data_dir, i_file) for i_file in files] if os.path.isfile(i_file)]
        low_res_files_full_path = [i_file for i_file in [os.path.join(low_res_directory, i_file) for i_file in low_res_files] if os.path.isfile(i_file)]

        return files_full_path, low_res_files_full_path

    files = []  # set to default value for keeping values in the loop

    # load trained model from defined checkpoint:
    assert os.path.isfile(checkpoint_path), f"not existing checkpoint: {checkpoint_path}"
    model_class = hydra.utils.get_class(cfg.model.net._target_)

    # load model from checkpoint with given model class instance
    model = model_class.load_from_checkpoint(checkpoint_path, cfg=cfg)
    logger.info(f"Model loaded. Using model checkpoint stored at {checkpoint_path}. Performing processing on device {device}.")
    # set model to evaluation mode and send to appropriate device
    model.eval()
    model.to(device)

    # loop until number of samples is reached
    current_file_count = 0
    train_loader = torch.utils.data.DataLoader(dataset=real_train_dataset, batch_size=batch_size, shuffle=False)

    gen_images_low_res = None
    # # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = next(dataiter) # dataiter.next()
    for current_batch in tqdm(train_loader, desc="generate synthetic data based on real data size/conditions"):
        # create noise matrix for batch
        this_batch_size = current_batch[0].shape[0]
        z = torch.randn(this_batch_size, model.latent_dim, device=device)

        # run synthesis according to model class
        if conditional_model:  # same conditions as real images
            # c = torch.randint(model.num_class, size=(batch_size, 1)).to(device)
            # c = torch.cat([torch.randint(c_dim, (batch_size, 1)) for c_dim in model.num_class], dim=1).to(device)
            c = current_batch[1].to(device)
            gen_images = _process_high_res((z, c), model)
            if add_low_res_images:
                gen_images_low_res = _process_low_res((z, c), upsample_factor, model)
        else:
            gen_images = _process_high_res(z, model)
            if add_low_res_images:
                gen_images_low_res = _process_low_res(z, upsample_factor, model)

        # pickle and save
        for batch_index in range(this_batch_size):
            sample = {"image": gen_images[batch_index]}
            if conditional_model:
                condit_values = [condit + "" + str(value.item()) for (condit, value) in zip(list(model.conditions), c[batch_index].flatten())]
                suffix = "_".join(condit_values)
            else:
                suffix = ""
            # suffix = f'_BMI_CAT_{c[batch_index].flatten().item()}' if conditional_model else ''
            out_file = os.path.join(
                synthetic_data_dir,
                file_prefix + f"_{(current_file_count + batch_index):05}" + "_" + suffix + ".pickle",
            )
            pickle.dump(
                sample,
                open(out_file, "wb"),
            )
            files.append(out_file)

            if add_low_res_images:
                sample_lr = {"image": gen_images_low_res[batch_index] if gen_images_low_res else None}
                out_file_low_res = os.path.join(
                    low_res_directory,
                    file_prefix + f"_{(current_file_count + batch_index):05}" + "_" + suffix + ".pickle",
                )
                pickle.dump(
                    sample_lr,
                    open(out_file_low_res, "wb"),
                )
                low_res_files.append(out_file_low_res)

        current_file_count += this_batch_size
    # get the number of data generated:
    files_generated = os.listdir(synthetic_data_dir)
    logger.success(
        f"Generated {len(files_generated)} synthetic data in directory {synthetic_data_dir}\nOriginal dataset included {len(real_train_dataset)} samples)"
    )
    return files, low_res_files


@logger.catch
def plot_synthetic_data(path_latest_checkpoint: str, figure_path: str, cfg, add_low_res_images: bool = False):
    """Generates and plots synthetic data using a specified checkpoint.

    Args:
        path_latest_checkpoint (str): Full path and filename to the model checkpoint to use.
        figure_path (str): Path to the directory where the plot will be saved.
        cfg (_type_): Hydra configuration object.
        add_low_res_images (bool, optional): Flag indicating whether to also save low-resolution images. Defaults to False.
    """

    assert os.path.isfile(path_latest_checkpoint), f"file {path_latest_checkpoint} does not exist!"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        temp = re.match(r"epoch\=(\d+)-step=(\d+).ckpt", os.path.basename(path_latest_checkpoint))
        model_epoch = temp.group(1)
    except:
        model_epoch = "None"

    # check if already done:
    final_figure_file = os.path.join(figure_path, "thumbnails_epoch" + str(model_epoch) + "_complete.png")
    function_outputs = [final_figure_file]
    if add_low_res_images:
        final_figure_file_low_res = os.path.join(
            figure_path,
            "thumbnails_epoch" + str(model_epoch) + "_complete_low_resolution.png",
        )
        function_outputs.append(final_figure_file_low_res)

    if all([os.path.isfile(this) for this in function_outputs]):
        sms_str = "\n".join(function_outputs)
        logger.info(f"already done, saved figure:\n{sms_str}")
        return

    # load model, load trained weight and set to eval mode
    model_class = hydra.utils.get_class(cfg.model.net._target_)
    logger.info(model_class)
    model = model_class.load_from_checkpoint(path_latest_checkpoint, cfg=cfg)
    _ = model.eval()
    _ = model.to(device)

    if not hasattr(model, "lattent_dim") and hasattr(model, "image_model"):  # for mmcgan
        model.latent_dim = model.image_model.latent_dim

    # check if conditional case:
    num_examples = cfg.evaluate.num_samples_to_plot
    if model.num_embeddings == 0:
        res = model.sample(num_samples=num_examples, batch_size=2, return_samples=True)
    else:
        list_range = [np.arange(0, this_val) for this_val in model.conditions_dict.values()]
        combinations = [list(p) for p in itertools.product(*list_range)]
        num_examples = len(combinations)
        res = model.sample(
            num_samples=num_examples,
            batch_size=2,
            return_samples=True,
            conditions=combinations,
        )

    if isinstance(res, tuple):
        images, table = res
    else:
        images = res

    # this_fig_name = 'thumbnails_epoch'+ str(model_epoch) +'_prev.png'
    # save_image_grid(images.clip(0,1), n_per_row=num_examples, save_dir=figure_path, file_name=this_fig_name,
    #                 add_transposed=True, flip_images=True, cmap='viridis', figsize=(48,3))

    # some tests here for better brain orientation:
    # input = images[:,0,:,:,:].clone()
    # print(input.shape)
    # rotated_images = torch.rot90(input, k=1, dims=[1, 3]) # alteranively use torch.rotate: https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.rotate.html
    # print(rotated_images.shape)
    # # img1 = input[0,:,:,:].numpy()
    # # img2 = rotated_images[0,:,:,:].numpy()
    # rotated_images_batch = images.clone()
    # rotated_images_batch[:,0,:,:,:] = rotated_images
    # this_fig_name = 'thumbnails_epoch'+ str(model_epoch) +'_rotated.png'
    # save_image_grid_2(rotated_images_batch.clip(0,1), n_per_row=num_examples, save_dir=figure_path, file_name=this_fig_name,
    #                 cmap='viridis', figsize=(48,3))

    # the following is related to the orientation of my dataset (might be different for other starting orientations)
    input = images[:, 0, :, :, :].clone()
    # print(input.shape)
    rotated_images = torch.rot90(
        input, k=1, dims=[1, 3]
    )  # alteranively use torch.rotate: https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.rotate.html
    rotated_images = torch.rot90(rotated_images, k=1, dims=[2, 3])
    rotated_images = torch.rot90(rotated_images, k=1, dims=[2, 3])
    # logger.info(rotated_images.shape)
    # img1 = input[0,:,:,:].numpy()
    # img2 = rotated_images[0,:,:,:].numpy()
    rotated_images_batch = images.clone()
    rotated_images_batch[:, 0, :, :, :] = rotated_images
    this_fig_name = "thumbnails_epoch" + str(model_epoch) + ".png"
    save_image_grid(
        rotated_images_batch.clip(0, 1),
        n_per_row=num_examples,
        save_dir=figure_path,
        file_name=this_fig_name,
        cmap="viridis",
        figsize=(48, 3),
    )

    # combine figures in a unique one
    try:
        png_files = glob.glob(figure_path + "/*" + this_fig_name)
        if len(png_files) == 3:  # combine the 3 main view (transverse, sag)
            plot_utils.combine_png(png_list=png_files, output_file=final_figure_file, mode="vertical")
    except:
        pass

    logger.info(f"saved figure:\n{final_figure_file}")


@logger.catch
def perform_dim_reduction(train_embeddings_files: List[str], activation_files: List[str], methods: List[str]) -> list:
    """Performs dimensionality reduction on image embeddings by means of the chosen methods and saves the result as numpy arrays
    back to the embedding directory.

    Args:
        train_data_path (str): Training data required when using UMAP, for other methods this can be None.
        roots (List[str] or str): Directories containing image embeddings, named "activations.npy"
        methods (List[str]): List of desired methods for dim reduction. Currently 'umap', 'tsne' and 'mds' are supported.

    Returns:
        list: list of saved files
    """
    if not isinstance(methods, list):
        methods = [methods]
    outputfile_list = []

    for this_activation_file in activation_files:
        dataset_path, _ = os.path.split(this_activation_file)
        for method in methods:
            # check if already computed
            output_file = os.path.join(dataset_path, f"{method}_coords.npy")
            if not os.path.isfile(output_file):
                transformed_embeddings = dim_reduction(
                    method=method,
                    train_embeddings_files=train_embeddings_files,
                    activation_file=this_activation_file,
                )
                np.save(
                    output_file,
                    transformed_embeddings,
                )

                logger.info(f"Saved {transformed_embeddings.shape[0]} datapoints to {os.path.join(dataset_path, f'{method}_coords.npy')}")

            outputfile_list.append(output_file)
    return outputfile_list
