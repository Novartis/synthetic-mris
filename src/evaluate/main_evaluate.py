# pylint: disable=C0302
"""evaluation script for synthesis models"""

import glob
import math
import os
import shutil
import sys

import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from tqdm import tqdm

# load local functions
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(SRC_PATH)

from data import BaseImageDataset, ConditionalDataset, make_train_validation_split
from helpers import evaluate_utils
from helpers.clustering.plot_clusters import plot_clusters
from helpers.fid_score_medicalnet_3d import compute_fid, get_activations
from helpers.precision_recall_metric import compute_precision_recall
from model_architecture import convert_medicalnet

# Constants
BATCH_SIZE = 4
NUM_WORKERS = 4


def get_condition_info(
    file_list: list,
    condit_dict: dict,
    clinical_table: pd.DataFrame,
    method="clinical_table",
    coordinates: list = None,
    identifier: str = "",
) -> dict:
    """returns conditions info based on the method

    Args:
        file_list (list): list of files you want to get the condition info
        condit_dict (dict): conditions dict
        clinical_table (pd.DataFrame): dataframe with all info including the conditions
        method (str, optional): method used to get the conditions, implemented solutions are:
            "clinical_table": gets info from pandas dataframe
            "filename":  gets info from filename.
            Defaults to "clinical_table".
        coordinates (list, optional): _description_. Defaults to None.
        identifier (str, optional): column name to use as identifier. Defaults to "".

    Returns:
        dict: dictionary including following key-values:
            out_dict = {
                "file": file_list,
                "X1": coordinates[:, 0],
                "X2": coordinates[:, 1],
                "all_conditions": condition_str,
            }
    """
    each_condition, condition_str = [], []

    if method == "clinical_table":
        # reads info from clinical table
        if identifier == "":
            identifier = "sample"
        if identifier not in clinical_table.columns:
            raise ValueError("Invalid clinical data identifier: {identifier} not present in dataframe:{clinical_table.columns}")

        clinical_table.index = clinical_table[identifier]
        for i_file in file_list:
            _, filename_only = os.path.split(i_file)
            temp = filename_only.split("_")
            sample_str = temp[0]
            try:
                sample_clinical_info = clinical_table.loc[sample_str]
                each_condition.append([str(sample_clinical_info[i_key]) for i_key in condit_dict])
                condition_str.append("_".join([str(i_key) + str(sample_clinical_info[i_key]) for i_key in condit_dict]))
            except KeyError:
                condition_str.append("")
                each_condition.append([""] * len(condit_dict))

    elif method == "filename":
        # reads info from filename (is used for synthetic data)
        for i_file in file_list:
            _, filename = os.path.split(i_file)
            filename = filename.split(".")[0]

            this_file_conditions = []
            for this_condit in condit_dict:
                temp = filename.split(this_condit)
                this_cond_val = temp[1].split("_")
                this_file_conditions.append(this_cond_val[0])

            each_condition.append(this_file_conditions)
            condition_str.append("_".join([str(i_key) + str(i_val) for (i_key, i_val) in zip(condit_dict, this_file_conditions)]))

    out_dict = {
        "file": file_list,
        "X1": coordinates[:, 0],
        "X2": coordinates[:, 1],
        "all_conditions": condition_str,
    }
    for idx, i_condit in enumerate(condit_dict):
        out_dict[i_condit] = [this[idx] for this in each_condition]

    return out_dict


def medicalnet_activations(
    dataset_path: str,
    pre_trained_model: str,
    output_path: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    num_samples: int = None,
):
    """Extracts latent representation of images using medicalnet.

    Args:
        dataset_path (str): Path to dataset of images to be processed.
        pre_trained_model (str): Path to the trained weights of medicalnet.
        batch_size (int, optional): Batch size used for processing. Defaults to 4.
        num_workers (int, optional): Number of workers employed for
            the process. Defaults to 4.
        num_samples (int, optional): Number of samples to randomly select
            from the dataset. If None, full dataset is processed. Defaults to None.
    """
    # if activations.npy already computed returns the full path to it:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_data = os.path.join(output_path, "activations.npy")
    output_data_csv = os.path.join(output_path, "files_in_activations.csv")

    if os.path.exists(output_data) and os.path.exists(output_data_csv):
        logger.info(f"already computed activations: {output_data}")
        return

    os.makedirs(output_path, exist_ok=True)
    # create dataset object
    dataset = BaseImageDataset(dataset_path, image_crop=(256, 256, 256), downsample=(1, 1, 1))
    file_path_list = dataset.file_paths

    if num_samples is not None:
        if num_samples < len(dataset):
            dataset, _ = make_train_validation_split(dataset, num_samples)
            file_path_list = [file_path_list[index] for index in dataset.indices]
        else:
            logger.info(f"Full dataset only contains {len(dataset)} samples, cannot subset the requested {num_samples} samples.")

    # store file paths used to generate activations
    file_paths = pd.DataFrame(file_path_list)
    file_paths.to_csv(output_data_csv, index=False)

    model = convert_medicalnet.get_medicalnet_backbone(256, pre_trained_model).to(device)

    # extract activations using medicalnet3d
    act = get_activations(model, dataset, batch_size=batch_size, num_workers=num_workers, device=device)

    # save results as numpy array
    np.save(output_data, act)

    logger.info(f"Saved activations of shape {act.shape} to {os.path.join(dataset_path, 'activations.npy')}")


def get_upsampled_low_res_dir(cfg: DictConfig) -> str:
    """
    Constructs the directory path for upsampled low resolution real data.

    Args:
        cfg (DictConfig): Configuration object containing the evaluation settings.

    Returns:
        str: The directory path for upsampled low resolution real data.
    """
    # defines the upsampled low resolution real data directory
    # to be used in different places
    upsampled_low_res_dir = os.path.join(
        cfg.evaluate.main_evaluate_folder,
        "original_data_embeddings",
        "upsampled_low_res",
    )
    return upsampled_low_res_dir


def step1_generate_high_res_from_low_res(cfg: DictConfig) -> list:
    """upsample low resolution real data

    Args:
        cfg (DictConfig): configuration

    Returns:
        list: list of files upsampled
    """
    # get input info
    num_samples = cfg.evaluate.synthetic_dataset_samples

    upsampled_low_res_dir = get_upsampled_low_res_dir(cfg)
    # check if already computed: inside the function

    upsampled_low_res_files = evaluate_utils.generate_high_res_from_low_res_data(output_dir=upsampled_low_res_dir, num_samples=num_samples, cfg=cfg)
    # the conditional case same as implemented here (conditions don't affect this step)

    return upsampled_low_res_files


def step1_generate_synthetic_data(cfg: DictConfig, synthetic_data_dir: str, add_low_res_images: bool = False):
    """Generates synthetic data and plots some example for visual inspection"""

    # get input info
    num_samples = cfg.evaluate.synthetic_dataset_samples

    this_checkpoint_path = evaluate_utils.get_checkpoint_file(main_path=cfg.evaluate.checkpoint_path, epoch_num=cfg.evaluate.epoch_num)

    # generate synthetic data:
    if len(cfg.model.conditions) > 0:
        (
            files,
            low_res_files,
        ) = evaluate_utils.generate_synthetic_data_with_real_data_conditions(
            checkpoint_path=this_checkpoint_path,
            synthetic_data_dir=synthetic_data_dir,
            cfg=cfg,
            num_samples=num_samples,
            add_low_res_images=add_low_res_images,
        )

    else:
        files, low_res_files = evaluate_utils.generate_synthetic_data(
            checkpoint_path=this_checkpoint_path,
            synthetic_data_dir=synthetic_data_dir,
            cfg=cfg,
            num_samples=num_samples,
            add_low_res_images=add_low_res_images,
        )
    return files, low_res_files


def compute_embeddings(
    embeddings_dirs: list,
    dataset_paths: list,
    pre_trained_medicalnet_model: str,
    batch_size: int = 1,
) -> list:
    """computes embeddings from data specified in dataset_paths
    the computed embeddings are saved in embeddings_dirs as "activations.npy"

    NOTE: the order of embeddings_dirs and dataset_paths should be the same
    (ie first real and then synthetic paths)

    Args:
        embeddings_dirs (list): paths to the real and synthetic embeddings directory
        dataset_paths (list):paths to the real and synthetic data directory
        pre_trained_medicalnet_model (str): path to the pre_trained medicalnet model
        batch_size (int): size of the batch (recommended value is 1)

    Returns:
        list: saved embeddings files
    """
    embeddings_file, created_bool = [], []
    for this_embedding_dir in embeddings_dirs:
        this_file = os.path.join(this_embedding_dir, "activations.npy")
        embeddings_file.append(this_file)
        created_bool.append(os.path.isfile(this_file))

    if not all(created_bool):
        # check required info for this step
        assert os.path.isfile(pre_trained_medicalnet_model), f"file {pre_trained_medicalnet_model} does not exist"

        # compute embeddings for real and fake data
        for i_data, out_dir in zip(dataset_paths, embeddings_dirs):
            medicalnet_activations(
                dataset_path=i_data,
                output_path=out_dir,
                pre_trained_model=pre_trained_medicalnet_model,
                batch_size=batch_size,
                num_workers=4,
                num_samples=None,
            )

    logger.info(f"embeddings saved as:\n{embeddings_file}")
    return embeddings_file


def step2_compute_embeddings_and_fid(cfg: DictConfig, add_low_resolution: bool = False) -> dict:
    """Computes embeddings using medicalnet (saved as activations) and FID"""

    # get embeddings for full resolution data:

    # specify input and ouptut for the embedding fcn
    real_data_path = cfg.data.paths.training_data
    synthetic_data_path = cfg.evaluate.synthetic_data_dir
    dataset_paths = [real_data_path, synthetic_data_path]
    embeddings_dirs = [
        cfg.evaluate.origin_data_embeddings,
        cfg.evaluate.synthetic_embeddings_dir,
    ]
    embeddings_file = compute_embeddings(
        embeddings_dirs=embeddings_dirs,
        dataset_paths=dataset_paths,
        pre_trained_medicalnet_model=cfg.evaluate.medicalnet_model_weights,
        batch_size=cfg.train.optimizers.batch_size,
    )

    # compute fid
    fid = compute_fid(
        embeddings_file_dataset1=embeddings_file[0],
        embeddings_file_dataset2=embeddings_file[1],
    )
    logger.info(f"FID for {cfg.evaluate.synthetic_data_dir}: {fid}")

    output_dict = {"embeddings_file": embeddings_file, "fid": fid}

    # same for low resolution
    if add_low_resolution:
        # get embeddings for low resolution data:
        # specify input and ouptut for the embedding fcn
        real_data_low_res_path = get_upsampled_low_res_dir(cfg)
        synthetic_data_low_res_path = cfg.evaluate.synthetic_data_dir + "_low_res"
        dataset_paths = [real_data_low_res_path, synthetic_data_low_res_path]
        embeddings_dirs = [
            os.path.join(cfg.evaluate.origin_data_embeddings, "low_res"),
            os.path.join(cfg.evaluate.synthetic_embeddings_dir, "low_res"),
        ]
        embeddings_file_low_res = compute_embeddings(
            embeddings_dirs=embeddings_dirs,
            dataset_paths=dataset_paths,
            pre_trained_medicalnet_model=cfg.evaluate.medicalnet_model_weights,
            batch_size=cfg.train.optimizers.batch_size,
        )

        # compute fid for low resolution data:
        fid_low_res = compute_fid(
            embeddings_file_dataset1=embeddings_file_low_res[0],
            embeddings_file_dataset2=embeddings_file_low_res[1],
        )
        logger.info(f"FID LOW RESOLUTION for {cfg.evaluate.synthetic_data_dir}: {fid_low_res}")

        output_dict["embeddings_file_low_res"] = embeddings_file_low_res
        output_dict["fid_low_res"] = fid_low_res

    return output_dict


def step2_compute_precision_recall(cfg: DictConfig, add_low_resolution: bool = False) -> dict:
    """computes precision and recall

    Args:
        cfg (DictConfig): config file

    Returns:
        floats: precision and recall
    """
    real_data_embeddings_file = os.path.join(cfg.evaluate.origin_data_embeddings, "activations.npy")
    syn_data_embeddings_file = os.path.join(cfg.evaluate.synthetic_embeddings_dir, "activations.npy")

    ipr = compute_precision_recall(
        real_data_embeddings_file=real_data_embeddings_file,
        syn_data_embeddings_file=syn_data_embeddings_file,
    )
    # logs and outputs
    logger.info(f"Precision for {cfg.evaluate.synthetic_data_dir}: {ipr['precision']}")
    logger.info(f"Recall for {cfg.evaluate.synthetic_data_dir}: {ipr['recall']}")
    out_dict = {"precision": ipr["precision"], "recall": ipr["recall"]}

    # same for low resolution
    if add_low_resolution:
        real_data_embeddings_file_low_res = os.path.join(cfg.evaluate.origin_data_embeddings, "low_res", "activations.npy")
        syn_data_embeddings_file_low_res = os.path.join(cfg.evaluate.synthetic_embeddings_dir, "low_res", "activations.npy")

        ipr_low_res = compute_precision_recall(
            real_data_embeddings_file=real_data_embeddings_file_low_res,
            syn_data_embeddings_file=syn_data_embeddings_file_low_res,
        )
        # logs and outputs
        logger.info(f"Precision (low resolution){cfg.evaluate.synthetic_data_dir}: {ipr_low_res['precision']}")
        logger.info(f"Recall (low resolution) {cfg.evaluate.synthetic_data_dir}: {ipr_low_res['recall']}")
        out_dict["precision_low_res"] = ipr_low_res["precision"]
        out_dict["recall_low_res"] = ipr_low_res["recall"]

    return out_dict


def step3_dim_red_umap(cfg: DictConfig, add_low_resolution: bool = False):
    """
    Perform dimensionality reduction using UMAP and plot the results.

    Args:
        cfg (DictConfig): hydra configuration object containing paths and settings for evaluation.
        add_low_resolution (bool, optional): If True, also perform dimensionality
            reduction on low-resolution data. Defaults to False.

    Raises:
        AssertionError: If any of the specified embedding files do not exist.

    """
    # compute umap/method coordinates

    train_embeddings_files = [os.path.join(cfg.evaluate.origin_data_embeddings, "activations.npy")]
    embeddings_file = [
        os.path.join(cfg.evaluate.origin_data_embeddings, "activations.npy"),
        os.path.join(cfg.evaluate.synthetic_embeddings_dir, "activations.npy"),
    ]
    # methods_list = list(cfg.evaluate.dim_reduction_method)
    # method = cfg.evaluate.dim_reduction_method
    assert all([os.path.isfile(this) for this in embeddings_file])
    coordinate_files = evaluate_utils.perform_dim_reduction(train_embeddings_files, embeddings_file, cfg.evaluate.dim_reduction_method)

    # save umap plot:
    dataset_dirs = coordinate_files

    save_path = cfg.evaluate.figure_path
    os.makedirs(save_path, exist_ok=True)
    methods_list = [cfg.evaluate.dim_reduction_method]
    for method in methods_list:
        plot_clusters(
            dataset_dirs,
            method,
            save_path,
            marker_size=4,
            labels=["real_data", "synthetic_data"],
        )  # for better visualization use the notebook

    if add_low_resolution:
        train_embeddings_files_low_res = [os.path.join(cfg.evaluate.origin_data_embeddings, "low_res", "activations.npy")]
        embeddings_file_low_res = [
            os.path.join(cfg.evaluate.origin_data_embeddings, "low_res", "activations.npy"),
            os.path.join(cfg.evaluate.synthetic_embeddings_dir, "low_res", "activations.npy"),
        ]
        # methods_list = list(cfg.evaluate.dim_reduction_method)
        # method = cfg.evaluate.dim_reduction_method
        assert all([os.path.isfile(this) for this in embeddings_file_low_res])
        coordinate_files_low_res = evaluate_utils.perform_dim_reduction(
            train_embeddings_files_low_res,
            embeddings_file_low_res,
            cfg.evaluate.dim_reduction_method,
        )

        # save umap plot:
        dataset_dirs = coordinate_files_low_res
        save_path = os.path.join(cfg.evaluate.figure_path, "low_res")
        os.makedirs(save_path, exist_ok=True)
        methods_list = [cfg.evaluate.dim_reduction_method]
        for method in methods_list:
            plot_clusters(
                dataset_dirs,
                method,
                save_path,
                marker_size=4,
                labels=["real_data", "synthetic_data"],
            )


def get_condition_df(loc_dict: dict, method: str, condit_dict: list, clinical, identifier=""):
    """returns a dataframe with umap coordinates and corresponding filenames, and conditions

    Args:
        loc_dict (dict): inlcuding keys: 'path', 'label'
        method (str): method name
        condit_dict (list): list of string, including the conditions
        clinical (pd.DataFrame): clinical dataframe

    Returns:
        pandas dataframe: dataframe with umap coordinates and
            corresponding filenames, and conditions
    """
    loc = loc_dict["path"]
    this_label = loc_dict["label"]

    condit_temp = "_".join(condit_dict)
    files_in_activations = os.path.join(loc, "files_in_activations_with_" + condit_temp + ".csv")
    if os.path.isfile(files_in_activations):
        # already computed conditions
        output_df = pd.read_csv(files_in_activations)

    elif os.path.isfile(os.path.join(loc, "files_in_activations.csv")):
        coord_file = os.path.join(loc, method + "_coords.npy")
        assert os.path.isfile(coord_file), "missing input file: '{coord_file}' does not exits"

        coords = np.load(coord_file)

        this_file = os.path.join(loc, "files_in_activations.csv")
        file_series = pd.read_csv(this_file)
        # compute conditions
        file_list = list(file_series["0"])
        if this_label == "real":
            condit_method = "clinical_table"
            clinical_table = clinical.copy(deep=True)
        elif this_label == "synthetic":
            condit_method = "filename"
            clinical_table = ""
        else:
            raise ValueError(f"Invalid label: {this_label}")

        extracted_condit_dict = get_condition_info(
            file_list,
            condit_dict,
            method=condit_method,
            clinical_table=clinical_table,
            coordinates=coords,
            identifier=identifier,
        )
        # check dimension
        dims_dict = [len(extracted_condit_dict[i_key]) for i_key in list(extracted_condit_dict.keys())]
        assert all(this == dims_dict[0] for this in dims_dict)

        # save the conditions for next time:
        output_df = pd.DataFrame(extracted_condit_dict)

        # remove missing condition rows only for real cases:
        if this_label == "real":
            output_df.to_csv(files_in_activations, index=False)
            file_series = pd.read_csv(files_in_activations)  # make sure that empty data are always the same
            missing_condit = pd.isnull(file_series["all_conditions"])  # | file_series['condit_str']=="nan"
            coords = coords[~missing_condit]
            file_series_complete_condit = file_series[~missing_condit].copy(deep=True).reset_index()
            # overwrite:
            output_df = file_series_complete_condit

        # file_series_grp = file_series_complete_condit.groupby("all_conditions")
        # condit_conv_dict = {item:idx for idx, item in enumerate(file_series_grp.groups.keys()) }
        # condit_number = [condit_conv_dict[i_condit] for i_condit
        #       in list(file_series_complete_condit["all_conditions"])]
        # file_series["all_conditions_grp_number"] = condit_number
        output_df.to_csv(files_in_activations, index=False)

    else:
        raise FileNotFoundError(f"No files_in_activations file found in {loc}")

    return output_df


def step4_condition_umap(cfg: DictConfig):
    """plots umap with conditions in different colours

    Args:
        cfg (DictConfig): configuration object

    """

    # get info from cfg:
    real_data_path = cfg.evaluate.origin_data_embeddings  # cfg.data.paths.training_data
    synthetic_data_path = cfg.evaluate.synthetic_embeddings_dir  # cfg.evaluate.synthetic_data_dir  # needs to be a list
    clinical_data_dir = cfg.data.paths.clinical_data

    save_path = cfg.evaluate.figure_path
    os.makedirs(save_path, exist_ok=True)

    method = cfg.evaluate.dim_reduction_method

    cluster_locs = [
        {"path": f"{real_data_path}", "name_offset": 45, "label": "real"},
        {"path": f"{synthetic_data_path}", "name_offset": 45, "label": "synthetic"},
    ]

    # or replicate here step by step:
    if clinical_data_dir.endswith(".parquet") or clinical_data_dir.endswith(".pq"):
        clinical = pd.read_parquet(clinical_data_dir)
    elif clinical_data_dir.endswith(".csv") or clinical_data_dir.endswith(".csv.gz"):
        clinical = pd.read_csv(clinical_data_dir)
    else:
        raise ValueError("Unknown file type for clinical data -- this must be either .csv or Parquet.")

    conditional_dataset = ConditionalDataset(
        dataset_dir=cfg.data.paths.training_data,
        clinical_data_dir=cfg.data.paths.clinical_data,
        clinical_data_processing=cfg.data.clinical_data_processing,
        conditions_list=cfg.model.conditions,
    )

    clinical, conditions_dict = (
        conditional_dataset.clinical,
        conditional_dataset.conditions_dict,
    )

    condit_dict = list(conditions_dict.keys())
    condit_temp = "_".join(condit_dict)

    # get coords and conditional info
    data_condit_df = {}
    identifier = cfg.data.clinical_data_processing.identifier
    for _, loc_dict in enumerate(cluster_locs):
        this_label = loc_dict["label"]
        complete_condit_df = get_condition_df(loc_dict, method, condit_dict, clinical, identifier)
        # add this label to the df
        complete_condit_df["label"] = [this_label] * complete_condit_df.shape[0]
        data_condit_df[this_label] = complete_condit_df

        # save a plot to understand umap for each cluster:
        for this_condit in condit_dict:
            _ = plt.figure(figsize=(8, 10))
            _ = plt.gca()
            sns.scatterplot(data=complete_condit_df, x="X1", y="X2", hue=this_condit)
            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                title=this_condit,
            )
            fig_filename = os.path.join(save_path, f"{method}_{this_label}_scatter_{this_condit}.png")
            plt.savefig(fig_filename, dpi=300, bbox_inches="tight")  # , bbox_inches='tight' to keep the full legend when saving image
            plt.clf()

        _ = plt.figure(figsize=(8, 10))
        _ = plt.gca()
        sns.scatterplot(data=complete_condit_df, x="X1", y="X2", hue="all_conditions")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        fig_filename = os.path.join(save_path, f"{method}_{this_label}_scatter_all_conditions.png")
        plt.savefig(fig_filename, dpi=300, bbox_inches="tight")  # , bbox_inches='tight' to keep the full legend when saving image
        plt.clf()

    overall_df = pd.concat(
        [
            data_condit_df[cluster_locs[0]["label"]],
            data_condit_df[cluster_locs[1]["label"]],
        ],
        ignore_index=True,
        axis=0,
    )

    # plot and save
    _fig = plt.figure(figsize=(8, 10))
    _ax = plt.gca()
    sns.scatterplot(data=overall_df, x="X1", y="X2", hue="label", style="all_conditions")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig_filename = os.path.join(save_path, f"{method}_{condit_temp}_scatter_1.png")
    plt.savefig(fig_filename, dpi=300, bbox_inches="tight")  # , bbox_inches='tight' to keep the full legend when saving image

    _fig = plt.figure(figsize=(8, 10))
    _ax = plt.gca()
    sns.scatterplot(data=overall_df, x="X1", y="X2", hue="all_conditions", style="label")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig_filename = os.path.join(save_path, f"{method}_{condit_temp}_scatter_2.png")
    plt.savefig(fig_filename, dpi=300, bbox_inches="tight")

    logger.info(f"saved scatter plot it:\n{fig_filename}")


def make_gif(png_files: list, gif_filename: str, duration: int = 200) -> None:
    """combine png_files in a gif

    Args:
        png_files (list): list of png files to combine in a gif
        gif_filename (str): filename of the gif to save
        duration (int, optional): duration of the gif. Defaults to 200.
    """

    if len(png_files) <= 1:
        logger.warning(f"Not correct input files (len(png_files)={len(png_files)}): no gif saved! check png_files values")
        return

    try:
        frames = [Image.open(image) for image in png_files]
    except:  # pylint: disable=[bare-except]
        frames = []
        for i_file in png_files:
            this = i_file[0]
            if os.path.exists(this):
                img = Image.open(str(this))
                frames.append(img)

    frame_one = frames[0]
    frame_one.save(
        gif_filename,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
    )
    logger.info(f"combining {len(png_files)} plots over epochs, saved in {gif_filename}")
    return


@logger.catch
def main_evaluate(cfg: DictConfig, epoch_num: int, add_low_resolution: bool = False) -> dict:
    """main evaluation function for gans

    the main evaluation function includes the following steps:
    - step 1: generate synthetic data:
    - step 2: compute embeddings (using mednet)
    - step 3: compute umap/method coordinates and make a simple umap plot (real vs synthetic)
    - step 4: (optional) additional plot for the conditional case
    - step 5: delete synthetic data to save disc space

    Args:
        cfg (DictConfig):config files with all info/variable
        epoch_num (int): epoch number to load for the evaluation
        add_low_resolution (bool, optional): set to True to
            add low resolution evaluation. Defaults to False.

    Returns:
        dict: including evaluation metrics and corresponding values
    """

    # define epoch specific output folders
    with open_dict(cfg):
        cfg.evaluate.epoch_num = int(epoch_num)
        cfg.evaluate.output_folder = os.path.join(cfg.evaluate.main_evaluate_folder, "epoch_" + str(epoch_num))
        # outputs for synthetic data:
        cfg.evaluate.figure_path = os.path.join(cfg.evaluate.output_folder, "figures")
        cfg.evaluate.synthetic_data_dir = os.path.join(cfg.evaluate.output_folder, "generated_data")
        # outputs for embeddings:
        cfg.evaluate.synthetic_embeddings_dir = os.path.join(cfg.evaluate.output_folder, "embeddings")
        cfg.evaluate.origin_data_embeddings = os.path.join(
            os.path.dirname(cfg.evaluate.output_folder), "original_data_embeddings"
        )  # note original data embeddings are the same for all epochs, because computed on original data
    logger.info(f"evaluate at epoch {epoch_num} ------------- ")

    # since synthetic data are removed after embedding, check if the embedding was already computed
    # and skip first steps
    if not os.path.isfile(os.path.join(cfg.evaluate.synthetic_embeddings_dir, "activations.npy")):
        # step 1: generate synthetic data:
        step1_generate_synthetic_data(
            cfg,
            synthetic_data_dir=cfg.evaluate.synthetic_data_dir,
            add_low_res_images=add_low_resolution,
        )

        # generate high resolution from low resoltuion real image for low resolution evaluation
        if add_low_resolution:
            _ = step1_generate_high_res_from_low_res(cfg)

        # plots for visual inspection:
        figure_path = cfg.evaluate.figure_path
        this_checkpoint_path = evaluate_utils.get_checkpoint_file(main_path=cfg.evaluate.checkpoint_path, epoch_num=cfg.evaluate.epoch_num)
        evaluate_utils.plot_synthetic_data(
            this_checkpoint_path,
            figure_path,
            cfg,
            add_low_res_images=add_low_resolution,
        )
        logger.info(f"synthetic data example in :\n{figure_path}")

    # step 2: compute embeddings (using mednet) and fid
    embedding_fid_dict = step2_compute_embeddings_and_fid(cfg, add_low_resolution=add_low_resolution)

    # step 2 additionally compute precision and recall
    precision_recall_dict = step2_compute_precision_recall(cfg, add_low_resolution=add_low_resolution)

    # step 3: compute umap/method coordinates and make a simple umap plot (real vs synthetic)
    step3_dim_red_umap(cfg, add_low_resolution=add_low_resolution)

    ##################################################################
    # additional plot for the conditional case
    if len(cfg.model.conditions) > 0:
        step4_condition_umap(cfg)

    # merge dict output
    evaluation_metrics_dict = embedding_fid_dict.copy()
    evaluation_metrics_dict.update(precision_recall_dict)

    # remove synthetic data when done to save space
    if cfg.evaluate.save_space:
        synthetic_data_dir = cfg.evaluate.synthetic_data_dir
        try:
            shutil.rmtree(synthetic_data_dir)
            if add_low_resolution:
                shutil.rmtree(synthetic_data_dir + "_low_res")

        except OSError as exception_obj:
            print(f"Error: {exception_obj.filename}, {exception_obj.strerror}")

    return evaluation_metrics_dict


def get_eval_metrics(cfg: DictConfig, epoch_list: list, main_eval_folder: str) -> pd.DataFrame:
    """gets fid, precision and recall over epochs

    Args:
        cfg (DictConfig): config file
        epoch_list (list): list of epoch number
        main_eval_folder (str): path to evaluation folder

    Returns:
        pd.Dataframe: dataframe with evaluation metrics
    """

    # get fid, precision and recall over epochs:
    epoch_list_ok, fid_list, precision_list, recall_list = [], [], [], []
    for i_epoch in epoch_list:
        # modify cfg according to epoch info:
        with open_dict(cfg):
            cfg.evaluate.epoch_num = int(i_epoch)
            cfg.evaluate.output_folder = os.path.join(main_eval_folder, "epoch_" + str(i_epoch))
            cfg.evaluate.synthetic_data_dir = os.path.join(cfg.evaluate.output_folder, "generated_data")
            cfg.evaluate.synthetic_embeddings_dir = os.path.join(cfg.evaluate.output_folder, "embeddings")
            cfg.evaluate.figure_path = os.path.join(cfg.evaluate.output_folder, "figures")
        if not os.path.exists(os.path.join(cfg.evaluate.synthetic_embeddings_dir, "activations.npy")):
            continue
        fid = compute_fid(cfg.evaluate.origin_data_embeddings, cfg.evaluate.synthetic_embeddings_dir)
        precision, recall = step2_compute_precision_recall(cfg)
        fid_list.append(fid)
        precision_list.append(precision)
        recall_list.append(recall)
        epoch_list_ok.append(i_epoch)

    metrics_df = pd.DataFrame(
        {
            "epoch": epoch_list_ok,
            "fid": fid_list,
            "precision": precision_list,
            "recall": recall_list,
        }
    )
    logger.info(f"computed metrics_df, shape: {metrics_df.shape}, metrics_df.head():\n{metrics_df.head()}")
    return metrics_df


def get_epochs(cfg: DictConfig, checkpoint_path: str):
    """returns min_epoch, max_epoch, delta_epochs using config info and available checkpoints

    Args:
        cfg (DictConfig): config file
        checkpoint_path (str): path to the saved checkpoints

    Returns:
        min_epoch, max_epoch, delta_epochs
    """
    # check available trained model
    # takes the last checkpoint from the specified dir in cfg.train.continue_train

    checkpoints = evaluate_utils.list_checkpoints(checkpoint_path)
    if not checkpoints:
        raise ValueError(f"No checkpoints found at / under {checkpoint_path}")
    min_epoch_available = min(p[1] for p in checkpoints if p[1] is not None)
    max_epoch_available = max(p[1] for p in checkpoints if p[1] is not None)
    min_epoch = (
        cfg.evaluate.epochs.min_epoch
        if (cfg.evaluate.epochs.min_epoch and cfg.evaluate.epochs.min_epoch >= min_epoch_available and cfg.evaluate.epochs.min_epoch <= max_epoch_available)
        else min_epoch_available
    )
    max_epoch = (
        cfg.evaluate.epochs.max_epoch
        if (cfg.evaluate.epochs.max_epoch and cfg.evaluate.epochs.max_epoch <= max_epoch_available and cfg.evaluate.epochs.max_epoch >= min_epoch_available)
        else max_epoch_available
    )
    delta_epochs = (
        cfg.evaluate.epochs.delta_epochs
        if min_epoch + cfg.evaluate.epochs.delta_epochs <= max_epoch_available
        else math.ceil((max_epoch_available + 1 - min_epoch_available) / 2)
    )

    logger.info(f"Evaluate using checkpoints from epoch {min_epoch} to {max_epoch} with {delta_epochs} epoch interval")
    return min_epoch, max_epoch, delta_epochs


def delete_file(file: str) -> str:
    """Attempts to remove a file.

    This function tries to remove the specified file, and logs informational messages about whether
    the removal was successful. If the file does not exist, it logs a message stating that the file
    does not exist. If there is a PermissionError, it logs a message stating that there was a
    PermissionError. Likewise, if there is any other exception, it logs that exception.

    Parameters:
    file (str): The path to the file to delete.

    Raises:
    FileNotFoundError: If `file` does not exist.
    PermissionError: If the program does not have enough permission to delete the file.
    Exception: For general exceptions (like if the file is
        currently open or used by another process).

    Returns:
    str: message about the status of the deletion process
    """
    try:
        os.remove(file)
        return f"{file} has been removed successfully"
    except FileNotFoundError:
        return f"{file} does not exist"
    except PermissionError:
        return f"Permission denied to delete {file}"
    except Exception as exception_obj:  # pylint: disable=broad-except
        return f"Unable to delete {file} due to: {str(exception_obj)}"


def get_metrics_df(evaluation_metrics: list, epoch_list: list):
    """
    Generate a DataFrame containing selected evaluation metrics for each epoch.

    Args:
        evaluation_metrics (list): A list of dictionaries containing evaluation metrics.
        epoch_list (list): A list of epoch numbers corresponding to the evaluation metrics.

    Returns:
        pd.DataFrame: A DataFrame containing the selected evaluation metrics for each epoch.
    """
    # define columns to keep for the final df
    keep_cols_list = [
        "epoch",
        "fid",
        "precision",
        "recall",
        "fid_low_res",
        "precision_low_res",
        "recall_low_res",
    ]

    # extract info from input
    out_df = pd.DataFrame.from_dict(evaluation_metrics)
    out_df["epoch"] = epoch_list

    this_cols = [this for this in keep_cols_list if this in out_df.columns]

    # some post_processing:
    out_df = out_df.astype(
        {
            "precision": "float",
            "recall": "float",
            "precision_low_res": "float",
            "recall_low_res": "float",
        }
    )
    metrics_df = out_df[this_cols].copy(deep=True)

    # # save selected metrics into output file
    # metrics_df.to_parquet(output_file)

    return metrics_df


def _print_logger(cfg: DictConfig):
    """print model logger info

    Args:
        cfg (DictConfig): hydra config
    """
    logger.info(f"Training data:   {cfg.data.paths.training_data}")
    logger.info(f"Job directory:   {cfg.job_directory}")
    logger.info(f"training using model from: {cfg.model.net}")

    if "loss_fcn" in cfg.model.architecture:
        logger.info(f"loss function uses: {cfg.model.architecture.loss_fcn}")
        if cfg.model.architecture.loss_fcn == "all":
            logger.info(f"weight for gen loss:  {cfg.model.architecture.g_high_res_loss_weight}")
            logger.info(f"weight for disc loss: {cfg.model.architecture.d_high_res_loss_weight}")

    if "train_generators" in cfg.model.architecture:
        logger.info(f"training generators: {cfg.model.architecture.train_generators}")

    logger.info("\n")


def plot_precision_recall(epoch_list: list, precision_list: list, recall_list: list, plot_over_epochs_dir: str):
    """Plot precision and recall scores over epochs.

    Args:
        epoch_list (list): List of epoch numbers.
        precision_list (list): List of precision scores corresponding to the epochs.
        recall_list (list): List of recall scores corresponding to the epochs.
        plot_over_epochs_dir (str): Directory where the plot will be saved.
    """
    os.makedirs(plot_over_epochs_dir, exist_ok=True)

    # plot precision and recall:
    plt.figure()
    fig_filename = os.path.join(plot_over_epochs_dir, "precision_recall_score_over_epochs.png")
    plt.plot(epoch_list, precision_list, label="precision")
    plt.plot(epoch_list, recall_list, label="recall")
    plt.ylabel("score")
    plt.xlabel("# epochs")
    plt.legend()
    plt.savefig(fig_filename)


def plot_over_epochs(
    cfg: DictConfig,
    main_eval_folder: str = "",
    input_epoch_list: list = None,
    metrics_df: pd.DataFrame = pd.DataFrame(),
    plot_over_epochs_dir: str = "",
    add_low_resolution: bool = False,
):
    """takes all created figures over epochs and creates a gif

    Args:
        cfg (DictConfig): config file
    """

    if main_eval_folder == "":
        main_eval_folder = cfg.evaluate.main_evaluate_folder

    # get metrics (if metrics_df==pd.DataFrame())
    if len(metrics_df) == 0:
        logger.warning("plot_over_epochs cannot be done because empty input")
        return

    # some plots for visualizations in plot_over_epochs_dir:
    if plot_over_epochs_dir == "":
        plot_over_epochs_dir = os.path.join(main_eval_folder, "metrics_over_epochs")
    logger.info(f"plotting fid, recall and precision over epochs in folder: {plot_over_epochs_dir}")

    # plot precision and recall:
    plot_precision_recall(
        epoch_list=metrics_df["epoch"],
        precision_list=metrics_df["precision"],
        recall_list=metrics_df["recall"],
        plot_over_epochs_dir=plot_over_epochs_dir,
    )
    if add_low_resolution:
        plot_precision_recall(
            epoch_list=metrics_df["epoch"],
            precision_list=metrics_df["precision_low_res"],
            recall_list=metrics_df["recall_low_res"],
            plot_over_epochs_dir=os.path.join(plot_over_epochs_dir, "low_res"),
        )

    # plot umaps in a gif
    # fig_names = ["umap_scatter", "umap_scatter_with_intens_contours"]
    fig_names = [
        "umap_scatter",
        "umap_scatter_with_intens_contours",
        "thumbnails_epoch*_complete",
    ]

    # get plots per epoch
    fig_sub_folder = "/epoch_*/figures/"

    png_files_list = [glob.glob(f"{main_eval_folder}" + fig_sub_folder + fig_name + ".png") for fig_name in fig_names]
    gif_filenames_list = [os.path.join(plot_over_epochs_dir, i_fig_name + "_over_epochs.gif") for i_fig_name in fig_names]

    list(
        map(
            lambda i_png_files, i_gif_filenames_list: make_gif(png_files=i_png_files, gif_filename=i_gif_filenames_list),
            tqdm(png_files_list, desc="plots over epochs"),
            gif_filenames_list,
        )
    )

    if add_low_resolution:
        # same plot for low resolution case:
        fig_sub_folder = "/epoch_*/figures/low_res/"

        low_res_png_files_list = [glob.glob(f"{main_eval_folder}" + fig_sub_folder + fig_name + ".png") for fig_name in fig_names]
        sub_folder_path = os.path.join(plot_over_epochs_dir, "low_res")
        os.makedirs(sub_folder_path, exist_ok=True)
        gif_output_filenames_list = [os.path.join(sub_folder_path, i_fig_name + "_over_epochs.gif") for i_fig_name in fig_names]

        list(
            map(
                lambda i_png_files, i_gif_filenames_list: make_gif(png_files=i_png_files, gif_filename=i_gif_filenames_list),
                tqdm(low_res_png_files_list, desc="plots over epochs"),
                gif_output_filenames_list,
            )
        )


@hydra.main(
    config_path=os.path.abspath(os.path.join(SRC_PATH, "..", "conf")),
    config_name="default",
    version_base="1.3",
)
@logger.catch
def main(cfg: DictConfig):
    """main function for GANs evaluations
    note: in cfg.evaluate you can additionally specify the following behavior:
    - epochs: can specify the min_epoch, max_epoch and delta_epochs to use for evaluation
    - save_space: True, delete synthetic data and save disc space
    - evaluate_low_resolution: True for additional low resolution plots

    the main evaluation function includes the following steps:
    - step 1: generate synthetic data
    - step 2: compute embeddings (using mednet)
    - step 3: compute umap/method coordinates and make a simple umap plot (real vs synthetic)
    - step 4: (optional) additional plot for the conditional case

    Args:
        cfg (DictConfig): config files with all info/variable
    """
    assert os.path.isdir(cfg.data_basedir) and os.path.isdir(cfg.runs_basedir), (
        f"Ensure base directories have been passed -- use hydra arg overrides for these: python {__file__} data_basedir=<...> runs_basedir=<...>"
    )

    # load the saved config from hydra run:
    checkpoint_path = os.path.join(cfg.train.logger.save_dir, cfg.train.logger.name, cfg.train.logger.version)
    assert os.path.isdir(checkpoint_path), f"not a valid directory specified in hydra.run.dir: {checkpoint_path}"

    # retrieve config from saved params
    temp = OmegaConf.load(checkpoint_path + "/hparams.yaml")
    cfg_training = temp.cfg

    # here replace with any new evaluate change at call
    cfg_new = cfg_training
    cfg_new.evaluate = cfg.evaluate
    cfg = cfg_new

    _print_logger(cfg)

    # note: if the number of data is not divisible by batch_size -> then,
    # since we use the dataloader with drop_last=True, we are not computing all
    # embeddings.
    # In this case we set batch_size=1 to be ensure evaluation of all data
    with open_dict(cfg):
        cfg.evaluate.checkpoint_path = checkpoint_path
        cfg.train.optimizers.batch_size = 1

    logger.info(f"evaluate checkpoints from folder:\n{checkpoint_path}")

    add_low_resolution = cfg.evaluate.evaluate_low_resolution
    if add_low_resolution:
        logger.info("with the evaluation of low resolution branch")

    min_epoch, max_epoch, delta_epochs = get_epochs(cfg=cfg, checkpoint_path=checkpoint_path)
    epoch_list = np.arange(min_epoch, int(max_epoch + 1), delta_epochs)

    # check which epochs were already computed and add the missing once:
    main_eval_folder = cfg.evaluate.main_evaluate_folder  # os.path.dirname(cfg.evaluate.output_folder)
    metrics_filename = os.path.join(main_eval_folder, "metrics_result.parquet")
    if os.path.isfile(metrics_filename):
        metrics_df = pd.read_parquet(metrics_filename)
        done_epochs = list(metrics_df.epoch)
        epoch_list = [this for this in epoch_list if this not in done_epochs]

    logger.info(f"running evaluation over {len(epoch_list)} epochs, outputs in folder {main_eval_folder} (epochs: {epoch_list})")

    if len(epoch_list) >= 1:
        evaluation_metrics_list = list(
            map(
                lambda this_epoch_number: main_evaluate(
                    cfg=cfg,
                    epoch_num=this_epoch_number,
                    add_low_resolution=add_low_resolution,
                ),
                tqdm(epoch_list, desc="running evaluation over epochs"),
            )
        )
        logger.info("evaluation metrics computed for all epochs")

        # save metrics as a dataframe
        new_metrics_df = get_metrics_df(evaluation_metrics=evaluation_metrics_list, epoch_list=epoch_list)

        # save full metrics into output file
        if os.path.isfile(metrics_filename):
            full_metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)
            full_metrics_df.to_parquet(metrics_filename)
        else:
            new_metrics_df.to_parquet(metrics_filename)

    # metrics_df = pd.read_csv(csv_filename)
    metrics_df = pd.read_parquet(metrics_filename)
    plot_over_epochs(cfg=cfg, metrics_df=metrics_df, add_low_resolution=add_low_resolution)
    logger.info(f"plots available in {cfg.evaluate.main_evaluate_folder}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
