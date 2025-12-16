# This script clusters original MRI images in UMAP space and saves the results in an Excel file.
# It uses DBSCAN and Gaussian Mixture methods for clustering and generates density plots.
# The script should be run after generating UMAP embeddings of the original images.

## Load Packages
import os
import sys

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig, open_dict
from scipy.stats import kde

# clustering methods
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# load local functions
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(SRC_PATH)

# import project specific functions
from data import BaseImageDataset, ConditionalDataset


@hydra.main(
    config_path=os.path.abspath(os.path.join(SRC_PATH, "..", "conf")),
    config_name="default",
)
@logger.catch
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.data.paths.training_data = os.path.join(cfg.data.paths.training_data, "T1")
        cfg.model.conditions = ["cluster_id"]

    model_name = cfg.model.net._target_
    model_name = model_name.split(".")[-1].lower()

    ## Load data

    # Look at original data in mni space
    dataset_dir = cfg.data.paths.training_data
    data_crop_size = cfg.data.image_processing.data_crop_size
    data_downsampling_factor = cfg.data.image_processing.data_downsampling_factor
    crop_mode = cfg.data.image_processing.crop_mode
    image_crop = (tuple([data_crop_size] * 3)) if data_crop_size is not None else None

    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"image_crop: {image_crop}")
    logger.info(f"crop_mode: {crop_mode}")
    logger.info(f"data_downsampling_factor: {data_downsampling_factor}")

    dataset_dir = cfg.data.paths.training_data
    conditional = cfg.model.conditions

    if conditional:
        dataset = ConditionalDataset(
            dataset_dir=cfg.data.paths.training_data,
            clinical_data_dir=cfg.data.paths.clinical_data,
            clinical_data_processing=cfg.data.clinical_data_processing,
            conditions_list=cfg.model.conditions,
        )
    else:
        dataset = BaseImageDataset(
            cfg.data.paths.training_data,
            image_crop=(256, 256, 256),
            downsample=(1, 1, 1),
        )

    logger.info(f"dataset includes {len(dataset)} elements")

    ## Cluster the original data in umap space
    original_data_embeddings = os.path.join(cfg.evaluate.main_evaluate_folder, "original_data_embeddings")
    umap_origin_coords_file = os.path.join(original_data_embeddings, "umap_coords.npy")
    # or/alternatively use the embeddings: the embedding_file
    # embedding_file = os.path.join(original_data_embeddings, "activations.npy")

    if not os.path.isfile(umap_origin_coords_file):
        raise FileNotFoundError(f"The file at path {umap_origin_coords_file} does not exist.\n run evaluation script to generate it")

    umap_origin_coords = np.load(umap_origin_coords_file)
    logger.info(f"loaded coordinates #{umap_origin_coords.shape} from file:\n{umap_origin_coords_file}")

    ### Look at the graph
    # density plot from real data
    x, y = umap_origin_coords[:, 0], umap_origin_coords[:, 1]

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    dx, dy = 0.05, 0.05
    z_this = zi.reshape(xi.shape)
    levels = MaxNLocator(nbins=15).tick_values(zi.min(), zi.max())
    plt.figure(figsize=(6, 6))
    plt.contourf(xi + dx / 2.0, yi + dy / 2.0, z_this, levels=levels, cmap="Blues")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(original_data_embeddings, "density_plot.png"))

    ### DBSCAN method
    # DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise
    clustering = DBSCAN().fit(umap_origin_coords)

    cluster_id = clustering.labels_
    logger.info(f"identified {len(set(cluster_id))} unique labels: {np.unique(cluster_id)}")
    # get the counts of each label
    cluster_series = pd.Series(cluster_id)
    cluster_counts_pandas = cluster_series.value_counts()
    logger.info("Cluster counts (DBSCAN):")
    logger.info(f"\n{cluster_counts_pandas}")

    plt.figure(figsize=(6, 6))
    plt.contourf(xi + dx / 2.0, yi + dy / 2.0, z_this, levels=levels, cmap="Blues")
    for i_grp in np.unique(cluster_id):
        mask = cluster_id == i_grp
        plt.scatter(x=x[mask], y=y[mask], s=2, label=i_grp)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(original_data_embeddings, "DBSCAN_clustering_plot.png"))

    ### Gaussian mixture method
    gauss_mix = GaussianMixture(n_components=5, random_state=0).fit_predict(umap_origin_coords)

    logger.info(f"identified {len(set(gauss_mix))} unique labels: {np.unique(gauss_mix)}")
    # get the counts of each label
    gm_cluster_counts_pandas = pd.Series(gauss_mix).value_counts()
    logger.info("Cluster counts using Gaussian mixture method:")
    logger.info(f"\n{gm_cluster_counts_pandas}")

    plt.figure(figsize=(6, 6))
    plt.contourf(xi + dx / 2.0, yi + dy / 2.0, z_this, levels=levels, cmap="Blues")
    for i_grp in np.unique(gauss_mix):
        mask = gauss_mix == i_grp
        plt.scatter(x=x[mask], y=y[mask], s=2, label=i_grp)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(original_data_embeddings, "GaussianMixture_clustering_plot.png"))

    ### define random clusters
    activation_file = os.path.join(original_data_embeddings, "files_in_activations.csv")
    if not os.path.isfile(activation_file):
        raise FileNotFoundError(f"The file at path {activation_file} does not exist.\n run evaluation script to generate it")
    activation_id = pd.read_csv(activation_file)

    filenames_df = activation_id["0"].str.rsplit(os.path.sep, expand=True).iloc[:, -1]
    real_id = filenames_df.str.split("_", expand=True)[0]
    real_id.name = "IXI_ID"
    random_cluster = torch.randint(0, 6, (len(real_id),)).numpy()

    logger.info(f"defined {len(set(random_cluster))} unique labels: {np.unique(random_cluster)}")
    rand_cluster_counts_pandas = pd.Series(random_cluster).value_counts()
    logger.info("Cluster counts (random):")
    logger.info(f"\n{rand_cluster_counts_pandas}")

    # save clusters
    rand_cluster = pd.Series(random_cluster, name="rand_cluster", dtype=pd.Int64Dtype)
    den_cluster = pd.Series(cluster_id, name="cluster_id", dtype=pd.Int64Dtype)
    cluster_table = pd.concat([real_id, den_cluster, rand_cluster], axis=1)

    cluster_table.head(4)

    ### Save it
    clinical = pd.read_csv(cfg.data.paths.clinical_data)
    # clinical["IXI_ID"]=clinical["IXI_ID"].astype('string')
    clinical.__len__()
    columns_to_remove = ["cluster_id", "rand_cluster"]
    clinical = clinical.drop(columns=columns_to_remove)

    new_clinical = clinical.join(cluster_table.set_index("IXI_ID"), on="IXI_ID")

    new_clinical["cluster_id"] = new_clinical["cluster_id"].fillna(20)
    new_clinical["rand_cluster"] = new_clinical["rand_cluster"].fillna(20)

    # new_clinical.dtypes
    clinical_data_dir = cfg.data.paths.clinical_data.rsplit(os.path.sep, 1)[0]
    new_clinical_file = os.path.join(clinical_data_dir, "demographic_info_IXI_postprocessed_cluster_new.csv")
    new_clinical.to_csv(new_clinical_file)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
