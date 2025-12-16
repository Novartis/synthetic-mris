"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

code from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
adapted for medicalnet and 3d images

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code adapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool3d
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Number of processes to use for data loading",
)
parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("Paths to the generated images or to .npz statistic files"),
)

IMAGE_EXTENSIONS = {"np", "npy", "pkl", "pickle"}


def get_activations(model, dataset, batch_size=50, dims=2048, device="cpu", num_workers=8):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
    """

    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    pred_arr = np.empty((len(dataset) - len(dataset) % batch_size, dims))

    start_idx = 0

    # for idx in range(len(dataset)):
    #     try:
    #         tt=dataset.__getitem__(idx)
    #     except:
    #         tt=dataset.__getitem__(idx)
    #         print(idx)

    # # dataloader_iterator = iter(dataloader)
    # # X, Y = next(dataloader_iterator)
    # try:
    #     for batch in tqdm(dataloader, desc="compute activations"):
    #         batch = batch.to(device)
    # except:
    #     pass

    for batch in tqdm(dataloader, desc="compute activations"):
        batch = batch.to(device)
        if len(batch.shape) == 4:
            batch = batch.unsqueeze(axis=1)

        with torch.no_grad():
            pred = model(batch)  # pred:torch.Size([2, 2048, 32, 32, 32])

        pred = adaptive_avg_pool3d(pred, output_size=1)  # pred: torch.Size([2, 2048, 1, 1, 1])

        pred = pred.view(batch_size, dims).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def mu_sigma_from_act(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_fid(embeddings_file_dataset1: str, embeddings_file_dataset2: str):
    """computes Fréchet inception distance (FID)
    the FID compares the distribution of generated images with the distribution of a set of real images
    ("ground truth").
    Interpretation: the lower the better
    Classifiers that achieve the best (lowest) FID score tend to have greater sample variety

    Args:
        embeddings_file_dataset1 (str): file to the saved embeddings of one dataset
        embeddings_file_dataset2 (str): file to the saved embeddings of another dataset

    Returns:
        float: _description_
    """
    assert os.path.isfile(embeddings_file_dataset1), f"File does not exist: {embeddings_file_dataset1}"
    assert os.path.isfile(embeddings_file_dataset2), f"File does not exist: {embeddings_file_dataset2}"

    act1 = np.load(embeddings_file_dataset1)
    act2 = np.load(embeddings_file_dataset2)
    mu1, sigma1 = mu_sigma_from_act(act1)
    mu2, sigma2 = mu_sigma_from_act(act2)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
