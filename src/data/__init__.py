import glob
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset, random_split


def prepare_clinical_data(clinical_data_dir: str, clinical_data_processing: Dict, conditions: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Handles raw clinical data and creates dataset for MMCGAN/CopulaGAN training.
    A columns list is taken to define which columns are kept and processed, NAs can be removed if required.
    Additional compound columns such as categorical bmi and progression-free survival for more than one year are added.

    Args:
        clinical_data_dir: path to clinical data
        clinical_data_processing: dict with clinical processing parameters
        conditions: dict with condition

    Returns:
        pd.DataFrame: curated clinical data for training.
    """

    if clinical_data_dir.endswith(".parquet") or clinical_data_dir.endswith(".pq"):
        ds = pd.read_parquet(clinical_data_dir)
    elif clinical_data_dir.endswith(".csv") or clinical_data_dir.endswith(".csv.gz"):
        ds = pd.read_csv(clinical_data_dir)
    else:
        raise ValueError("Unknown file type for clinical data -- this must be either .csv or Parquet.")

    # conditions must not be missing
    if conditions:
        ds = ds.dropna(subset=conditions)

    # ensure all columns defined in columns list are available in dataset
    for column, columntype in clinical_data_processing["columns"].items():
        assert column in ds, f"Requested column {column} not found in dataset."
        if columntype == "category":
            ds.loc[:, column] = ds.loc[:, column].astype(pd.CategoricalDtype(ordered=True))
            logger.info(f"Converting column {column} to numerical codes. The mapping is as follows:\n{dict(enumerate(ds[column].cat.categories))}")
        elif columntype == "int_category":
            ds.loc[:, column] = ds.loc[:, column].fillna(sys.maxsize).astype("int").astype(pd.CategoricalDtype(ordered=True))
            logger.info(f"Converting column {column} to ints, then numerical codes. The mapping is as follows:\n{dict(enumerate(ds[column].cat.categories))}")
        else:
            ds.loc[:, column] = ds.loc[:, column].astype(columntype)

    conditions_dict = {col: len(ds[col].unique()) for col in conditions}
    return ds, conditions_dict


def compute_pads(in_size: int, out_size: int):
    """Computes the size of the padding to be applied on either side for a single dimesion. If out_size is smaller than in_size, pad length is clipped to zero.

    Args:
        in_size (int): input size
        out_size (int): desired output size

    Returns:
        tuple[int, int]: Pad lengths for both sides.
    """
    if in_size < 0 or out_size < 0:
        raise ValueError(f"Vector lengths should be positive integers but received {(in_size, out_size)}.")

    diff = out_size - in_size
    if diff <= 0:
        pad_len_1 = pad_len_2 = 0
    else:
        pad_len_1 = diff // 2
        pad_len_2 = diff - pad_len_1
    return pad_len_1, pad_len_2


def pad(image: "np.array", outsize: Union[Tuple[int, int, int], int]):
    """Pads image symmetrically accoring to desired outsize.
       If image is larger than outsize on a specific axis, the larger input shape is maintained for that axis.

    Args:
        image (np.array): Input image array of shape DHW.
        outsize (Union[Tuple[int, int, int], int]): Desired output size.
        If outsize is given as int, then padding is applied on all axes using outsize.
        If tuple of ints is given, then padding is applied accordingly to the individual axes.

    Returns:
        np.array: padded image
    """

    # set desired image output dimensions
    if isinstance(outsize, int):
        outsize = [outsize] * 3

    size_d, size_h, size_w = outsize

    # get image shape per dimension and image data type
    img_d, img_h, img_w = image.shape
    dtype = image.dtype

    # set output size to be the bigger choice between the input size and the desired outsize.
    size_d = max(size_d, img_d)
    size_h = max(size_h, img_h)
    size_w = max(size_w, img_w)

    # pad along the separate axes
    pad_len_1, pad_len_2 = compute_pads(img_d, size_d)

    pad1 = np.zeros((pad_len_1, img_h, img_w), dtype=dtype)
    pad2 = np.zeros((pad_len_2, img_h, img_w), dtype=dtype)

    image = np.concatenate([pad1, image, pad2], axis=0)

    pad_len_1, pad_len_2 = compute_pads(img_h, size_h)
    pad1 = np.zeros((size_d, pad_len_1, img_w), dtype=dtype)
    pad2 = np.zeros((size_d, pad_len_2, img_w), dtype=dtype)

    image = np.concatenate([pad1, image, pad2], axis=1)

    pad_len_1, pad_len_2 = compute_pads(img_w, size_w)
    pad1 = np.zeros((size_d, size_h, pad_len_1), dtype=dtype)
    pad2 = np.zeros((size_d, size_h, pad_len_2), dtype=dtype)

    image = np.concatenate([pad1, image, pad2], axis=2)

    return image


def image_resize(
    image: np.array,
    size_d: int,
    size_h: int,
    size_w: int,
    mode: str = "vertical_center",
):
    """Resizes image according to the defined size per dimension. Multiple modes supported.
    Cropping along vertical axis possible with "vertical_top", "vertical_bottom", "vertical_center".
    If any axis of the input image is smaller than the desired output shape, the tensor is symmetrically padded with zeros.

    Args:
        image (np.array): Input array of shape DHW
        size_d (int): Desired output shape along D-axis.
        size_h (int): Desired output shape along H-axis.
        size_w (int): Desired output shape along W-axis.
        mode (str, optional): Mode for cropping. Supports vertical cropping with
            "vertical_top", "vertical_bottom", "vertical_center". Defaults to 'vertical_center'.


    Returns:
        np.array: cropped input array of shape [size_d, size_h, size_w].
    """
    # extract shape of image
    im_shape = np.array(image.shape, dtype=int)
    img_d, img_h, img_w = image.shape

    if any([size_d > img_d, size_h > img_h, size_w > img_w]):
        image = pad(image, (size_d, size_h, size_w))
        im_shape = np.array(image.shape, dtype=int)

    im_center = np.array(im_shape / 2, dtype=int)

    # define start and end values for indexing according to the defined cropping mode.
    if mode == "vertical_center":
        st_d = int(np.floor(im_center[0])) - int(np.floor(size_d / 2))
        end_d = int(np.ceil(im_center[0])) + int(np.ceil(size_d / 2))
    elif mode == "vertical_top":
        st_d = int(0)
        end_d = int(size_d)
    elif mode == "vertical_bottom":
        img_d_pad, _, _ = image.shape
        st_d = int(img_d_pad - size_d)
        end_d = int(img_d_pad)
    else:
        raise ValueError(f"Unknown crop mode {mode}.")

    # set start and end points for H and W axis using center crop mode.
    st_h = int(np.floor(im_center[1])) - int(np.floor(size_h / 2))
    end_h = int(np.ceil(im_center[1])) + int(np.ceil(size_h / 2))

    st_w = int(np.floor(im_center[2])) - int(np.floor(size_w / 2))
    end_w = int(np.ceil(im_center[2])) + int(np.ceil(size_w / 2))

    # apply cropping by indexing input image.
    im_crop = image[st_d:end_d, st_h:end_h, st_w:end_w]

    return im_crop


class BaseImageDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        image_crop: Optional[Tuple[int, int, int]] = None,
        downsample: Optional[Tuple[int, int, int]] = None,
        crop_mode: str = "vertical_center",
    ):
        """Dataset class used with pickled images. The dataset implements a map-based style, images can be loaded using dataset[idx].
        Additionally, class objects have the ability to crop and downsample on-the-fly.

        Args:
            dataset_dir (str): Image pickles location. Should either be directly referenced as numpy objects of shape d x h x w, as dictionary with
                the image stored under the key 'image' or as tuple containing said dictionary as the first entry.
            image_crop (tuple[int, int, int], optional): If set, images are cropped to the defined dimension.
                If inputs are smaller than crop size in any dimension, the image is zero-padded. Type of cropping is defined by "crop_mode". Defaults to None.
            downsample (tuple[int, int, int], optional): Downsampling factor per dimension. No downsampling is performed and only integer downsampling factors
                are allowed. Defaults to None.
            crop_mode (str, optional): Cropping mode. Defaults to 'vertical_center'.
                Current options are 'vertical_top', 'vertical_center', and 'vertical_bottom'.
        """
        self.dataset_dir = dataset_dir
        self.file_paths = []
        self.image_crop = image_crop
        self.crop_mode = crop_mode

        # Make sure the dataset location was given.
        if not (dataset_dir and os.path.isdir(dataset_dir)):
            raise ValueError(f"The supplied path {dataset_dir} does not lead to a valid directory.")
        # recursively scan folder structure for images (i.e. files ending in ".pickle")
        self.file_paths = glob.glob(os.path.join(dataset_dir, "**", "*.pickle"), recursive=True)

        self.downsample = None
        if downsample is not None:
            # If downsample is [1, 1, ,1], revert to None
            if any(ds > 1 for ds in downsample):
                self.downsample = downsample

    def __len__(self):
        return len(self.file_paths)

    def load_image(self, path):
        with open(path, "rb") as f:
            # print(path)
            sample = pickle.load(f)
            if isinstance(sample, tuple):
                sample = sample[0]["image"]
            elif isinstance(sample, dict):
                sample = sample["image"]
        return sample

    def reshape_image(self, image):
        # crop image according to defined parameters
        if self.image_crop is not None and any([crop is not None for crop in self.image_crop]):
            image = image_resize(image, *self.image_crop, mode=self.crop_mode)

        # downsample image according to defined downsampling ratios.
        if self.downsample:
            image = image[:: self.downsample[0], :: self.downsample[1], :: self.downsample[2]]
        return image

    def __getitem__(self, idx):
        # load pickled image
        sample = self.load_image(self.file_paths[idx])
        sample = self.reshape_image(sample)

        sample = torch.tensor(sample, dtype=torch.float)
        return sample


class ConditionalDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_dir: str,
        clinical_data_dir: str,
        clinical_data_processing: Dict,
        conditions_list: List,
        image_crop: Optional[Tuple[int, int, int]] = None,
        downsample: Optional[Tuple[int, int, int]] = None,
        crop_mode: str = "vertical_center",
    ):
        """Conditional dataset based on BaseImageDataset class.
        Behaves mostly like parent class, except for the required argument defining the clinical data directory.

        Args:
            dataset_dir (str): Image pickles location. Should either be directly referenced as numpy objects of shape d x h x w, as dictionary with
                the image stored under the key 'image' or as tuple containing said dictionary as the first entry.
            clinical_data_dir (str): String containing the clinical data associated with the image files. The data can either be in parquet format
                (file ending in 'pq' or 'parquet') or as pandas csv (file name ending in csv or csv.gz).
            clinical_data_processing (Dict): dict mapping columns to types, default is to read all columns as strings
            conditions_list (List): list of columns that are used for conditioning
            image_crop (tuple[int, int, int], optional): If set, images are cropped to the defined dimension.
                If inputs are smaller than crop size in any dimension, the image is zero-padded. Type of cropping is defined by "crop_mode". Defaults to None.
            downsample (tuple[int, int, int], optional): Downsampling factor per dimension. No downsampling is performed and only integer downsampling factors
                are allowed. Defaults to None.
            crop_mode (str, optional): Cropping mode. Defaults to 'vertical_center'.
                Current options are 'vertical_top', 'vertical_center', and 'vertical_bottom'.
        """

        super().__init__(
            dataset_dir=dataset_dir,
            image_crop=image_crop,
            downsample=downsample,
            crop_mode=crop_mode,
        )

        # declare here for later
        self.usubjid_to_file_path = {}
        self.file_path_to_usubjid = {}

        # add conditional info
        self.clinical_data_processing = clinical_data_processing

        self.clinical, self.conditions_dict = prepare_clinical_data(clinical_data_dir, self.clinical_data_processing, conditions_list)

        self.conditions_in_order = []
        for this_condition, ccount in sorted(self.conditions_dict.items()):
            logger.info(f"Condition column {this_condition} has {ccount} distinct values")
            self.conditions_in_order.append((this_condition, ccount))

        usubjid_from_clinical = set(self.clinical[self.clinical_data_processing.identifier])

        self.usubjid_to_file_path = {}
        self.file_path_to_usubjid = {}

        updated_file_paths = []
        for path in self.file_paths:
            usubjid = path.rsplit(os.path.sep, 1)[1].split("_")[0]
            if usubjid in usubjid_from_clinical:
                updated_file_paths.append(path)
                self.file_path_to_usubjid[path] = usubjid
                if usubjid in self.usubjid_to_file_path:
                    self.usubjid_to_file_path[usubjid].append(path)
                else:
                    self.usubjid_to_file_path[usubjid] = [path]
        self.file_paths = updated_file_paths

    def get_conditions(self):
        # ordered list of tuples condition, count giving
        # the number of possible values for each condition
        return self.conditions_in_order

    def __getitem__(self, idx: int):
        # get file path string.
        file_path = self.file_paths[idx]

        # get row of clinical data matching unique subject ID
        subjid = self.file_path_to_usubjid[file_path]
        features = self.clinical.loc[self.clinical[self.clinical_data_processing.identifier].isin([subjid])]

        assert len(features) == 1, f"Invalid number of feature vectors received. Should be 1 but found {len(features)}."

        # load image file.
        sample = super().__getitem__(idx)

        conditions = np.asarray([features[feature].item() for feature, _ in self.conditions_in_order], dtype=int)

        return sample, conditions


def make_train_validation_split(full_dataset, split_fraction=0.9):
    """
    Creates a train and validation split of a given dataset.

    Args:
        full_dataset ([type]): Dataset to be split
        split_fraction (int or float, optional): Percentage or absolute number of samples to be used for train set.
        Floats are treated as percentage, ints are treated as absolute number of samples. Defaults to 0.9.

    Returns:
        Tuple: train and validation datasets
    """

    num_samples = len(full_dataset)
    if isinstance(split_fraction, int):
        train_split = split_fraction
    else:
        train_split = int(split_fraction * num_samples)
    train_dataset, validation_dataset = random_split(
        full_dataset,
        [train_split, num_samples - train_split],
        generator=torch.Generator(),
    )
    return train_dataset, validation_dataset
