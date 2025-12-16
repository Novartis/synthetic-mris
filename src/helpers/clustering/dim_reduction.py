import os
import pickle

import numpy as np
import umap
from sklearn.manifold import MDS, TSNE


class UmapReducer:
    """Class for UMAP dimensionality reduction that roughly matches the sklearn manifolds' behaviour."""

    def __init__(
        self,
        umap_model_file: str,
        train_embeddings_files: str,
        n_components: int = 2,
        n_neighbors: int = 20,
        min_dist: float = 0.0,
        filename: str = "activations.npy",
        retrain: bool = False,
    ):
        """Sets up all necessary member variables and trains the UMAP if no pretrained model is found in the
        train_datasets_root directory.

        Args:
            train_datasets_root (str): String pointing to the root folder containing training datasets.
            n_neighbors (int, optional): Number of neighbors used for UMAP processing. Defaults to 100.
            min_dist (float, optional): Minimum distance used for UMAP processing. Defaults to 0.001.
            filename (str, optional): File name of the activations to be loaded and processed.
                Defaults to 'activations.npy'.
            retrain (bool, optional): If set to True, umap model will be retrained, disregarding any
                available pre-trained models in the directory. Defaults to False.
        """
        # check for pretrained umap in the given train_datasets_root. If found, load.

        if os.path.isfile(umap_model_file) and not retrain:
            with open(umap_model_file, "rb") as f:
                self.reducer = pickle.load(f)

        # else load training data, train umap model and save pickle
        else:
            train_data = []
            # load embeddings to be used for umap train
            try:
                for i_train_data in train_embeddings_files:
                    train_data.append(np.load(i_train_data))
            except Exception as e:
                raise Exception(f"{e}: not valid files {train_embeddings_files}")

            self.train_data = np.concatenate(train_data, axis=0)

            self.reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                n_components=n_components,
            )
            self.reducer.fit(self.train_data)
            with open(umap_model_file, "wb") as f:
                pickle.dump(self.reducer, f)

    def fit_transform(self, eval_data: np.array):
        """Loads the supplied eval_dataset and performs the UMAP dimensionality reduction.
            Returns the reduced activation tensor.

        Args:
            eval_dataset (np.array): Array containing eval activations.

        Returns:
            [np.array]: n_components-dimensional array containing the reduced activations.
        """
        return self.reducer.transform(eval_data)


def dim_reduction(method: str, activation_file: str, train_embeddings_files, n_components=2, **kwargs):
    # TODO: change input so that either dataset paths and training paths can be used, or pre loaded numpy arrays
    """Function to apply dimensionality reduction using mds, t-sne or umap to an input tensor of dimensionality (samples, dim).
    Returns an array of dim (samples,n_components).

    Args:
        dataset_path (str): directory containing the activation numpy tensor
        method (str): string describing which method to use. Currently "mds", "tsne", and "umap" are supported.
        train_data_path (str, optional): If using umap, a directory root containing the training dataset(s) must be supplied. Defaults to None.
        n_components (int, optional): Number of components to reduce to. Defaults to 2.
        n_neighbors (int, optional): Number of neighbors for UMAP processing. Defaults to 100.
        n_components (int, optional): Minimum distance for UMAP processing. Defaults to 0.001.
        **kwargs: Optional keyword arguments for umap processing. See UmapReducer documentation.
    Returns:
        [np.array]: The reduced activation tensor as numpy array.
    """
    # check inputs for incorrect data types
    assert isinstance(method, str), f"method should be a string defining the method to be used, received {method} of type {type(method)}."
    assert isinstance(n_components, int), f"n_components should be an int defining the output dimension, received {n_components} of type {type(n_components)}."

    # load activation from dataset_path
    if not os.path.isfile(activation_file):
        raise Exception(f"file {activation_file} does not exists")
    activations = np.load(activation_file)
    # path, filename = os.path.split(activation_file)

    umap_dir, _ = os.path.split(train_embeddings_files[0])

    # select reduction method according to "method" string and prepare reducer instance
    if method == "mds":
        reducer = MDS(n_components)
    elif method == "tsne":
        reducer = TSNE(n_components)
    elif method == "umap":
        # make sure the required training dataset is supplied
        # assert train_data_path is not None and isinstance(
        #     train_data_path, str), f'When using UMAP, a training dataset must be supplied, received {train_data_path}.'
        umap_model_file = os.path.join(umap_dir, "umap_model.pickle")  # saved umap model for this case
        reducer = UmapReducer(
            umap_model_file=umap_model_file,
            train_embeddings_files=train_embeddings_files,
            n_components=n_components,
            **kwargs,
        )
    else:
        raise NotImplementedError(f'Currently only "tsne", "mds" and "umap" keywords accepted, but {method} given.')

    # perform dimensionality reduction
    act_transformed = reducer.fit_transform(activations)

    return act_transformed
