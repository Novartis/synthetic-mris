import glob
import os
import pickle

import numpy as np
import umap


class UmapReducer:
    """Class for UMAP dimensionality reduction that roughly matches the sklearn manifolds' behaviour."""

    def __init__(
        self,
        train_datasets_root: str,
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
        if os.path.isfile(os.path.join(train_datasets_root, "umap_model.pickle")) and not retrain:
            with open(os.path.join(train_datasets_root, "umap_model.pickle"), "rb") as f:
                self.reducer = pickle.load(f)

        # else load training data, train umap model and save pickle
        else:
            train_data = []
            # load from root folder
            try:
                train_data.append(np.load(os.path.join(train_datasets_root, filename)))
            except Exception as e:
                print(f"{e}: Dataset {train_datasets_root} does not contain valid activations, skipping.")
            datasets = glob.glob(f"{train_datasets_root}/*/")
            for dataset in datasets:
                try:
                    train_data.append(np.load(os.path.join(dataset, filename)))
                except Exception as e:
                    print(f"{e}: Dataset {dataset} does not contain valid activations, skipping.")

            self.train_data = np.concatenate(train_data, axis=0)

            self.reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                n_components=n_components,
            )
            self.reducer.fit(self.train_data)
            with open(os.path.join(train_datasets_root, "umap_model.pickle"), "wb") as f:
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
