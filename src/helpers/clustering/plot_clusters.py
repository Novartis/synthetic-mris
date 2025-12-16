import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from scipy.stats import kde


def density_distr(
    coords_all,
    figsize,
    save_path: str,
    method: str = "undefined",
    labels=None,
    marker_size: float = 1,
    dpi: int = 300,
):
    plt.figure(figsize=figsize)
    # density plot from real data
    coords = coords_all[0]
    x, y = coords[:, 0], coords[:, 1]

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot: https://python-graph-gallery.com/85-density-plot-with-matplotlib/
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Blues', alpha=0.8)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="Blues")

    # plot the synthetic data on top:
    coords = coords_all[1]
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        label=labels[1],
        c="black",
        edgecolors="red",
        marker=".",
        s=marker_size,
    )

    plt.show()
    # add legend to plot and save under save_path.
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{method}_scatter_with_intens.png"), dpi=dpi)
    plt.close()

    # same using contours: https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_levels.html
    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05
    z_this = zi.reshape(xi.shape)
    levels = MaxNLocator(nbins=15).tick_values(zi.min(), zi.max())
    plt.figure(figsize=figsize)
    plt.contourf(xi + dx / 2.0, yi + dy / 2.0, z_this, levels=levels, cmap="Blues")

    # plot only synythetic data on top
    coords = coords_all[1]
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        label=labels[1],
        c="black",
        edgecolors="red",
        marker=".",
        s=marker_size,
    )

    plt.show()
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{method}_scatter_with_intens_contours.png"), dpi=dpi)
    plt.close()

    return


def draw_simple_ellipse(
    position,
    width,
    height,
    angle,
    ax=None,
    from_size=0.1,
    to_size=0.5,
    n_ellipses=3,
    alpha=0.1,
    color=None,
    **kwargs,
):
    ax = ax or plt.gca()
    angle = (angle / np.pi) * 180
    width, height = np.sqrt(width), np.sqrt(height)
    # Draw the Ellipse
    for nsig in np.linspace(from_size, to_size, n_ellipses):
        ax.add_patch(
            Ellipse(
                position,
                nsig * width,
                nsig * height,
                angle,
                alpha=alpha,
                lw=0,
                color=color,
                **kwargs,
            )
        )


# code from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def plot_ellipse_on_top(
    coords_all,
    figsize,
    save_path: str,
    method: str = "undefined",
    labels=None,
    marker_size: float = 1,
    dpi: int = 300,
):
    # ref : https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    plt.figure(figsize=figsize)
    # iterate through all dataset paths
    color = ["blue", "orange", "green"]
    for idx, coords in enumerate(coords_all):
        # add method to scatter plot
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color[idx],
            label=labels[idx],
            marker=".",
            s=marker_size,
        )
        ax = plt.gca()
        # get color from scatter plot
        # color=sc.to_rgba(c[idx])
        confidence_ellipse(coords[:, 0], coords[:, 1], ax, edgecolor=color[idx])

    # add legend to plot and save under save_path.
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{method}_scatter_with_ellipse.png"), dpi=dpi)
    plt.close()


def plot_clusters(
    dataset_dirs: list,
    method: str,
    save_path: str,
    figsize: tuple = (6, 6),
    suffix: str = "_coords.npy",
    marker_size: float = 1,
    dpi: int = 300,
    labels=None,
):
    """This method takes precalculated two-dimensional coordinates from dimensionality reduction algorithms such as PCA or UMAP and creates the associated scatter plots.

    Args:
        dataset_dirs (list): List of directories containing the numpy arrays to be plotted.
        method (str): String naming the method. This is used in the naming convention "method_suffix.npy" to load, plot and save.
        save_path (str): Directory in which to save the plot.
        figsize (tuple, optional): Tuple describing the desired figure size. Defaults to (6, 6).
        suffix (str, optional): Suffix used for naming convention "method_suffix.npy" for loading. Defaults to '_coords.npy'.
        marker_size (int, optional): Desired marker size of the scatter plot. Defaults to 1.
        dpi (int, optional): Desired resolution of the scatter plot. Defaults to 300.

    Returns:
        None
    """

    # make sure the type of dataset_dirs is correct
    assert isinstance(dataset_dirs, list), f"Received {dataset_dirs} for dataset dirs, should be list of directories."

    if len(dataset_dirs) == 0:
        return None

    # get all coordinates and labels
    coords_all = []
    if labels is None:
        labels = []

    for idx, dataset_dir in enumerate(dataset_dirs):
        # check if a file matching the defined suffix exists in this folder
        # - this is a double check when combining with scan_dir_for_coords
        if os.path.isfile(os.path.join(dataset_dir, method + suffix)):
            # load the numpy array mathing the correct suffix and method name
            coords = np.load(os.path.join(dataset_dir, method + suffix))
        elif dataset_dir.endswith(method + suffix):
            coords = np.load(dataset_dir)
        else:
            warnings.warn(f"No coordinate file found for {method} method.")
            break
        print(f"Loading {coords.shape[0]} coordinates for {dataset_dir} dataset.")

        # create label from directory name, if not given as input
        if labels is None:
            temp = dataset_dir.split(os.path.sep)
            label = temp[-3]  # dataset_dir.rsplit('/', 2)[1]  # NOTE: might require manual adjustment if you define new path structure
            labels.append(label)

        coords_all.append(coords)
    assert len(coords_all) == len(labels)

    # scatter plot
    plt.figure(figsize=figsize)
    # iterate through all dataset paths
    for idx, coords in enumerate(coords_all):
        label = labels[idx]
        # add method to scatter plot
        plt.scatter(coords[:, 0], coords[:, 1], label=label, marker=".", s=marker_size)

    # add legend to plot and save under save_path.
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{method}_scatter.png"), dpi=dpi)
    plt.close()

    density_distr(
        coords_all=coords_all,
        method=method,
        figsize=figsize,
        labels=labels,
        save_path=save_path,
    )

    return
