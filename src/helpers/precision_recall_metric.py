"""
Simplified precision / recall computation in embedding space following

Improved Precision and Recall Metric for Assessing Generative Models
Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila

https://arxiv.org/abs/1904.06991
"""

import argparse
import os
import tempfile

import numpy as np
import torch


def batch_pairwise_sq_dists(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise squared Euclidean distances between two batches of vectors.

    Args:
        u (torch.Tensor): A tensor of shape (m, d) representing m vectors of dimension d.
        v (torch.Tensor): A tensor of shape (n, d) representing n vectors of dimension d.

    Returns:
        torch.Tensor: A tensor of shape (m, n) where each element [i, j] is the squared Euclidean distance
                      between u[i] and v[j].

    Notes:
        - Assumes input tensors are of type float32 and on the same device.
        - Uses a numerically stable formulation for distance computation.
    """
    u_norm = (u * u).sum(dim=1, keepdim=True)  # (m, 1)
    v_norm = (v * v).sum(dim=1, keepdim=True).T  # (1, n)
    d2 = u_norm - 2.0 * (u @ v.T) + v_norm  # (m, n)
    return torch.clamp(d2, min=0.0)


@torch.no_grad()
def evaluate(
    features_ref: torch.Tensor,
    features_eval: torch.Tensor,
    k: int = 3,
    row_batch_size: int = 10000,
    col_batch_size: int = 50000,
) -> float:
    """Evaluates the fraction of samples in `features_eval` that fall inside the union of k-nearest neighbor hyperspheres
    (induced by `features_ref`) using squared Euclidean distance.

    Args:
        features_ref (torch.Tensor): Reference feature tensor of shape (M, d), where M is the number of reference points and d is the feature dimension.
        features_eval (torch.Tensor): Evaluation feature tensor of shape (N, d), where N is the number of evaluation points.
        k (int, optional): Number of nearest neighbors to define the hypersphere radius for each reference point. Default is 3.
        row_batch_size (int, optional): Batch size for processing rows (reference/eval points) to manage memory usage. Default is 10,000.
        col_batch_size (int, optional): Batch size for processing columns (reference points) to manage memory usage. Default is 50,000.

    Returns:
        float: Fraction of evaluation samples that fall inside the union of k-NN hyperspheres defined by the reference features.
    """
    device = features_ref.device
    M = features_ref.shape[0]
    N = features_eval.shape[0]

    # 1) Radii^2 for ref points: distance to k-th NN in ref (excluding self)
    radii_sq = torch.empty(M, device=device, dtype=torch.float32)

    # We'll allocate a (row_batch_size, M) distance buffer ONCE on device
    # Caution: can be large.
    distance_row = torch.empty(row_batch_size, M, device=device, dtype=torch.float32)

    for r0 in range(0, M, row_batch_size):
        r1 = min(r0 + row_batch_size, M)
        rows = features_ref[r0:r1]  # (Rb, d)
        # Fill distances to ALL columns in chunks
        for c0 in range(0, M, col_batch_size):
            c1 = min(c0 + col_batch_size, M)
            cols = features_ref[c0:c1]  # (Cb, d)
            d2 = batch_pairwise_sq_dists(rows, cols)  # (Rb, Cb)

            # Mask self when row/col blocks overlap
            if r0 == c0:
                diag = torch.arange(0, min(r1 - r0, c1 - c0), device=device)
                d2[diag, diag] = float("inf")

            distance_row[: (r1 - r0), c0:c1] = d2

        # Take k-th smallest per row
        # We want the distance to the k-th nearest neighbor (self excluded above),
        # so kthvalue with k (1-indexed). Use k to be 3 as in the paper.
        radii_sq[r0:r1] = distance_row[: (r1 - r0)].kthvalue(k, dim=1).values

    # ---- 2) Membership test for eval points: min_i (||x - a_i||^2 - r_i^2) <= 0
    inside = 0
    for e0 in range(0, N, row_batch_size):
        e1 = min(e0 + row_batch_size, N)
        qs = features_eval[e0:e1]  # (Qb, d)

        # Track min margin over reference chunks
        min_margin = torch.full((e1 - e0,), float("inf"), device=device)

        for c0 in range(0, M, col_batch_size):
            c1 = min(c0 + col_batch_size, M)
            cols = features_ref[c0:c1]  # (Cb, d)
            r2 = radii_sq[c0:c1]  # (Cb,)
            d2 = batch_pairwise_sq_dists(qs, cols)  # (Qb, Cb)
            margin = d2 - r2.unsqueeze(0)  # broadcast (Qb, Cb)
            min_margin = torch.minimum(min_margin, margin.min(dim=1).values)

        inside += (min_margin <= 0).sum().item()

    return inside / float(N)


@torch.no_grad()
def compute_precision_recall(
    real_data_embeddings_file: str,
    syn_data_embeddings_file: str,
    k: int = 3,
    row_batch_size: int = 10000,
    col_batch_size: int = 50000,
    equalize_counts: bool = True,
    normalize: bool = False,
    use_cuda_if_available: bool = True,
):
    assert os.path.isfile(real_data_embeddings_file) and os.path.isfile(syn_data_embeddings_file), (
        f"Input files do not exist: {real_data_embeddings_file} or {syn_data_embeddings_file}"
    )

    # Load as float32 torch tensors
    real = torch.from_numpy(np.load(real_data_embeddings_file)).float()
    fake = torch.from_numpy(np.load(syn_data_embeddings_file)).float()

    # (Optional but often helpful) L2-normalize features to stabilize distances
    if normalize:
        real = torch.nn.functional.normalize(real, p=2, dim=1)
        fake = torch.nn.functional.normalize(fake, p=2, dim=1)

    # Equalize counts as in the paper (reporting at same N for both)
    if equalize_counts:
        n = min(real.shape[0], fake.shape[0])
        real = real[:n]
        fake = fake[:n]

    device = torch.device("cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu")
    real = real.to(device)
    fake = fake.to(device)

    # Precision: % of fake inside manifold(real)
    precision = evaluate(real, fake, k=k, row_batch_size=row_batch_size, col_batch_size=col_batch_size)

    # Recall: % of real inside manifold(fake)
    recall = evaluate(fake, real, k=k, row_batch_size=row_batch_size, col_batch_size=col_batch_size)

    return {"precision": float(precision), "recall": float(recall)}


# Small brute force sanity check


def _make_clusters(
    n_clusters=10,
    points_per_cluster=10,
    spacing=2.0,  # distance between neighboring centers
    sigma=0.05,  # intra-cluster std
    seed=123,
):
    """
    Returns (real_points, centers) with shape (n_clusters*points_per_cluster, 2).
    Centers are laid out evenly on the x-axis at multiples of `spacing`.
    """
    rng = np.random.default_rng(seed)
    centers = np.stack([np.arange(n_clusters) * spacing, np.zeros(n_clusters)], axis=1)  # (C, 2)
    pts = []
    for c in centers:
        noise = rng.normal(loc=0.0, scale=sigma, size=(points_per_cluster, 2))
        pts.append(c + noise)
    pts = np.concatenate(pts, axis=0)  # (C*K, 2)
    return pts.astype(np.float32), centers.astype(np.float32)


def _make_fake_from_real(
    centers,
    points_per_cluster=10,
    sigma=0.05,
    misplaced_idx=9,
    shift=8.0,  # large displacement so membership does NOT overlap
    axis=(1, 0),  # shift along x by default; tuple acts as basis vector
    seed=456,
):
    """
    Copies the real centers, then moves one cluster center by `shift` (far away).
    Returns fake points array of shape (C*K, 2).
    """
    rng = np.random.default_rng(seed)
    fake_centers = centers.copy()
    shift_vec = np.array(axis, dtype=np.float32) / np.linalg.norm(axis) * shift
    fake_centers[misplaced_idx] = fake_centers[misplaced_idx] + shift_vec

    pts = []
    for c in fake_centers:
        noise = rng.normal(loc=0.0, scale=sigma, size=(points_per_cluster, 2))
        pts.append(c + noise)
    pts = np.concatenate(pts, axis=0)
    return pts.astype(np.float32)


def _write_test_npy(
    real_path="real.npy",
    fake_path="fake.npy",
    n_clusters=10,
    points_per_cluster=10,
    spacing=2.0,
    sigma=0.05,
    misplaced_idx=9,
    shift=8.0,
    seed_real=123,
    seed_fake=456,
):
    real, centers = _make_clusters(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        spacing=spacing,
        sigma=sigma,
        seed=seed_real,
    )
    fake = _make_fake_from_real(
        centers,
        points_per_cluster=points_per_cluster,
        sigma=sigma,
        misplaced_idx=misplaced_idx,
        shift=shift,
        axis=(1, 0),
        seed=seed_fake,
    )
    np.save(real_path, real)
    np.save(fake_path, fake)
    return real, fake, centers


def _pairwise_sq_dists(A, B):
    # A: (m,2), B: (n,2)
    A2 = (A * A).sum(axis=1, keepdims=True)  # (m,1)
    B2 = (B * B).sum(axis=1, keepdims=True).T  # (1,n)
    D2 = A2 - 2.0 * (A @ B.T) + B2
    return np.maximum(D2, 0.0)


def _knn_radii_sq(X, k=3):
    # distance^2 to k-th nearest neighbor within X (excluding self)
    D2 = _pairwise_sq_dists(X, X)
    # mask diagonal
    np.fill_diagonal(D2, np.inf)
    # kth smallest along rows
    kth = np.partition(D2, kth=k - 1, axis=1)[:, k - 1]  # k is 1-indexed notionally
    return kth  # (N,)


def _fraction_inside(ref, qry, k=3):
    # Returns fraction of qry points within the union of ref hyperspheres
    r2 = _knn_radii_sq(ref, k=k)  # (Nr,)
    D2 = _pairwise_sq_dists(qry, ref)  # (Nq, Nr)
    inside = (D2 <= r2[None, :]).any(axis=1)
    return inside.mean()


def _smoke_test(real, fake, k=3):
    prec = _fraction_inside(real, fake, k=k)  # fake inside real-manifold
    rec = _fraction_inside(fake, real, k=k)  # real inside fake-manifold
    return float(prec), float(rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate precision and recall metrics for generative models. Run without args to test.")
    parser.add_argument("--real", type=str, help="Path to real data embeddings (.npy file)")
    parser.add_argument("--fake", type=str, help="Path to synthetic data embeddings (.npy file)")
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest neighbors to consider (default: 3)",
    )
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=2048,
        help="Batch size for processing rows (default: 2048)",
    )
    parser.add_argument(
        "--col-batch-size",
        type=int,
        default=4096,
        help="Batch size for processing columns (default: 4096)",
    )
    parser.add_argument(
        "--equalize-counts",
        action="store_true",
        default=True,
        help="Equalize real and fake sample counts (default: True)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="L2-normalize feature vectors (default: False)",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")

    # Parameters for synthetic test data generation
    parser.add_argument(
        "--test-n-clusters",
        type=int,
        default=10,
        help="Number of clusters for test data (default: 10)",
    )
    parser.add_argument(
        "--test-points-per-cluster",
        type=int,
        default=1000,
        help="Points per cluster for test data (default: 1000)",
    )
    parser.add_argument(
        "--test-spacing",
        type=float,
        default=2.0,
        help="Spacing between cluster centers (default: 2.0)",
    )
    parser.add_argument(
        "--test-sigma",
        type=float,
        default=0.1,
        help="Standard deviation for intra-cluster points (default: 0.1)",
    )
    parser.add_argument(
        "--test-shift",
        type=float,
        default=100.0,
        help="Shift distance for displaced cluster (default: 100.0)",
    )
    parser.add_argument(
        "--test-misplaced-idx",
        type=int,
        default=9,
        help="Index of cluster to displace (default: 9)",
    )

    args = parser.parse_args()

    # If real and fake paths are not provided, run the smoke test with synthetic data
    if args.real is None or args.fake is None:
        print("No input files specified, running synthetic data smoke test...")

        # Create temporary files for the smoke test
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as real_temp:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fake_temp:
                real_path = real_temp.name
                fake_path = fake_temp.name

                # Use provided paths if specified
                if args.real is not None:
                    real_path = args.real
                if args.fake is not None:
                    fake_path = args.fake

                try:
                    # Generate synthetic test data
                    real, fake, centers = _write_test_npy(
                        real_path=real_path,
                        fake_path=fake_path,
                        n_clusters=args.test_n_clusters,
                        points_per_cluster=args.test_points_per_cluster,
                        spacing=args.test_spacing,
                        sigma=args.test_sigma,
                        misplaced_idx=args.test_misplaced_idx,
                        shift=args.test_shift,
                    )
                    print(f"Wrote temporary test files: {real_path} and {fake_path}")

                    # Quick PR sanity check (no GPU, tiny N)
                    p, r = _smoke_test(real, fake, k=args.k)
                    print(f"Smoke test (k={args.k}) → precision ≈ {p:.3f}, recall ≈ {r:.3f}")

                    # Run full precision-recall computation on the generated data
                    res = compute_precision_recall(
                        real_data_embeddings_file=real_path,
                        syn_data_embeddings_file=fake_path,
                        k=args.k,
                        row_batch_size=args.row_batch_size,
                        col_batch_size=args.col_batch_size,
                        equalize_counts=args.equalize_counts,
                        normalize=args.normalize,
                        use_cuda_if_available=not args.no_cuda,
                    )
                    expected_p = str(round(res["precision"], 1))
                    expected_r = str(round(res["recall"], 1))

                    # Check if precision and recall match when rounded to one decimal place
                    actual_p = str(round(p, 1))
                    actual_r = str(round(r, 1))
                    if actual_p == expected_p and actual_r == expected_r:
                        print(f"SUCCESS: Precision and recall both round to {actual_p} / {actual_r}")
                    else:
                        print(f"ERROR: Precision rounds to {actual_p} != {expected_p}, recall to {actual_r} != {expected_r}")

                finally:
                    # Clean up temporary files unless they were provided by the user
                    if args.real is None and os.path.exists(real_path):
                        os.unlink(real_path)
                        print(f"Removed temporary file: {real_path}")
                    if args.fake is None and os.path.exists(fake_path):
                        os.unlink(fake_path)
                        print(f"Removed temporary file: {fake_path}")

    else:
        # Run the full precision-recall computation with user-provided file paths
        res = compute_precision_recall(
            real_data_embeddings_file=args.real,
            syn_data_embeddings_file=args.fake,
            k=args.k,
            row_batch_size=args.row_batch_size,
            col_batch_size=args.col_batch_size,
            equalize_counts=args.equalize_counts,
            normalize=args.normalize,
            use_cuda_if_available=not args.no_cuda,
        )
        print(f"Results: {res}")
