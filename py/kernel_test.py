import numpy as np
import pandas as pd
from HMatrix import (
    HMatrix,
    measure_compression,
    next_power_of_two,
    pad_matrix,
    crop_matrix,
    reconstruct_dense,
)


def exponential_kernel(x, y, lambd=1.0):
    return np.exp(-lambd * abs(x - y))


def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-((x - y) ** 2) / (2 * sigma**2))


def laplace_1d(x, y):
    r = abs(x - y)
    return -np.log(r) if r > 1e-10 else 0


def kernel_matrix(points, kernel):
    n = len(points)
    K = np.array([[kernel(points[i], points[j]) for j in range(n)] for i in range(n)])

    return K


if __name__ == "__main__":
    n = 2000
    min_size = 32
    tol = 1e-4
    max_rank = 10

    points_1d = np.linspace(0, 1, n).reshape(-1, 1)

    K = kernel_matrix(points_1d, lambda x, y: laplace_1d(x[0], y[0]))
    # K = kernel_matrix(points_1d, lambda x, y: gaussian_kernel(x[0], y[0]))

    n_pad = next_power_of_two(n)
    K_padded = pad_matrix(K, (n_pad, n_pad))

    hK = HMatrix(
        K_padded,
        max_rank,
        min_size,
        tol,
        row_points=points_1d,
        col_points=points_1d,
        adaptive=True,
    )

    compression_results = measure_compression(hK)

    K_recon = reconstruct_dense(hK.root, hK.root.shape)

    K_recon = crop_matrix(K_recon, (n, n))

    error = np.linalg.norm(K - K_recon, "fro") / np.linalg.norm(K, "fro")

    [print(f"{k}: {v}") for k, v in compression_results.items()]
