import numpy as np
import pandas as pd
import sys
from scipy.linalg import svd
import time
import os

from HMatrix import (
    MatrixNode,
    next_power_of_two,
    pad_matrix,
    crop_matrix,
    compress_block,
    get_dense_from_node,
    add_nodes,
    construct_tree,
    HMatrix,
    HMultiply,
    reconstruct_dense,
    HMult_dense,
    count_stored_elements,
    measure_compression,
)

print("Successfully imported HMatrix components.")


def measure_hmatrix_compression(hmatrix: HMatrix, original_matrix: np.ndarray) -> dict:
    original_elements = original_matrix.size
    original_size_bytes = original_elements * original_matrix.dtype.itemsize
    compressed_elements = count_stored_elements(hmatrix.root)
    compressed_size_bytes = compressed_elements * original_matrix.dtype.itemsize
    compression_ratio = (
        original_size_bytes / compressed_size_bytes
        if compressed_size_bytes > 0
        else float("inf")
    )
    return {
        "hmatrix_stored_elements": compressed_elements,
        "hmatrix_compressed_size_bytes": compressed_size_bytes,
        "compression_ratio": compression_ratio,
    }


# --- Function to run a single HMatrix test configuration ---
def run_hmatrix_test(
    A_orig, A_padded, B_orig, B_padded, C_direct, max_rank, min_size, tol
):
    """Runs HMatrix construction and tests for a given parameter set."""
    test_results = {
        "max_rank": max_rank,
        "min_size": min_size,
        "tol": tol,
        "construction_time_sec": None,
        "hmatrix_stored_elements": None,
        "hmatrix_compressed_size_bytes": None,
        "compression_ratio": None,
        "reconstruction_time_sec": None,
        "reconstruction_relative_error": None,
        "hmatrix_mult_time_sec": None,
        "hmatrix_mult_relative_error": None,
        "status": "OK",  # To track errors for specific configs
    }
    m_orig, n_orig = A_orig.shape
    n_pad, p_pad = B_padded.shape  # p_pad is TEST_VECTOR_COLS

    try:
        # Construct H Matrix
        start_time = time.time()
        hA = HMatrix(matrix=A_padded, max_rank=max_rank, min_size=min_size, tol=tol)
        test_results["construction_time_sec"] = time.time() - start_time

        # Measure Compression
        compression_results = measure_hmatrix_compression(hA, A_orig)
        test_results.update(compression_results)

        # Reconstruction Accuracy
        start_time = time.time()
        A_reconstructed_padded = reconstruct_dense(hA.root, A_padded.shape)
        A_reconstructed = crop_matrix(A_reconstructed_padded, A_orig.shape)
        test_results["reconstruction_time_sec"] = time.time() - start_time
        # Use np.linalg.norm(A_orig) pre-calculated if A_orig is huge? Maybe not needed.
        norm_A_orig = np.linalg.norm(A_orig)
        test_results["reconstruction_relative_error"] = (
            np.linalg.norm(A_orig - A_reconstructed) / norm_A_orig
            if norm_A_orig > 0
            else 0
        )

        # Matrix-Vector Accuracy
        start_time_h = time.time()
        C_hmatrix_padded = HMult_dense(hA.root, B_padded)
        test_results["hmatrix_mult_time_sec"] = time.time() - start_time_h
        C_hmatrix = crop_matrix(
            C_hmatrix_padded, C_direct.shape
        )  # Crop to original C shape

        norm_C_direct = np.linalg.norm(C_direct)
        test_results["hmatrix_mult_relative_error"] = (
            np.linalg.norm(C_direct - C_hmatrix) / norm_C_direct
            if norm_C_direct > 0
            else 0
        )

    except np.linalg.LinAlgError as e:
        print(
            f"  WARN: SVD Error for config (rank={max_rank}, size={min_size}, tol={tol}): {e}"
        )
        test_results["status"] = f"SVD Error: {e}"
    except MemoryError as e:
        print(
            f"  WARN: Memory Error for config (rank={max_rank}, size={min_size}, tol={tol}): {e}"
        )
        test_results["status"] = f"Memory Error: {e}"
    except Exception as e:
        print(
            f"  ERROR: Unexpected error for config (rank={max_rank}, size={min_size}, tol={tol}): {e}"
        )
        test_results["status"] = f"Error: {e}"
        # Depending on severity, you might want to re-raise or sys.exit here

    return test_results


if __name__ == "__main__":
    # Configuration
    # WEIGHT_FILE = "weights/layers.5.feed_forward.w1.weight.tsv"
    WEIGHT_FILE = (
        "weights/l6_l9/layers.6.feed_forward.w1.weight.tsv"  # Store WEIGHT_FILE
    )

    # Test Vector Dimension
    TEST_VECTOR_COLS = 1

    # Define Search Space
    param_grid = {
        # "max_rank": [8, 16, 32, 64, 128, 256, 512],
        # "min_size": [8, 16, 32, 64, 128, 256, 512],
        # "tol": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        "max_rank": [1, 2, 4, 8],
        "min_size": [32, 64, 128, 256],
        "tol": [128],
    }

    # --- 1. Load Data (Once) ---
    print(f"Loading weights from: {WEIGHT_FILE}")
    if not os.path.exists(WEIGHT_FILE):
        print(f"Error: Weight file not found at '{WEIGHT_FILE}'")
        sys.exit(1)
    try:
        df = pd.read_table(WEIGHT_FILE, header=None)
        A_orig = df.to_numpy().astype(np.float64)
        print(f"Successfully loaded matrix A_orig with shape: {A_orig.shape}")
    except Exception as e:
        print(f"Error loading or processing weight file: {e}")
        sys.exit(1)

    m, n = A_orig.shape

    # Preprocessing
    m_pad = next_power_of_two(m)
    n_pad = next_power_of_two(n)
    print(f"Padding original matrix from {A_orig.shape} to ({m_pad}, {n_pad})")
    A_padded = pad_matrix(A_orig, (m_pad, n_pad))

    # Prepare Test Vector
    print(f"Preparing test matrix B with shape ({n}, {TEST_VECTOR_COLS})...")
    B_orig = np.random.rand(n, TEST_VECTOR_COLS).astype(np.float64)
    B_padded = pad_matrix(B_orig, (n_pad, TEST_VECTOR_COLS))

    # Calculate Direct Matmul
    print("Calculating direct multiplication result (A @ B)...")
    start_time_d = time.time()
    C_direct = A_orig @ B_orig
    time_direct_mult = time.time() - start_time_d
    print(f"Direct multiplication finished in {time_direct_mult:.4f} seconds.")

    # Perform grid search
    all_results = []
    total_configs = (
        len(param_grid["max_rank"])
        * len(param_grid["min_size"])
        * len(param_grid["tol"])
    )
    print(f"\n--- Starting Grid Search ({total_configs} configurations) ---")
    config_count = 0

    for tol in param_grid["tol"]:
        for min_size in param_grid["min_size"]:
            for max_rank in param_grid["max_rank"]:
                config_count += 1
                print(
                    f"\n[{config_count}/{total_configs}] Testing config: max_rank={max_rank}, min_size={min_size}, tol={tol:.1e}"
                )

                # run test for config
                results = run_hmatrix_test(
                    A_orig,
                    A_padded,
                    B_orig,
                    B_padded,
                    C_direct,
                    max_rank,
                    min_size,
                    tol,
                )
                all_results.append(results)

    print("\n--- Grid Search Finished ---")

    # --- Analyze and Display Results ---
    if not all_results:
        print("No results collected.")
    else:
        results_df = pd.DataFrame(all_results)

        # Define columns to display and their formatting
        pd.set_option("display.max_rows", 100)  # Show more rows if needed
        pd.set_option("display.width", 1000)  # Adjust terminal width
        display_cols = [
            "max_rank",
            "min_size",
            "tol",
            "compression_ratio",
            "reconstruction_relative_error",
            "hmatrix_mult_relative_error",
            "hmatrix_mult_time_sec",
            "construction_time_sec",
            "status",
        ]

        # Sort results
        results_df_sorted = results_df.sort_values(
            by=["compression_ratio", "hmatrix_mult_relative_error"],
            ascending=[False, True],
        )

        print("\n--- Grid Search Results Summary ---")
        print(f"(Direct multiplication time: {time_direct_mult:.4f} seconds)")

        # Format Table and Columns
        print(
            results_df_sorted[display_cols].to_string(
                index=False,
                formatters={
                    "tol": "{:.1e}".format,
                    "compression_ratio": "{:.3f}x".format,
                    "reconstruction_relative_error": "{:.3e}".format,
                    "hmatrix_mult_relative_error": "{:.3e}".format,
                    "hmatrix_mult_time_sec": "{:.4f}".format,
                    "construction_time_sec": "{:.4f}".format,
                },
            )
        )

        results_df_sorted.to_csv(
            f"hmatrix_grid_search_results_{WEIGHT_FILE.split('/')[-1]}.csv",
            index=False,
        )
        print(f"\nResults saved to hmatrix_grid_search_results_{WEIGHT_FILE}.csv")
