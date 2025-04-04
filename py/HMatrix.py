import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.spatial import distance


class MatrixNode:
    """
    Matrix node blocks that contain the various children of a given block
    Defined parameters of data ranges and indicator of whether the block is a leaf
    """

    # Lower and Upper bound
    def __init__(
        self,
        row_range: tuple[int, int],
        col_range: tuple[int, int],
        data=None,
        is_leaf: bool = False,
        is_low_rank: bool = False,
    ):
        self.row_range = row_range
        self.col_range = col_range
        self.data = None
        self.children = []
        self.is_leaf = is_leaf  # Indicator of leaf node

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.row_range[1] - self.row_range[0],
            self.col_range[1] - self.col_range[0],
        )


def next_power_of_two(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    return 1 << (x - 1).bit_length()


def pad_matrix(M: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Pad matrix M with zeros to match target_shape."""
    m, n = M.shape
    target_m, target_n = target_shape
    padded = np.zeros((target_m, target_n), dtype=M.dtype)
    padded[:m, :n] = M
    return padded


def crop_matrix(M: np.array, orig_shape: tuple) -> np.array:
    """Crop matrix M to the original shape."""
    m, n = orig_shape
    return M[:m, :n]


def is_compressable(
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    row_points: np.array,
    col_points: np.array,
    distance_criterion: float,
):
    """Function to determine approximatability of a matrix

    Utilizes a distance criterion for the rows and columns in a matrix
    and returns whether we can approximate it


    Args:
        row_range tuple[int,int]: The start and end of rows
        col_range tuple[int,int]: The start and end of cols
        row_points np.array: Points contained within row range
        col_points np.array: Points contained within col range
        distance_criterion float: approximation criteria
    """
    if row_points is None or col_points is None:
        return False

    row_subset = row_points[row_range[0] : row_range[1]]
    col_subset = col_points[col_range[0] : col_range[1]]

    if len(row_subset) == 0 or len(col_subset) == 0:
        return False

    row_center = np.mean(row_subset, axis=0)
    col_center = np.mean(col_subset, axis=0)

    row_diam = np.max(distance.pdist(row_subset)) if len(row_subset) > 1 else 0
    col_diam = np.max(distance.pdist(col_subset)) if len(col_subset) > 1 else 0

    center_dist = np.linalg.norm(row_center - col_center)

    return center_dist > distance_criterion * max(row_diam, col_diam)


def compress_block(M: np.array, tol: float, max_rank: int):
    """
    Compresses Matrix into SVD form if approximation is beneficial,
    otherwise returns the dense block.

    Args:
        M: Input Matrix block.
        tol: Tolerance for singular values to determine rank.
        max_rank: Maximum allowable rank for low-rank approximation.

    Returns:
        Either a tuple (U_compressed, Vh_compressed) for low-rank blocks
        or the original dense block M if compression is not performed.
    """
    rows, cols = M.shape

    # If the block is too small and dense is more efficient
    if rows <= 2 or cols <= 2:
        return M.copy()

    try:
        # Perform SVD
        U, s, Vh = svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        # Handle cases where SVD might fail (e.g., all zeros, NaNs, Infs)
        print(f"  WARN: SVD failed for block shape {M.shape}. Storing dense.")
        return M.copy()
    except Exception as e:
        print(
            f"  WARN: Unexpected error during SVD for block shape {M.shape}: {e}. Storing dense."
        )
        return M.copy()

    frob_norm = np.linalg.norm(M, ord="fro")

    # Zero Matrices
    if frob_norm <= 1e-14:
        return np.zeros_like(M)

    # Get the energy (Squared Frobenisu Norm)
    energy = np.cumsum(s**2) / np.sum(s**2)
    rank = np.searchsorted(energy, 1.0 - tol) + 1

    rank = min(rank, max_rank)
    # Check if the matrix is effectively zero rank based on tolerance
    if rank == 0:
        return np.zeros_like(M)

    # Calculate storage costs (number of elements)
    low_rank_cost = rank * (rows + cols)
    dense_cost = rows * cols

    # Check if low-rank approximation is valid and efficient
    if rank <= max_rank and low_rank_cost <= dense_cost:
        U_compressed = np.ascontiguousarray(U[:, :rank] * s[:rank].reshape(1, -1))
        Vh_compressed = np.ascontiguousarray(Vh[:rank, :])
        return (U_compressed, Vh_compressed)
    else:
        # print(f"  Block {M.shape}: Rank={rank}. Storing dense (LR cost {low_rank_cost} > Dense cost {dense_cost} or rank > max_rank)") # Optional debug
        return M.copy()


def get_dense_from_node(node: MatrixNode):
    """
    Reconstruct the dense representation of the matrix
    """

    # if the node is already a leaf, return its data directly
    if node.is_leaf:
        if isinstance(node.data, tuple):
            U, V = node.data
            return U @ V
        else:
            return node.data
    else:
        # Recursively get to the leaf node of each child and reconstruct dense

        r_start, r_end = node.row_range
        c_start, c_end = node.col_range

        dense = np.zeros((node.shape[0], node.shape[1]), dtype=np.float64)

        for child in node.children:
            rs, re = child.row_range
            cs, ce = child.col_range
            dense[rs - r_start : re - r_start, cs - c_start : ce - c_start] = (
                get_dense_from_node(child)
            )

        return dense


def add_nodes(node1: MatrixNode, node2: MatrixNode, tol=1e-6, max_rank=10):
    """Add two MatrixNodes

    Args:
        node1: First Block
        node2: Second Block
        tol float: Tolerance for rank determination
        max_rank int: Max rank to determine whether to approximate or not
    """

    # Get denes nodes
    A_dense = get_dense_from_node(node1)
    B_dense = get_dense_from_node(node2)

    # Sum dense blocks
    sum_dense = A_dense + B_dense
    # Create new node to store compressed dense sum
    new_node = MatrixNode(node1.row_range, node1.col_range, is_leaf=True)
    new_node.data = compress_block(sum_dense, tol, max_rank)
    return new_node


def construct_tree(
    matrix: np.array,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    min_size: int,
    tol: float,
    max_rank: int,
    row_points=None,
    col_points=None,
    adaptive=False,
) -> MatrixNode:
    """Function to perform the compression pass of the HMatrix

    Args:
        matrix: Input matrix
        row_range: Tuple of row start and end
        col_range: Tuple of col start and end
        min_size: Minimum size to decide compression/making of leaf node
        tol: Tolerance for s matrix
        max_rank: Maximum rank to determine compression
        row_points: row points to check compressability
        col_points: col points to check compressability
        adaptive: Whether to run adaptive compression with row and col points

    Returns:
        Root of the newly created Matrix Node
    """

    node = MatrixNode(row_range, col_range)
    rows = row_range[1] - row_range[0]
    cols = col_range[1] - col_range[0]

    if rows <= min_size or cols <= min_size:
        node.is_leaf = True
        sub_matrix = matrix[row_range[0] : row_range[1], col_range[0] : col_range[1]]
        node.data = compress_block(sub_matrix, tol, max_rank)
        if isinstance(node.data, tuple):
            node.is_low_rank = True
        return node

    if adaptive and row_points is not None and col_points is not None:
        if is_compressable(
            row_range, col_range, row_points, col_points, distance_criterion=0.5
        ):
            node.is_leaf = True
            sub_matrix = matrix[
                row_range[0] : row_range[1],
                col_range[0] : col_range[1],
            ]
            node.data = compress_block(sub_matrix, tol, max_rank)
            if isinstance(node.data, tuple):
                node.is_low_rank = True
            return node

    # Divide into 4 quadrants

    r_mid = (row_range[0] + row_range[1]) // 2
    c_mid = (col_range[0] + col_range[1]) // 2

    top_left = construct_tree(
        matrix,
        (row_range[0], r_mid),
        (col_range[0], c_mid),
        min_size,
        tol,
        max_rank,
        row_points,
        col_points,
        adaptive,
    )
    top_right = construct_tree(
        matrix,
        (row_range[0], r_mid),
        (c_mid, col_range[1]),
        min_size,
        tol,
        max_rank,
        row_points,
        col_points,
        adaptive,
    )
    bottom_left = construct_tree(
        matrix,
        (r_mid, row_range[1]),
        (col_range[0], c_mid),
        min_size,
        tol,
        max_rank,
        row_points,
        col_points,
        adaptive,
    )
    bottom_right = construct_tree(
        matrix,
        (r_mid, row_range[1]),
        (c_mid, col_range[1]),
        min_size,
        tol,
        max_rank,
        row_points,
        col_points,
        adaptive,
    )

    node.children = [top_left, top_right, bottom_left, bottom_right]
    return node


class HMatrix:
    """
    Class to convert a dense matrix into it's hierarchical structure (Analogous to the interval divisions for sources and charges)
    Rather than using a binary tree as in the field case, Matrix Multiplication requires a QuadTree structure
    We will use this property to subdivide the matrix into blocks
    """

    def __init__(
        self,
        matrix: np.array,
        max_rank=5,
        min_size=16,
        tol=1e-6,
        row_points=None,
        col_points=None,
        adaptive=True,
    ):
        self.matrix = matrix
        self.nrows, self.ncols = matrix.shape
        self.max_rank = max_rank
        self.min_size = min_size
        self.tol = tol
        self.row_points = row_points
        self.col_points = col_points
        self.adaptive = adaptive

        # Construct tree from root matrix
        self.root = construct_tree(
            matrix,
            (0, self.nrows),
            (0, self.ncols),
            min_size,
            tol,
            max_rank,
            row_points,
            col_points,
            adaptive,
        )


def HMultiply(
    A: MatrixNode,
    B: MatrixNode,
    tol: float,
    max_rank: int,
    min_size: int,
) -> MatrixNode:
    """[TODO:summary]

    [TODO:description]

    Args:
        A: First Matrix Block
        B: Second Matrix Block
        tol: tolerance for s matrix
        max_rank: rank to determine low rank approximation
        min_size: min size to determine whether to sub divide

    Returns:
        Returns the product of two Matrix Nodes
    """

    # Leaves are dense multiplied
    if A.is_leaf or B.is_leaf:
        A_dense = get_dense_from_node(A)
        B_dense = get_dense_from_node(B)
        dense_product = A_dense @ B_dense
        new_node = MatrixNode(A.row_range, B.col_range, is_leaf=True)
        new_node.data = compress_block(dense_product, tol, max_rank)
        return new_node

    A11, A12, A21, A22 = A.children
    B11, B12, B21, B22 = B.children

    # Add and Multiply each of the 4 children

    C11 = add_nodes(
        HMultiply(A11, B11, tol, max_rank, min_size),
        HMultiply(A12, B21, tol, max_rank, min_size),
        tol,
        max_rank,
    )

    C12 = add_nodes(
        HMultiply(A11, B12, tol, max_rank, min_size),
        HMultiply(A12, B22, tol, max_rank, min_size),
        tol,
        max_rank,
    )

    C21 = add_nodes(
        HMultiply(A21, B11, tol, max_rank, min_size),
        HMultiply(A22, B21, tol, max_rank, min_size),
        tol,
        max_rank,
    )

    C22 = add_nodes(
        HMultiply(A21, B12, tol, max_rank, min_size),
        HMultiply(A22, B22, tol, max_rank, min_size),
        tol,
        max_rank,
    )

    new_node = MatrixNode(A.row_range, B.col_range, is_leaf=False)
    new_node.children = [C11, C12, C21, C22]
    return new_node


def reconstruct_dense(node: MatrixNode, shape: tuple) -> np.array:
    """Reconstruct the dense from a given HMatrix

    Args:
        node: Matrix Block to reconstruct dense
        shape: Shape property

    Returns:
        Returns the matrix
    """

    M = np.zeros(shape, dtype=np.float64)

    def fill(node, M):
        rs, re = node.row_range
        cs, ce = node.col_range
        if node.is_leaf:
            if isinstance(node.data, tuple):
                U, V = node.data
                M[rs:re, cs:ce] = U @ V
            else:
                M[rs:re, cs:ce] = node.data
        else:
            for child in node.children:
                fill(child, M)

    fill(node, M)
    return M


def HMult_dense(A: MatrixNode, B: np.array) -> np.ndarray:
    col_start, col_end = A.col_range
    sub_matrix_B = B[col_start:col_end, :]

    if A.is_leaf:
        if isinstance(A.data, tuple):
            U, V = A.data
            return U @ (V @ sub_matrix_B)
        return A.data @ sub_matrix_B

    else:
        # recursively multiply
        top_left = HMult_dense(A.children[0], B)
        top_right = HMult_dense(A.children[1], B)
        top_result = top_left + top_right

        bottom_left = HMult_dense(A.children[2], B)
        bottom_right = HMult_dense(A.children[3], B)
        bottom_result = bottom_left + bottom_right

        return np.vstack([top_result, bottom_result])


def count_stored_elements(node: MatrixNode) -> int:
    """
    Recursively counts the total number of elements stored in the HMatrix tree.
    This represents the memory footprint relative to a dense matrix.

    Args:
        node: Current MatrixNode.

    Returns:
        Total number of numerical elements stored in the subtree rooted at node.
    """
    if node.is_leaf:
        if isinstance(node.data, tuple):
            # Low-rank storage: U and V factors
            U, V = node.data
            # Return total elements stored in U and V
            return U.size + V.size
        elif node.data is not None:
            # Dense block storage
            # Return total elements stored in the dense block
            return node.data.size
        else:
            # Handle cases where a leaf might have None data (shouldn't normally happen)
            return 0
    else:  # Internal node
        # Recursively sum elements stored in children
        return sum(count_stored_elements(child) for child in node.children)


def measure_compression(hmatrix: HMatrix) -> dict:
    original_size_bytes = hmatrix.nrows * hmatrix.ncols * hmatrix.matrix.dtype.itemsize
    compressed_elements = count_stored_elements(hmatrix.root)

    # Calculate compressed size in BYTES
    compressed_size_bytes = compressed_elements * hmatrix.matrix.dtype.itemsize

    compression_ratio = (
        original_size_bytes / compressed_size_bytes
        if compressed_size_bytes > 0
        else float("inf")
    )

    return {
        "original_size_bytes": original_size_bytes,  # Key clarifies units
        "compressed_size_bytes": compressed_size_bytes,  # Key clarifies units
        "compression_ratio": compression_ratio,  # Now should be 1.0
        "memory_saved_pct": (1 - compressed_size_bytes / original_size_bytes) * 100
        if original_size_bytes > 0
        else 0,
    }


if __name__ == "__main__":
    # A (m x n)
    # B (n x p)
    m, n, p = 10000, 10000, 1
    min_size = 16
    tol = 1e-4
    max_rank = 16

    # Create random matrices A and B.
    A_orig = np.random.rand(m, n)
    B_orig = np.random.rand(n, p)

    # Compute padded dimensions (next power of 2 for each).
    m_pad = next_power_of_two(m)
    n_pad = next_power_of_two(n)
    p_pad = next_power_of_two(p)

    # Pad A and B.
    A_padded = pad_matrix(A_orig, (m_pad, n_pad))
    B_padded = pad_matrix(B_orig, (n_pad, p_pad))

    # -------------------- Compression Pass --------------------
    # Build hierarchical matrix A
    hA = HMatrix(A_padded, max_rank, min_size, tol)
    print(measure_compression(hA))

    # -------------------- Evolution Pass --------------------
    # Compute dense product
    res = HMult_dense(hA.root, B_padded)
    res = crop_matrix(res, (m, p))

    # Reconstruction Error
    reconstructed = reconstruct_dense(hA.root, hA.root.shape)
    reconstructed = crop_matrix(reconstructed, (m, n))
    print(
        "Reconstruction Error",
        np.linalg.norm(A_orig - reconstructed) / np.linalg.norm(reconstructed),
    )

    # Regular Mat Mul
    direct = A_orig @ B_orig

    # Compute the normalized error
    error = np.linalg.norm(res - direct) / np.linalg.norm(direct)
    print("Original dimensions: A:", A_orig.shape, "B:", B_orig.shape)
    print("Relative error between hierarchical and direct multiplication:", error)
