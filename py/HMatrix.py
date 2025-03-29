import numpy as np
import sys
from scipy.linalg import svd
from scipy.sparse import random
import time


class MatrixNode:
    """
    Matrix node blocks that contain the various children of a given block
    Defined parameters of data ranges and indicator of whether the block is a leaf
    """

    # Lower and Upper bound
    def __init__(
        self,
        row_range: tuple,
        col_range: tuple,
        data=None,
        is_leaf: bool = False,
    ):

        self.row_range = row_range
        self.col_range = col_range
        self.data = None
        self.children = []
        self.is_leaf = is_leaf  # Indicator of leaf node
        self.multipole = None
        self.local = None

    @property
    def shape(self):
        return (
            self.row_range[1] - self.row_range[0],
            self.col_range[1] - self.col_range[0],
        )


def next_power_of_two(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    return 1 << (x - 1).bit_length()


def pad_matrix(M: np.array, target_shape: tuple) -> np.array:
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


def compress_block(M: np.array, tol: float, max_rank: int):
    """Compresses Matrix into SVD form for sparse matrix

    Args:
        M: Input Matrix
        tol: Rank sum tolerance
        max_rank: Rank to determine if matrix can be approximated
    """

    # Perform SVD
    U, s, Vh = svd(M, full_matrices=False)

    # Rank determined by column sums of s vector
    rank = np.sum(s > tol)
    # Compress the matrix
    if rank <= max_rank:
        return (U[:, :rank] * s[:rank], Vh[:rank, :])
    else:
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
    row_range: tuple,
    col_range: tuple,
    min_size: int,
    tol: float,
    max_rank: int,
) -> MatrixNode:
    """Function to perform the compression pass of the HMatrix

    Args:
        matrix: Input matrix
        row_range: Tuple of row start and end
        col_range: Tuple of col start and end
        min_size: Minimum size to decide compression/making of leaf node
        tol: Tolerance for s matrix
        max_rank: Maximum rank to determine whether the

    Returns:
        Root of the newly created Matrix Node
    """

    node = MatrixNode(row_range, col_range)
    rows = row_range[1] - row_range[0]
    cols = col_range[1] - col_range[0]

    sub_matrix = matrix[row_range[0] : row_range[1], col_range[0] : col_range[1]]

    if rows <= min_size or cols <= min_size:
        node.is_leaf = True
        node.data = compress_block(sub_matrix, tol, max_rank)
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
    )
    top_right = construct_tree(
        matrix,
        (row_range[0], r_mid),
        (c_mid, col_range[1]),
        min_size,
        tol,
        max_rank,
    )
    bottom_left = construct_tree(
        matrix,
        (r_mid, row_range[1]),
        (col_range[0], c_mid),
        min_size,
        tol,
        max_rank,
    )
    bottom_right = construct_tree(
        matrix,
        (r_mid, row_range[1]),
        (c_mid, col_range[1]),
        min_size,
        tol,
        max_rank,
    )

    node.children = [top_left, top_right, bottom_left, bottom_right]
    return node


class HMatrix:
    """
    Class to convert a dense matrix into it's hierarchical structure (Analogous to the interval divisions for sources and charges)
    Rather than using a binary tree as in the field case, Matrix Multiplication requires a QuadTree structure
    We will use this property to subdivide the matrix into blocks
    """

    def __init__(self, matrix: np.array, max_rank=5, min_size=16, tol=1e-6):
        self.matrix = matrix
        self.nrows, self.ncols = matrix.shape
        self.root = MatrixNode(0, self.nrows, 0, self.ncols)
        self.max_rank = max_rank
        self.min_size = min_size
        self.tol = tol

        # Construct tree from root matrix
        self.root = construct_tree(
            matrix, (0, self.nrows), (0, self.ncols), min_size, tol, max_rank
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


def HMult_dense(A: MatrixNode, B: np.array):

    col_start, col_end = A.col_range
    sub_matrix_B = B[col_start:col_end, :]

    if A.is_leaf:

        if isinstance(A.data, tuple):

            U, V = A.data
            return U @ (V @ sub_matrix_B)
        else:
            return A.data @ sub_matrix_B

    else:

        # recurisvely multiply
        top_left = HMult_dense(A.children[0], B)
        top_right = HMult_dense(A.children[1], B)
        top_result = top_left + top_right

        bottom_left = HMult_dense(A.children[2], B)
        bottom_right = HMult_dense(A.children[3], B)
        bottom_result = bottom_left + bottom_right

        return np.vstack([top_result, bottom_result])


def count_nonzeros(node: MatrixNode) -> int:
    """Function to recurisvely count zeros
    Args:
        node: Current Block

    Returns:
        Returns the number of 0 elements in a block
    """
    if node.is_leaf:
        if isinstance(node.data, tuple):
            U, V = node.data
            return np.count_nonzero(U) + np.count_nonzero(V)
        else:
            return np.count_nonzero(node.data)

    else:
        return sum(count_nonzeros(child) for child in node.children)


def measure_compression(hmatrix: HMatrix) -> dict:
    """Function to determine compression of an hmatrix

    Recursively calls count_nonzero function on the tree

    Args:
        hmatrix: The constructed HMatrix

    Returns:
        Dictionary containing the original size, compressed size and ratio
    """

    original_size = (
        hmatrix.nrows * hmatrix.ncols * hmatrix.matrix.dtype.itemsize
    )  # assuming each occupies 8 bytes for float64
    compressed_size = count_nonzeros(hmatrix.root)

    compression_ratio = (
        original_size / compressed_size if compressed_size > 0 else float("inf")
    )

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
    }


if __name__ == "__main__":
    # A (m x n)
    # B (n x p)
    m, n, p = 10000, 10000, 1
    min_size = 512
    tol = 1e-6
    max_rank = 2
    sparse_mat = random(n, p, density=0.01, format="csr", dtype=np.float64).toarray()

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
    sparse_padded = pad_matrix(sparse_mat, (n_pad, p_pad))

    # -------------------- Compression Pass --------------------
    # Build hierarchical matrices from the padded versions.
    hA = HMatrix(A_padded, max_rank, min_size, tol)

    # print(measure_compression(hA))
    # hB = HMatrix(B_padded, max_rank, min_size, tol)

    res = HMult_dense(hA.root, B_padded)

    sparse_mult = HMult_dense(hA.root, sparse_padded)
    res = crop_matrix(res, (m, p))

    sparse_mult = crop_matrix(sparse_mult, (n, p))

    direct_sparse = A_orig @ sparse_mat

    error_sparse = np.linalg.norm(sparse_mult - direct_sparse) / np.linalg.norm(
        direct_sparse
    )
    print("Sparse Error", error_sparse)

    # print(res)

    # upward_pass(hA.root, tol, max_rank)

    # -------------------- Evolution Pass --------------------
    # Multiply the hierarchical matrices.
    # prod_root = HMultiply(hA.root, hB.root, tol, max_rank, min_size)

    # Reconstruct the full padded product.
    # prod_padded = reconstruct_dense(prod_root, (m_pad, p_pad))
    # Crop the product back to the original dimensions.
    # result = crop_matrix(prod_padded, (m, p))

    # Regular Mat Mul
    direct = A_orig @ B_orig

    # Compute the normalized error
    error = np.linalg.norm(res - direct) / np.linalg.norm(direct)

    # print("Direct multiplication result:")
    # print(direct)
    # print("Hierarchical multiplication result:")
    # print(result)
    print("Original dimensions: A:", A_orig.shape, "B:", B_orig.shape)
    # print("Padded dimensions: A:", A_padded.shape, "B:", B_padded.shape)
    # print("Result dimensions (after cropping):", result.shape)
    print("Relative error between hierarchical and direct multiplication:", error)
