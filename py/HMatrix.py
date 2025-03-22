import numpy as np
from scipy.linalg import svd


class MatrixNode:

    # Lower and Upper bound
    def __init__(self, row_start, row_end, col_start, col_end):

        self.row_range = (row_start, row_end)
        self.col_range = (col_start, col_end)
        self.children = []
        self.data = None
        self.is_leaf = False  # Indicator of leaf node

    @property
    def shape(self):
        r_start, r_end = self.row_range
        c_start, c_end = self.col_range
        return (r_end - r_start, c_end - c_start)


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
        self.construct_tree(self.root, matrix)
